import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import warnings

class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device='cuda:0', dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim) 
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm_coeff = self.head_dim ** 0.5

        self.attention_dropout = nn.Dropout(dropout)
        self.device = device
        self.head_dim = embed_dim // num_heads

    def forward(self, qkv):
        """
        qkv - query, key and value - it should be the same tensor since we implement self-attention
        """
        # 1. apply self.in_proj to qkv
        # 2. split the result of step 1 on three equal parts of size self.embed_dim: query, key, value
        # 3. compute scaled dot-product attention for different heads in loop.
        #    i-th head will work on query[:, :, i*head_dim: (i+1)*head_dim],
        #    key[:, :, i*head_dim: (i+1)*head_dim], value[:, :, i*head_dim: (i+1)*head_dim]
        # 4. apply dropout for each head result
        # 5. concat all results
        # 6. apply self.out_proj to the result of step 5
        qkv = self.in_proj(qkv)
        query, key, value = torch.chunk(qkv, 3, dim = 2)
        heads = torch.tensor([]).to(self.device)
        for i in range(self.num_heads):
            q_i = query[:, :, i*self.head_dim : (i+1)*self.head_dim]
            k_i = key[:, :, i*self.head_dim : (i+1)*self.head_dim]
            v_i = value[:, :, i*self.head_dim : (i+1)*self.head_dim]
            normalized_weights = torch.functional.F.softmax(q_i @ torch.transpose(k_i, 1, 2) / q_i.size(-1)**(1/2), dim = -1)
            head_i = normalized_weights @ v_i
            head_i = self.attention_dropout(head_i)
            heads = torch.cat([heads, head_i], dim = 2)
        return self.out_proj(heads)


class StepLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_epochs=3, warmup_lr_init=1e-5,
                 min_lr=1e-5,
                 last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0):
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        # YOUR CODE
        elif (self.last_epoch in range(0, self.warmup_epochs)):
            return [(base_lr - self.warmup_lr_init)/self.warmup_epochs * self.last_epoch + self.warmup_lr_init for base_lr in self.base_lrs]
        elif (self.last_epoch == self.warmup_epochs):
            return self.base_lrs
        elif (self.last_epoch - self.warmup_epochs) % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif ((self.last_epoch - self.warmup_epochs) % self.step_size == 0):
                return [group['lr']*self.gamma if group['lr']*self.gamma >= self.min_lr else group['lr'] for group in self.optimizer.param_groups]


class TokenizerCCT(nn.Module):
    def __init__(self,
                 kernel_size=3, stride=1, padding=1,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 ):
        super().__init__()
        self.tokenizer_layers = nn.Sequential(
                # Conv2d(n_input_channels, n_output_channels, kernel_size, stride, padding, bias=False) +
                #   + ReLU +
                #   + MaxPool(pooling_kernel_size, pooling_stride, pooling_padding)
                nn.Conv2d(n_input_channels, n_output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding))

        self.flattener = nn.Flatten(2, 3)  # flat h,w dims into token dim

    def forward(self, x):
        y = self.tokenizer_layers(x)
        y = self.flattener(y)
        y = y.transpose(-2, -1)  # swap token dim and embedding dim
        return y


class SeqPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention_pool = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # 1. apply self.attention_pool to x
        w = self.attention_pool(x)
        # 2. take softmax over the first dim (token dim)
        w = F.softmax(w, dim=1)
        # 3. transpose two last dims of w to make its shape be equal to [N, 1, n_tokens]
        w = torch.transpose(w, 1, 2)

        # 4. call torch.matmul from 'w' and input tensor 'x'
        y = w @ x

        # 5. now 'y' shape is [N, 1, embedding_dim]. Squeeze the second dim
        y = y.squeeze(1)

        return y

def create_mlp(embedding_dim, mlp_size, dropout_rate):
    return nn.Sequential(
        nn.Linear(embedding_dim, mlp_size),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(mlp_size, embedding_dim),
        nn.Dropout(dropout_rate)
    )


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, device='cuda:0'):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.device = device

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        #generate random tensor, binarize it, cast to x.dtype, multiply x by the mask, 
        z = torch.rand(shape, dtype=x.dtype).to(self.device)
        random_tensor = z > self.drop_prob
        random_tensor = random_tensor > self.drop_prob
        output = random_tensor.type(x.dtype) * x / keep_prob
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_size, dropout=0.1, attention_dropout=0.1,
                 drop_path_rate=0.1, device = 'cuda:0'):
        super().__init__()
        # YOUR CODE
        self.attention_pre_norm = nn.LayerNorm(embedding_dim)
        self.attention = CustomMultiHeadSelfAttention(
                embedding_dim, num_heads, device, attention_dropout)
        self.attention_output_dropout = nn.Dropout(dropout)

        self.mlp_pre_norm = nn.LayerNorm(embedding_dim)
        self.mlp = create_mlp(embedding_dim, mlp_size, dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        # first block
        y = self.attention_pre_norm(x)
        attention = self.attention(y)[0]
        attention = self.attention_output_dropout(attention)
        x = x + self.drop_path(attention)   # Residual connection
            
        # second block
        y = self.mlp_pre_norm(x)
        y = self.mlp(y)
        x = x + self.drop_path(y)  # Residual connection
        return x


class CompactConvTransformer3x1(nn.Module):
    def __init__(self,
                 input_height, input_width,
                 n_tokens,
                 n_input_channels,
                 embedding_dim,
                 num_layers,
                 num_heads=4,
                 num_classes=10,
                 mlp_ratio=2,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1):
        super().__init__()

        # 1. Tokenizer
        pooling_stride = 2
        self.tokenizer = TokenizerCCT(kernel_size=3, stride=1, padding=1,
                                      pooling_kernel_size=3, pooling_stride=pooling_stride, pooling_padding=1,
                                      n_output_channels=embedding_dim)
        n_tokens = input_height // pooling_stride

        # 2. Positional embeddings
        self.positional_embeddings = torch.nn.Parameter(
            torch.empty((1, n_tokens * n_tokens, embedding_dim)), requires_grad=True)
        torch.nn.init.trunc_normal_(self.positional_embeddings, std=0.2)

        # 3. TransformerEncoder with DropPath
        mlp_size = int(embedding_dim * mlp_ratio)
        layers_drop_path_rate = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = nn.Sequential(*[
            TransformerEncoder(
                embedding_dim, num_heads, mlp_size,
                dropout=dropout, attention_dropout=attention_dropout,
                drop_path_rate=layers_drop_path_rate[i])
            for i in range(num_layers)])

        # 4. normalization before pooling
        self.norm = nn.LayerNorm(embedding_dim)

        # 5. sequence pooling
        self.pool = SeqPooling(embedding_dim)

        # 6. layer for the final prediction
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # 1. apply tokenizer to x
        patch_embeddings = self.tokenizer(x)

        # 2. add position embeddings
        x = patch_embeddings + self.positional_embeddings

        # 3. apply transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        # 4. apply self.norm
        x = self.norm(x)

        # 5. apply sequence pooling
        x = self.pool(x)

        # 6. final prediction
        x = self.fc(x)
        return x