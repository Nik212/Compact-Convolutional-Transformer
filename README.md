# Compact-Convolutional-Transformer

This is a simple implementation of Compact Convolution Transformer (original paper [Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/pdf/2104.05704.pdf)) for cifar10 images classification task. 

This task was done as a part of home assignments for course "Deep Vision and Graphics" at Skolkovo Institute of Science and Technology. Original .ipynb notebook template belongs the course authors.


### **Usage**

- Experiments are contained in the .ipynb notebook. It also contains the code to download the dataset.
- All implementations are contained in .py file, including: custom ```MultiHeadSelfAttention``` module, ```StepLRWithWarmup``` scheduler and ```DropPath``` regularizer.
- A checkpoint for the model with **84.59 val accuracy** is provided

### Requirements
```pytorch``` 1.10.1

```torchvision``` 0.11.2
