# Vision-Transformer-from-Scratch-PyTorch-
Vision Transformer from Scratch | PyTorch 🔥
Vision Transformer from Scratch | PyTorch 🔥
Vision Transformer from Scratch | PyTorch 🔥


# Overview

The transformer is a general and powerful neural network architecture, able to tackle many real-world problems. Vision transformer (ViT) is a transformer for computer vision tasks. In this notebook, Vision Transformer (ViT) is implemented from scratch using PyTorch for image classification. Later, we will train the model on a subset of RSNA breast cancer detection dataset.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. About the Architecture
Transformers found their initial applications in natural language processing (NLP) tasks. To use this NLP model for computer vision tasks, we have to divide our input image into patches. After flattening the patches, we can treat each flattened patches as single word. We add positional embeddings to the linear projection of flattened patches. An extra token is added at the beginning for classification tasks. In BERT model, this token is called [CLS] token.

So if our input image size is (512, 512), after dividing the image into patches of size (16, 16), we get 1024 (32 times 32) patches. After flattening the patches and projecting the flattened patches, we have 1024 tokens. After adding positional embeddings and concatenating classification token at the beginning, we have 1025 tokens.

We then feed our tokens into the transformer encoder. Transformer encoder is made up of self attention and feedforward network. The [original paper](https://paperswithcode.com/paper/attention-is-all-you-need) on attention is an excellent read if you want to understand the whole attention mechanism. PyTorch have `torch.nn.MultiHeadAttention` for anyone who one to use attention mechanism for their next project. This [video](https://www.youtube.com/watch?v=_UVfwBqcnbM) by AssemblyAI is explains the transformer architecture beautifully.

The number of tokens in the output of the transformer encoder is equal to number of input tokens. We take the first token from the output (corresponds to the classification token) and feed the token in a multilayer perceptron head for classification.

For more details, you can go through [original paper](https://paperswithcode.com/method/vision-transformer) on Vision Transformer.

![Vision Transformer](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)
