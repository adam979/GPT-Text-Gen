# GPT Text Generation with KerasNLP

## Project Overview

Welcome to the GPT Text Generation project with KerasNLP! This project showcases the implementation of a mini-GPT (Generative Pre-trained Transformer) model for text generation using the KerasNLP library. GPT is renowned for its ability to generate coherent and contextually relevant text given a prompt.

We trained the model on the simplebooks-92 corpus, a dataset comprising 1,573 Gutenberg books. This dataset, chosen for its small vocabulary and high word frequency, is ideal for training a model with limited parameters, providing an efficient demonstration of the GPT architecture.

In this comprehensive project, we delve into various aspects:

- **Data setup and preprocessing**
- **Tokenization using KerasNLP**
- **Building a mini-GPT model**
- **Training the model**
- **Text generation using various techniques**

*Note: If you're running this project on Google Colab, consider enabling GPU runtime for faster training (not necessarily below 10 epochs).*

## Setup

Ensure you have KerasNLP installed before starting. You can install it with the following command:

```bash
pip install keras-nlp
````
## Data Loading

We downloaded the simplebooks-92 dataset, crafted from 1,573 Gutenberg books. The dataset's small vocabulary and high word frequency make it a perfect fit for our training. We diligently loaded and preprocessed both the training and validation sets to ensure a robust model.

## Tokenization

Tokenization, a pivotal step in our project, involves training a tokenizer on the dataset to create a sub-word vocabulary. This step is crucial for converting raw text into a format that our model can comprehend and generate meaningful responses.

## Model Building

Our mini-GPT model, exclusively built with KerasNLP, boasts essential components:

- Token and Position Embedding layer
- Multiple TransformerDecoder layers
- A final dense linear layer

We fine-tuned the model's hyperparameters, including batch size, sequence length, and embedding dimensions, ensuring optimal performance for our specific task.

## Training

The model underwent training on the prepared dataset. Validation on a separate dataset ensures robust performance. You have the flexibility to adjust the number of epochs, allowing for fine-tuning tailored to your specific requirements.

## Text Generation

Experience diverse text generation techniques using our trained model:

- **Greedy Search:** Selecting the most probable token at each step.
- **Beam Search:** Considering multiple probable sequences to reduce repetition.
- **Random Search:** Sampling tokens based on softmax probabilities.
- **Top-K Search:** Sampling from the top-K most probable tokens.
- **Top-P (Nucleus) Search:** Sampling based on a dynamic probability threshold.

We've demonstrated how to generate text using these methods, providing insights into their advantages and limitations.

## Future Directions

To further advance our understanding of Transformers, we plan to explore training full-sized GPT models.