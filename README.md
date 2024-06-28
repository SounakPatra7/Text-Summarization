# Text-Summarization
This repository offers a comprehensive suite of tools and models for text summarization, a vital task in natural language processing (NLP) aimed at condensing lengthy documents into concise summaries while preserving key information. Whether you're a researcher exploring cutting-edge techniques or an application developer seeking to integrate text summarization capabilities into your projects, this repository provides a robust foundation.

# Design FlowChart
Lucidchart link : https://lucid.app/lucidchart/39ecab18-c0f7-416b-8eb0-c04fa4d5490b/edit?viewport_loc=-1639%2C-1134%2C5120%2C2228%2C0_0&invitationId=inv_4e413e17-ddf7-4a04-ac40-64be25a16f4a

# Abstractive Summarization

# Data Collection
Dataset selected :

•	Dialoguesum : https://huggingface.co/datasets/knkarthick/dialogsum

# Model Training and Vlidation
For training our text summarization model, I have planned on using the T5/Pegasus models from Hugging Face.

Training

-> Framework: Hugging Face API

-> Model: google/pegasus-cnn_dailymail

-> Objective: Minimize the training loss

# Evaluation Metrics
To evaluate the performance of our model, we need to calculate:

-> Average Training Loss: To monitor how well the model is learning during training.

-> Average Validation Loss: To assess how well the model generalizes to unseen data.

# Performance Metrics
We will use the following metric to validate and evaluate our model:

-> ROUGE (Recall-Oriented Understudy for Gisting Evaluation): This metric helps in measuring the quality of summaries by comparing the overlap of n-grams, word sequences, and word pairs between the generated summary and a reference summary.

# Extractive Summarization
Extractive Text Summarization model using RoBERTa.

# Approach:

1. Convert the articles/passages into a list of sentences using nltk's sentence tokenizer.
  
2. For each sentence, extract contextual embeddings using Sentence transformer.

3. Apply K-means clustering on the embeddings. The idea is to cluster the sentences that are contextually similar to each other & pick one sentence from each cluster that is closest to the mean(centroid).
 
4. For each sentence embedding, calculate the distance from centroid.The distance would be zero if centroid itself is the actual sentence embedding.
   
5. For each cluster, select the sentence embedding with lowest distance from centroid & return the summary based on the order in which the sentences appear in the original text.

# Sentence Transformer

Sentence transformer is a python library that alow us to represent the sentences & paragraphs into dense vectors. This package is compatible with the state of the art models like BERT, RoBERTa, XLM-RoBERTa etc.

# Dataset used :

multi_news : https://huggingface.co/datasets/alexfabbri/multi_news

# Evaluation :
We will use the following metric to validate and evaluate our model:

-> ROUGE (Recall-Oriented Understudy for Gisting Evaluation): This metric helps in measuring the quality of summaries by comparing the overlap of n-grams, word sequences, and word pairs between the generated summary and a reference summary.
