<p align="center"><img src="https://github.com/user-attachments/assets/0f309e0e-dbbe-4109-8167-84ddbbb03a53" width="200" height="100"></p>
# Text-Summarization
This repository offers a comprehensive suite of tools and models for text summarization, a vital task in natural language processing (NLP) aimed at condensing lengthy documents into concise summaries while preserving key information. Whether you're a researcher exploring cutting-edge techniques or an application developer seeking to integrate text summarization capabilities into your projects, this repository provides a robust foundation.

## Abstractive Summarization

### Data Collection
Dataset selected

•	Dialoguesum : https://huggingface.co/datasets/knkarthick/dialogsum

### Model Training and Validation
For training our text summarization model, I have planned on using the T5/Pegasus models from Hugging Face.

Training

-> Framework: Hugging Face API

-> Model: google/pegasus-cnn_dailymail

-> Objective: Minimize the training loss

-> saved_model, file structure:
```
└── fine_tuned_pegasus

    ├── config.json

    ├── generation_config.json

    ├── model.safetensors

    ├── special_tokens_map.json

    ├── tokenizer.json

    ├── tokenizer_config.json

    └── spiece.model
```

### Evaluation Metrics
To evaluate the performance of our model, we need to calculate:

-> Average Training Loss: To monitor how well the model is learning during training.

-> Average Validation Loss: To assess how well the model generalizes to unseen data.

### Performance Metrics
We will use the following metric to validate and evaluate our model:

-> ROUGE (Recall-Oriented Understudy for Gisting Evaluation): This metric helps in measuring the quality of summaries by comparing the overlap of n-grams, word sequences, and word pairs between the generated summary and a reference summary.

## Extractive Summarization
Extractive Text Summarization model using RoBERTa.

### Approach:

1. Convert the articles/passages into a list of sentences using nltk's sentence tokenizer.
  
2. For each sentence, extract contextual embeddings using Sentence transformer.

3. Apply K-means clustering on the embeddings. The idea is to cluster the sentences that are contextually similar to each other & pick one sentence from each cluster that is closest to the mean(centroid).
 
4. For each sentence embedding, calculate the distance from centroid.The distance would be zero if centroid itself is the actual sentence embedding.
   
5. For each cluster, select the sentence embedding with lowest distance from centroid & return the summary based on the order in which the sentences appear in the original text.

### Modification

-> In place of considering only the distance from centroid scores, TF-IDF Vectorization is implemented and we are calculating the TF-IDF scores too. As a result, we are taking the combined score of these two we are considering those sentence embeddings and generated a more accurate summary.

### Sentence Transformer

Sentence transformer is a python library that alow us to represent the sentences & paragraphs into dense vectors. This package is compatible with the state of the art models like BERT, RoBERTa, XLM-RoBERTa etc.

### Dataset used :

multi_news : https://huggingface.co/datasets/alexfabbri/multi_news

### Evaluation :
We will use the following metric to validate and evaluate our model:

-> ROUGE (Recall-Oriented Understudy for Gisting Evaluation): This metric helps in measuring the quality of summaries by comparing the overlap of n-grams, word sequences, and word pairs between the generated summary and a reference summary.

## Interface 
Made an demo interface using Gradio which gives functionality to choose between which type of summarization we want to perform.

Public URL: https://66b8dafd039b11fad5.gradio.live (This share link expires in 72 hours)

## Application 
Python-based web application for text summarization using FastAPI and Hugging Face's Transformers library. The application provides both abstractive and extractive summarization methods and includes a user-friendly interface for inputting text and viewing summaries.

### Features :
-> Abstractive Summarization: Utilizes a pre-trained Seq2Seq model for generating summaries based on input text.

-> Extractive Summarization: Implements an algorithm to extract key sentences from the input text to form a summary.

-> User Interface: A clean and intuitive web interface powered by FastAPI and styled with HTML and CSS.

-> Error Handling: Robust error handling to manage cases such as empty input and provide clear feedback to users.

-> Performance: Optimized for speed and efficiency, leveraging GPU acceleration where available to minimize summarization time.

-> Deployment: Easily deployable using Docker, ensuring consistency across different environments.

### Usage :
-> Enter text in the input box.

-> Choose the type of summarization (abstractive or extractive).

-> Click the "Summarize" button to generate a summary.

-> Use the "Clear" button to reset the input and summary sections.

### Technologies used : 
-> Python

-> FastAPI

-> Hugging Face Transformers

-> HTML/CSS

-> Docker

## Report
Completed the report. Detailed information about the project is discussed in the Report.

## Presentation

Canva Link : https://www.canva.com/design/DAGK8VV_Sq8/gChyF0hYFwnS9la0wV2Acg/view?utm_content=DAGK8VV_Sq8&utm_campaign=designshare&utm_medium=link&utm_source=editor
