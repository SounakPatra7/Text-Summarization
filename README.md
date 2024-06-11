# Text-Summarization
This repository offers a comprehensive suite of tools and models for text summarization, a vital task in natural language processing (NLP) aimed at condensing lengthy documents into concise summaries while preserving key information. Whether you're a researcher exploring cutting-edge techniques or an application developer seeking to integrate text summarization capabilities into your projects, this repository provides a robust foundation.

# Design FlowChart
Lucidchart link : https://lucid.app/lucidchart/39ecab18-c0f7-416b-8eb0-c04fa4d5490b/edit?viewport_loc=-1639%2C-1134%2C5120%2C2228%2C0_0&invitationId=inv_4e413e17-ddf7-4a04-ac40-64be25a16f4a

# Data Collection
Datasets selected :
•	bbc-news-summary : https://huggingface.co/datasets/gopalkalpande/bbc-news-summary

•	Samsum : https://huggingface.co/datasets/Samsung/samsum

•	Dialoguesum : https://huggingface.co/datasets/knkarthick/dialogsum

# Model Training and Vlidation
For training our text summarization model, I have planned on using the T5/Pegasus models from Hugging Face.
Training
-> Framework: Hugging Face API

-> Model: T5/Pegasus

-> Objective: Minimize the training loss

# Evaluation Metrics
To evaluate the performance of our model, we need to calculate:
-> Average Training Loss: To monitor how well the model is learning during training.

-> Average Validation Loss: To assess how well the model generalizes to unseen data.

# Performance Metrics
We will use the following metric to validate and evaluate our model:
-> ROUGE (Recall-Oriented Understudy for Gisting Evaluation): This metric helps in measuring the quality of summaries by comparing the overlap of n-grams, word sequences, and word pairs between the generated summary and a reference summary.
