# Amazon Product Review NLP Capstone

## Project Overview

For this capstone project, I built an end to end NLP and machine learning pipeline focused on analyzing Amazon customer reviews.

The goal of the project was to explore how ecommerce companies can use NLP, machine learning, and deep learning techniques to better understand customer feedback at scale.

Using Amazon review data  I cleaned and processed raw review text. Then I  performed exploratory data analysis as well as engineered NLP features for going thru the words. Then I  trained multiple machine learning and deep learning models. Lastly I  built a Gradio application capable of generating customer review summaries.

This project combines:
- NLP preprocessing
- sentiment analysis
- TF-IDF feature engineering
- machine learning classification
- regression modeling
- deep learning with LSTM/GRU
- transformer-based text summarization
- lightweight AI application deployment

The overall objective was to turn large amounts of unstructured customer feedback into meaningful business insights and automated review analysis tools.


# Dataset

The dataset contains both structured ecommerce product data and unstructured customer review text.

## Structured Features

- product_id
- product_name
- category
- actual_price
- discounted_price
- discount_percentage
- rating
- rating_count
- user_name

## Unstructured Features

- review_title
- review_content
- about_product

The review text data was used heavily throughout the NLP and modeling pipeline.


# Project Pipeline

## 1. Data Preprocessing & Exploratory Data Analysis

The first stage of the project focused on cleaning and preparing the review data for NLP analysis and machine learning workflows.

This included:
- handling missing values
- cleaning review text
- removing HTML tags and URLs
- stopword removal
- tokenization
- lemmatization
- sentiment scoring using VADER

I also performed exploratory data analysis to better understand:
- rating distributions
- customer sentiment patterns
- pricing behavior
- review engagement
- product category trends

I created a few visualizations including sentiment distributions, word clouds, and product engagement charts. This was mainly to illustrate my findings.


## 2. Feature Engineering & Machine Learning

To prepare the review data for machine learning models, I engineered several NLP and business focused features, including:
- review length
- sentiment polarity
- encoded categorical variables
- price drop percentage

TF-IDF vectorization was used to transform customer review text into number representations. 

### Classification Models

The following models were trained for sentiment classification:
- Logistic Regression
- Decision Tree
- Random Forest

### Regression Models

The following regression models were trained to predict product ratings:
- Linear Regression
- Ridge Regression
- Lasso Regression

Among the regression models, Ridge Regression produced the strongest overall performance.

---

## 3. Deep Learning & Transformer NLP

To extend the project beyond traditional machine learning, I implemented sequential deep learning models for sentiment analysis.

### Deep Learning Models
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

These models were trained on tokenized customer review sequences to learn contextual language patterns from review text.

I also implemented transformer based review summarization using GPT-2 to generate abstractive summaries from multiple customer reviews.

---

# Gradio Review Summarization App

As the final stage of the project, I built a Gradio-based NLP application that allows users to:
- input multiple customer reviews
- generate extractive summaries
- generate transformer based abstractive summaries

The app demonstrates how NLP systems can be packaged into lightweight interactive AI tools for real world use cases.


# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- HuggingFace Transformers
- NLTK
- BeautifulSoup
- Matplotlib
- Seaborn
- Gradio



# Key Insights

One of the biggest findings from this project was how heavily imbalanced customer review datasets can be. Most reviews in the dataset were highly positive, which created challenges for classification performance and model generalization.

The project also demonstrated how effective NLP preprocessing and TF-IDF vectorization can be for sentiment analysis tasks, even with relatively lightweight machine learning models.

The deep learning portion of the project showed how LSTM and GRU architectures can learn sequential language patterns from review text, while transformer-based summarization demonstrated the potential for automatically condensing large volumes of customer feedback into concise summaries.



# Business Applications

This type of NLP pipeline could help ecommerce companies:
- monitor customer satisfaction
- identify product quality issues
- analyze large-scale review trends
- automate customer feedback analysis
- improve recommendation systems
- generate AI-powered review summaries

