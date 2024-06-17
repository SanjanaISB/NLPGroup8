import os
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import streamlit as st
from transformers import pipeline

# Ensure the model files are in the correct path
MODEL_PATH = 'transformer_model.h5'
SOURCE_VECTORIZATION_PATH = 'source_vectorization.pkl'
TARGET_VECTORIZATION_PATH = 'target_vectorization.pkl'

# Load the trained model
transformer = keras.models.load_model(MODEL_PATH)

# Load the vectorization files
with open(SOURCE_VECTORIZATION_PATH, 'rb') as f:
    source_vectorization = pickle.load(f)

with open(TARGET_VECTORIZATION_PATH, 'rb') as f:
    target_vectorization = pickle.load(f)

# Define vocabulary and decoding parameters
target_vocab = target_vectorization.get_vocabulary()
target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))
max_decoded_sentence_length = 30

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break

    # Remove [start] and [end] tokens
    decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
    return decoded_sentence

# Initialize sentiment analysis pipeline for the German language
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

# Streamlit app interface
st.title("English to German Translation and Sentiment Analysis")
st.write("Enter an English sentence and get its German translation along with sentiment analysis.")

input_sentence = st.text_input("English Sentence:")

if st.button("Translate"):
    if input_sentence:
        translated_sentence = decode_sequence(input_sentence)
        st.write("**German Translation:**", translated_sentence)
        sentiment = sentiment_pipeline(translated_sentence)
        st.write("**Sentiment Analysis:**", sentiment)
    else:
        st.write("Please enter an English sentence.")
