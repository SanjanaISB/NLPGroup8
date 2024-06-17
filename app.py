import os
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import streamlit as st
from transformers import pipeline

# Load the trained model
with open('transformer_model_architecture.json', 'r') as f:
    transformer = keras.models.model_from_json(f.read())
transformer.load_weights('transformer_model.h5')

# Load the vectorization files
with open('source_vectorization.pkl', 'rb') as f:
    source_vectorization = pickle.load(f)

with open('target_vectorization.pkl', 'rb') as f:
    target_vectorization = pickle.load(f)

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

# Initialize sentiment analysis pipeline for German language
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

st.title("English to German Translation and Sentiment Analysis")
st.write("Enter an English sentence and get its German translation along with sentiment analysis.")

input_sentence = st.text_input("English Sentence:")

if st.button("Translate"):
    translated_sentence = decode_sequence(input_sentence)
    st.write("**German Translation:**", translated_sentence)
    sentiment = sentiment_pipeline(translated_sentence)
    st.write("**Sentiment Analysis:**", sentiment)

