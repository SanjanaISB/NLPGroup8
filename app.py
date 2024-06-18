import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
import pickle
import numpy as np
import streamlit as st
from transformers import pipeline
import gdown

# Define the custom layer PositionalEmbedding
class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, d_model, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_emb = keras.layers.Embedding(input_dim=2048, output_dim=d_model)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        vocab_size = config.pop('vocab_size')
        d_model = config.pop('d_model')
        return cls(vocab_size=vocab_size, d_model=d_model, **config)

# Ensure the model files are in the correct path
MODEL_URL = 'https://drive.google.com/uc?export=download&id=18PvjVRNcJ50CqUKJ-sNILVovIa5vJUj3'
SOURCE_VECTORIZATION_PATH = 'source_vectorization.pkl'
TARGET_VECTORIZATION_PATH = 'target_vectorization.pkl'

# Function to download the model file using gdown
def download_model(url, filename):
    gdown.download(url, filename, quiet=False)

# Download the model
MODEL_PATH = 'transformer_model.h5'
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# Load the trained model with custom object scope
try:
    with keras.utils.custom_object_scope({'PositionalEmbedding': PositionalEmbedding}):
        transformer = keras.models.load_model(MODEL_PATH)
except Exception as e:
    transformer = None
    st.error(f"Error loading the model: {e}")

# Load the vectorization files
try:
    with open(SOURCE_VECTORIZATION_PATH, 'rb') as f:
        source_vectorization = pickle.load(f)
except Exception as e:
    source_vectorization = None
    st.error(f"Error loading source vectorization: {e}")

try:
    with open(TARGET_VECTORIZATION_PATH, 'rb') as f:
        target_vectorization = pickle.load(f)
except Exception as e:
    target_vectorization = None
    st.error(f"Error loading target vectorization: {e}")

# Ensure both vectorizations are loaded before proceeding
if source_vectorization and target_vectorization:
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
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
    except Exception as e:
        sentiment_pipeline = None
        st.error(f"Error initializing sentiment analysis pipeline: {e}")

    # Streamlit app interface
    st.title("English to German Translation and Sentiment Analysis")
    st.write("Enter an English sentence and get its German translation along with sentiment analysis.")

    input_sentence = st.text_input("English Sentence:")

    if st.button("Translate"):
        if input_sentence:
            translated_sentence = decode_sequence(input_sentence)
            st.write("**German Translation:**", translated_sentence)
            if sentiment_pipeline:
                try:
                    sentiment = sentiment_pipeline(translated_sentence)
                    st.write("**Sentiment Analysis:**", sentiment)
                except Exception as e:
                    st.error(f"Error during sentiment analysis: {e}")
        else:
            st.write("Please enter an English sentence.")
else:
    st.error("Failed to load necessary vectorization files.")
