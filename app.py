import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle

# Load the trained Transformer model
model_path = 'transformer_model.h5'
transformer = keras.models.load_model(model_path, custom_objects={"PositionalEmbedding": PositionalEmbedding, "MultiHeadAttention": MultiHeadAttention, "TransformerEncoder": TransformerEncoder, "TransformerDecoder": TransformerDecoder})

# Load the vectorization configurations
with open('source_vectorization.pkl', 'rb') as f:
    source_vectorization = pickle.load(f)

with open('target_vectorization.pkl', 'rb') as f:
    target_vectorization = pickle.load(f)

# Define decode function
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

    decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
    return decoded_sentence

# Streamlit app layout
st.title("English to German Translator")

st.write("Enter an English sentence and click 'Translate' to get the German translation.")

input_sentence = st.text_input("English Sentence")

if st.button("Translate"):
    if input_sentence:
        translated_sentence = decode_sequence(input_sentence)
        st.write("**German Translation:**")
        st.write(translated_sentence)
    else:
        st.write("Please enter an English sentence.")
