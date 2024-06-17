import re
import tensorflow as tf
import numpy as np
import streamlit as st
import pickle
from tensorflow.keras.layers import TextVectorization

# Define the custom standardization function
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'), '')

# Register the custom standardization function
@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'), '')

# Load vectorization objects
with open('source_vectorization.pkl', 'rb') as f:
    source_vectorization = pickle.load(f)

with open('target_vectorization.pkl', 'rb') as f:
    target_vectorization = pickle.load(f)

# Load Transformer model
transformer = tf.keras.models.load_model('transformer_model.h5')

# Define max decoded sentence length
max_decoded_sentence_length = 30

# Create target index lookup from target_vectorization
target_index_lookup = {v: k for k, v in enumerate(target_vectorization.get_vocabulary())}

# Function to decode sequence using Transformer model
def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup.get(sampled_token_index, '[unk]')
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break

    # Remove [start] and [end] tokens
    decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
    return decoded_sentence

# Streamlit UI
st.title('English to German Translation')

input_text = st.text_input('Enter English sentence:')
if st.button('Translate'):
    if input_text:
        translated_sentence = decode_sequence(input_text)
        st.success(f'Translated German sentence: {translated_sentence}')
    else:
        st.warning('Please enter an English sentence to translate.')
