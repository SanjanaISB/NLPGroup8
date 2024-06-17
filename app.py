import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from transformers import pipeline

# Load the trained Transformer model and vectorizers
@st.cache_resource(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('transformer_model.h5', custom_objects={
        'PositionalEmbedding': PositionalEmbedding,
        'TransformerEncoder': TransformerEncoder,
        'TransformerDecoder': TransformerDecoder,
        'MultiHeadAttention': MultiHeadAttention,
        'attention_mask': attention_mask,
        'mask_attn_weights': mask_attn_weights,
        'scaled_dot_product_attention': scaled_dot_product_attention,
        'shape_list': shape_list
    })
    return model

@st.cache_resource(allow_output_mutation=True)
def load_vectorizers():
    with open('source_vectorization.pkl', 'rb') as f:
        source_vectorization = pickle.load(f)
    with open('target_vectorization.pkl', 'rb') as f:
        target_vectorization = pickle.load(f)
    return source_vectorization, target_vectorization

transformer = load_model()
source_vectorization, target_vectorization = load_vectorizers()

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

target_vocab = target_vectorization.get_vocabulary()
target_index_lookup = {index: word for index, word in enumerate(target_vocab)}
max_decoded_sentence_length = 30

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for _ in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, _])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
    return decoded_sentence

st.title("English to German Translation with Sentiment Analysis")

input_sentence = st.text_input("Enter an English sentence:")
if input_sentence:
    translated_sentence = decode_sequence(input_sentence)
    st.write("**Translated Sentence in German:**")
    st.write(translated_sentence)

    sentiment = sentiment_pipeline(translated_sentence)
    st.write("**Sentiment Analysis of the Translated Sentence:**")
    st.write(sentiment)
