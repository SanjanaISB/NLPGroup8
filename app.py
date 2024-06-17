import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.layers import Layer

# Unified MultiHeadAttention class
class MultiHeadAttention(Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} should be divisible by the number of heads {num_heads}"
            )
        
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def concat_heads(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.embed_dim))

    def call(self, q, k, v, use_causal_mask=False):
        batch_size = tf.shape(k)[0]
        q = self.query_dense(q)
        k = self.key_dense(k)
        v = self.value_dense(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        attention = scaled_dot_product_attention(q, k, v, use_causal_mask)
        concat = self.concat_heads(attention, batch_size)
        output = self.combine_heads(concat)
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

def scaled_dot_product_attention(q, k, v, use_causal_mask=False):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if use_causal_mask:
        mask = tf.linalg.band_part(tf.ones_like(scaled_attention_logits), -1, 0)
        scaled_attention_logits += tf.math.log(mask * 1e-6 + 1.)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output

# Load vectorization objects
with open('source_vectorization.pkl', 'rb') as f:
    source_vectorization = pickle.load(f)

with open('target_vectorization.pkl', 'rb') as f:
    target_vectorization = pickle.load(f)

# Load the model architecture and weights
with open('transformer_model_architecture.json', 'r') as f:
    model_json = f.read()

transformer = tf.keras.models.model_from_json(model_json, custom_objects={
    'PositionalEmbedding': PositionalEmbedding,
    'custom_standardization': custom_standardization,
    'MultiHeadAttention': MultiHeadAttention,
    'TransformerEncoder': TransformerEncoder,
    'TransformerDecoder': TransformerDecoder
})

transformer.load_weights('transformer_model.h5')

# Define max decoded sentence length
max_decoded_sentence_length = 30

# Create target index lookup from target_vectorization
target_index_lookup = {i: token for i, token in enumerate(target_vectorization.get_vocabulary())}

# Function to decode sequence using Transformer model
def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence], training=False)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
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
