import streamlit as st
import requests
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

# Set the title of the Streamlit app
st.title("English to German Translation with Sentiment Analysis")

# Text input for the English sentence
english_sentence = st.text_input("Enter an English sentence:")

def translate_sentence(english_sentence):
    response = requests.post(
        "https://380f-34-83-77-44.ngrok-free.app/translate",
        json={"sentence": english_sentence}
    )
    if response.status_code == 200:
        data = response.json()
        return data['german_sentence']
    else:
        st.write("Error in translation:", response.status_code)
        return None

if st.button("Translate and Analyze Sentiment"):
    if english_sentence:
        german_sentence = translate_sentence(english_sentence)
        if german_sentence:
            st.write("### Translation")
            st.write(f"**German Sentence:** {german_sentence}")

            # Perform sentiment analysis on the German sentence
            sentiment = sentiment_pipeline(german_sentence)[0]

            st.write("### Sentiment Analysis")
            st.write(f"**Sentiment:** {sentiment['label']}")
            st.write(f"**Confidence:** {sentiment['score']:.2f}")
        else:
            st.write("Failed to translate the sentence.")
    else:
        st.write("Please enter an English sentence.")
