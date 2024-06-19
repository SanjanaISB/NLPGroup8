import streamlit as st
import requests
from flair.models import TextClassifier
from flair.data import Sentence

# Load the Flair sentiment analysis model
classifier = TextClassifier.load('sentiment')

# Set the title of the Streamlit app
st.title("English to German Translation with Sentiment Analysis")

# Text input for the English sentence
english_sentence = st.text_input("Enter an English sentence:")

def translate_sentence(english_sentence):
    response = requests.post(
        "https://5185-35-247-172-177.ngrok-free.app/translate",
        json={"sentence": english_sentence}
    )
    if response.status_code == 200:
        data = response.json()
        return data['german_sentence']
    else:
        st.write("Error in translation:", response.status_code)
        return None

def analyze_sentiment(german_sentence):
    sentence = Sentence(german_sentence)
    classifier.predict(sentence)
    sentiment = sentence.labels[0]
    return sentiment

if st.button("Translate and Analyze Sentiment"):
    if english_sentence:
        german_sentence = translate_sentence(english_sentence)
        if german_sentence:
            st.write("### Translation")
            st.write(f"**German Sentence:** {german_sentence}")

            # Perform sentiment analysis on the German sentence
            sentiment = analyze_sentiment(german_sentence)

            st.write("### Sentiment Analysis")
            st.write(f"**Sentiment:** {sentiment.value}")
            st.write(f"**Confidence:** {sentiment.score:.2f}")
        else:
            st.write("Failed to translate the sentence.")
    else:
        st.write("Please enter an English sentence.")
