import streamlit as st
import requests
from transformers import pipeline

# Set the title of the Streamlit app
st.title("English to German Translation with Sentiment Analysis")

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

# Text input for the English sentence
english_sentence = st.text_input("Enter an English sentence:")

if st.button("Translate and Analyze Sentiment"):
    if english_sentence:
        # Make a POST request to the FastAPI endpoint
        response = requests.post(
            "https://5185-35-247-172-177.ngrok-free.app/translate",
            json={"sentence": english_sentence}
        )

        if response.status_code == 200:
            data = response.json()
            german_sentence = data['german_sentence']
            st.write("### Translation")
            st.write(f"**German Sentence:** {german_sentence}")

            # Perform sentiment analysis on the German sentence
            sentiment = sentiment_pipeline(german_sentence)[0]
            st.write("### Sentiment Analysis")
            st.write(f"**Sentiment:** {sentiment['label']}")
            st.write(f"**Confidence:** {sentiment['score']:.2f}")
        else:
            st.write("Error:", response.status_code)
    else:
        st.write("Please enter an English sentence.")
