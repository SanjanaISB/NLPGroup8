import streamlit as st
import requests
from textblob import TextBlob

# Set the title of the Streamlit app
st.title("English to German Translation with Sentiment Analysis")

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
            blob = TextBlob(german_sentence)
            sentiment = blob.sentiment

            st.write("### Sentiment Analysis")
            st.write(f"**Polarity:** {sentiment.polarity}")
            st.write(f"**Subjectivity:** {sentiment.subjectivity}")
        else:
            st.write("Error:", response.status_code)
    else:
        st.write("Please enter an English sentence.")
