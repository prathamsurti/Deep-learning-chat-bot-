import streamlit as st
from chatbot import get_response, predict_class, intents

st.title("Chatbot")

user_input = st.text_input("You: ", "")  # Get user input

if st.button("Send"):  # When the Send button is clicked
    if user_input != "":  # If the input field isn't empty
        # Predict the class of the message
        ints = predict_class(user_input)
        
        # Get the bot's response
        bot_response = get_response(ints, intents)
        
        # Display the bot's response
        st.write("Bot: ", bot_response)
