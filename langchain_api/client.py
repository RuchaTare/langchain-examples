"""
Client server for the Langchain API.
"""

import requests
import streamlit as st
import logging


def get_openai_response(input_text: str):
    """
    Get the OpenAI response for the given prompt.

    Parameters
    ----------
    input_text : str
        The prompt to get the response for

    Returns
    -------
    str
        The OpenAI response
    """

    logging.info("Getting OpenAI response for the prompt")

    response = requests.post(
        "http://localhost:8000/essay/invoke", json={"input": {"topic": input_text}}
    )
    return response.json()["output"]


def get_ollama_response(input_text1: str):
    """
    Get the llama2 response for the given prompt.

    Parameters
    ----------
    input_text : str
        The prompt to get the response for

    Returns
    -------
    str
        The llama2 response
    """

    logging.info("Getting llama2 response for the prompt")

    response = requests.post(
        "http://localhost:8000/poem/invoke", json={"input": {"topic": input_text1}}
    )
    return response.json()["output"]


def setup_streamlit():
    """
    Setup the Streamlit app.
    """

    logging.info("Setting up Streamlit app")

    st.title("Langchain API Client")

    input_text = st.text_input("Write an essay on")

    input_text1 = st.text_input("Write a poem on")

    if input_text:
        st.write(get_openai_response(input_text))

    if input_text1:
        st.write(get_ollama_response(input_text1))


def main():
    """
    Main function for the client server.
    """

    logging.info("Starting client server")

    setup_streamlit()


if __name__ == "__main__":
    main()
