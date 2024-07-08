"""
Code to interact with OpenAI's GPT-3.5 Turbo API.

functions
---------
set_prompt_template()
    Set the prompt template for the chatbot.
streamlit_setup(chain)
    Setup the Streamlit app.
main()
    Main function to run the utility.
"""

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def set_prompt_template():
    """
    Set the prompt template for the chatbot.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant, that knows a lot about programming languages.You would give concise answers for queries and write optimised code snippets.",
            ),
            ("user", "{user_input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain


def streamlit_setup(chain):
    """
    Setup the Streamlit app.

    Parameters
    ----------
    chain :
        The LangChain object.
    """

    st.title("LangChain: Programming Language Assistant")
    st.write(
        "LangChain is a programming language assistant that can help you with your programming queries and write code snippets for you."
    )

    st.write("Please enter your query below:")
    user_input = st.text_area("Enter your query here:")

    if st.button("Submit"):
        st.write(chain.invoke(user_input))


def main():
    """
    Main function to run the utility.
    """
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    chain = set_prompt_template()
    streamlit_setup(chain)


if __name__ == "__main__":
    main()
