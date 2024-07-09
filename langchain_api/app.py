"""
Utility to use
"""

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()


def create_api() -> FastAPI:
    api = FastAPI(title="Langchain API Server", version="1.0", description="Test server ")
    return api


def create_models():
    openai_model = ChatOpenAI()
    llm_model = Ollama(model="llama2")
    return openai_model, llm_model


def setup_routes(api: FastAPI, openai_model: ChatOpenAI, llm_model: Ollama):
    prompt1 = ChatPromptTemplate.from_template(
        "Explain the concept {topic} with 5 bullet points only "
    )
    prompt2 = ChatPromptTemplate.from_template("Write me an essay about {topic} using 500 words.")

    add_routes(api, prompt1 | openai_model, path="/essay")
    add_routes(api, prompt2 | llm_model, path="/poem")


def main():
    """
    Main function to run the utility.
    """

    api = create_api()
    openai_model, llm_model = create_models()
    setup_routes(api, openai_model, llm_model)


if __name__ == "__main__":
    main()
