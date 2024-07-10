"""
Utility to run the API server for the Langchain project using FastAPI, that serves the OpenAI GPT-3 and llama2 models.
"""

import logging
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
)


def create_api() -> FastAPI:
    """
    Create the FastAPI instance.

    Returns
    -------
    FastAPI
        The FastAPI instance
    """

    logging.info("Creating FastAPI instance")

    api = FastAPI(title="Langchain API Server", version="1.0", description="Test server ")
    return api


def create_models():
    """
    Create the OpenAI and llama2 models instances.

    Returns
    -------
    openai_model, llm_model
        The OpenAI and llama2 models instances
    """

    logging.info("Creating OpenAI and llama2 models instances")

    openai_model = ChatOpenAI()
    llm_model = Ollama(model="llama2")
    return openai_model, llm_model


def setup_routes(api: FastAPI, openai_model: ChatOpenAI, llm_model: Ollama):
    """
    Setup the routes for the API server, using the OpenAI and llama2 models , prompts and api

    Parameters
    ----------
    api : FastAPI
        The FastAPI instance
    openai_model : ChatOpenAI
        The OpenAI model instance
    llm_model : Ollama
        The llama2 model instance
    """

    logging.info("Setting up routes for the API server")

    prompt1 = ChatPromptTemplate.from_template(
        "Explain the concept {topic} with 5 bullet points only "
    )
    prompt2 = ChatPromptTemplate.from_template("Write me an essay about {topic} using 500 words.")

    add_routes(api, prompt1 | openai_model, path="/essay")
    add_routes(api, prompt2 | llm_model, path="/poem")


def main():
    """
    Main function to run the utility.

    Calls the create_api : Creates the FastAPI instance.
    create_models: Creates the OpenAI and llama2 models instances.
    setup_routes : Setup the routes for the API server, using the OpenAI and llama2 models , prompts and api
    """

    logging.info("Running the utility to run the API server for the Langchain project")

    api = create_api()

    openai_model, llm_model = create_models()

    setup_routes(api, openai_model, llm_model)


if __name__ == "__main__":
    main()
