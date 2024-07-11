"""
Utility to run the API server for the Langchain project using FastAPI, that serves the OpenAI GPT-3 and llama2 models.
"""

import logging
import yaml
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
)


def load_env():
    """
    Load the environment variables.
    """

    logging.info("Loading the environment variables")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")


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


def create_models(model_name: str = "llama3"):
    """
    Create the OpenAI and llama2 models instances.

    Returns
    -------
    openai_model, llm_model
        The OpenAI and llama2 models instances
    """

    logging.info("Creating OpenAI and llama2 models instances")

    if model_name == "openai":
        model = ChatOpenAI()
    elif model_name == "llama3":
        model = Ollama(model="llama3")
    else:
        raise ValueError(f"Invalid model")

    return model


def setup_routes(api: FastAPI, model):
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

    prompt = ChatPromptTemplate.from_template(
        "Explain the concept {topic} with 5 bullet points only "
    )

    add_routes(api, prompt | model, path="/essay")


def main():
    """
    Main function to run the utility.

    Calls the create_api : Creates the FastAPI instance.
    create_models: Creates the OpenAI and llama2 models instances.
    setup_routes : Setup the routes for the API server, using the OpenAI and llama2 models , prompts and api
    """

    logging.info("Running the utility to run the API server for the Langchain project")

    with open("./langchain_api/config.yaml") as file:
        config_data = yaml.safe_load(file)

    load_env()

    api = create_api()

    model = create_models(config_data["model_name"])

    setup_routes(api, model)

    uvicorn.run(api, host="localhost", port=8000)


if __name__ == "__main__":
    main()
