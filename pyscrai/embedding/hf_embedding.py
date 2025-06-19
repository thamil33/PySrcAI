# Online embedder using Hugging Face Inference API
import os
import numpy as np
import logging
import requests
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.ERROR)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# List of models to cycle through
MODELS = [
    "BAAI/bge-base-en-v1.5",
    "Xenova/gte-small",
    "FlagEmbedding",
    "Ember",
    "E5"
]

# Function to cycle through models and retry on failure
def embedder(text: str) -> np.ndarray:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    for model in MODELS:
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        try:
            payload = {
                "inputs": text,
                "options": {
                    "wait_for_model": True
                }
            }
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()

            # BGE models return a list of embeddings
            embedding = np.array(response.json()[0])

            # Return the embedding vector
            return embedding
        except requests.exceptions.RequestException as e:
            logging.error(f"Error with model {model}: {str(e)}")
            time.sleep(2)  # Retry after a short delay

    # Return a zero vector as fallback (768 dims for bge-base-en-v1.5)
    logging.error("All models failed. Returning zero vector.")
    return np.zeros(768)
