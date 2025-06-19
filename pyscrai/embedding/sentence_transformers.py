# Online embedder using Hugging Face Inference API
import os
import numpy as np
import logging
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.ERROR)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# Using the BAAI/bge-base-en-v1.5 model which is well-supported by the Inference API
API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-base-en-v1.5"

def embedder(text: str) -> np.ndarray:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        # The bge models expect a different payload format
        payload = {
            "inputs": text,
            "options": {
                "wait_for_model": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # BGE models return a list of embeddings
        embedding = np.array(response.json()[0])
        
        # Return the embedding vector
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {str(e)}")
        # Return a zero vector as fallback (768 dims for bge-base-en-v1.5)
        return np.zeros(768)
