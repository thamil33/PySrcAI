# Setting up the SentenceTransformer embedder
import os
import numpy as np
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Initialize the SentenceTransformer model once
st_model = SentenceTransformer(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    device='cuda',  # or 'cpu'
    cache_folder=os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')),
    local_files_only=True
)

# Embedder function for the memory bank
def embedder(text: str) -> np.ndarray:
    try:
        # Encode the text and get embeddings
        embedding = st_model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {str(e)}")
        # Return a zero vector as fallback with same dimensions as the model's output
        return np.zeros(st_model.get_sentence_embedding_dimension())
