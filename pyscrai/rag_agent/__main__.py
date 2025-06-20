"""
RAG Agent Module Entry Point

Allows running the CLI with:
python -m pyscrai.rag_agent
"""
from dotenv import load_dotenv

load_dotenv()

from .src.cli import main


if __name__ == "__main__":
    main()
