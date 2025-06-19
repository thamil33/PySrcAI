from absl.testing import absltest
from unittest import mock
import os
import sys
import types
import tempfile
import numpy as np

from pyscrai.rag_agent.agents import ConcordiaAssistant, OpenRouterAssistant
from pyscrai.rag_agent.src.config_loader import load_config

# --- Fake components -------------------------------------------------------


class DummyVectorDBAdapter:
    def __init__(self, config):
        self.docs = []

    def add_documents(self, documents, embeddings, metadatas, ids):
        for doc, emb, meta in zip(documents, embeddings, metadatas):
            self.docs.append(
                {"content": doc, "embedding": np.array(emb), "metadata": meta}
            )

    def ingest_documents(
        self, file_paths, chunker, embedding_adapter, force_rebuild=False
    ):
        for fp in file_paths:
            for text in chunker.chunk_text_file(fp):
                emb = embedding_adapter.embed_text(text)
                self.add_documents([text], [emb], [{"source": fp}], ["id"])

    def query(self, question, embedding_adapter, top_k=5):
        q_emb = embedding_adapter.embed_text(question)
        results = []
        for idx, doc in enumerate(self.docs):
            sim = float(
                np.dot(q_emb, doc["embedding"])
                / (np.linalg.norm(q_emb) * np.linalg.norm(doc["embedding"]) + 1e-9)
            )
            results.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": sim,
                    "rank": idx + 1,
                }
            )
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:top_k]

    def get_collection_info(self):
        return {"name": "test", "count": len(self.docs), "persist_directory": "/tmp"}

    def clear_collection(self):
        self.docs = []


class DummyLLMAdapter:
    def __init__(self, config):
        pass

    def generate(self, prompt, **kwargs):
        return "dummy response"

    def get_model_info(self):
        return {"model_name": "dummy", "provider": "test"}


class DummyChunker:
    def __init__(self, config):
        pass

    def chunk_text_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------


def fake_post(url, headers=None, json=None):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [0.1, 0.1, 0.1]

    return Resp()


class AssistantsTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        os.environ["HF_API_TOKEN"] = "dummy"
        os.environ["OPENROUTER_API_KEY"] = "dummy"

        self.post_patch = mock.patch(
            "pyscrai.embedding.hf_embedding.requests.post", fake_post
        )
        self.post_patch.start()
        self.addCleanup(self.post_patch.stop)

        self.vector_patch = mock.patch(
            "pyscrai.rag_agent.adapters.vector_db_adapter.VectorDBAdapter",
            DummyVectorDBAdapter,
        )
        self.llm_patch = mock.patch(
            "pyscrai.rag_agent.adapters.llm_adapter.LLMAdapter",
            DummyLLMAdapter,
        )

        # Provide a dummy chunking module to satisfy relative import in
        # BaseRAGAgent._setup_chunker
        dummy_chunk_mod = types.ModuleType("pyscrai.rag_agent.src.chunking")
        dummy_chunk_mod.DocumentChunker = DummyChunker
        sys.modules["pyscrai.rag_agent.src.chunking"] = dummy_chunk_mod
        self.addCleanup(sys.modules.pop, "pyscrai.rag_agent.src.chunking")

        for p in (self.vector_patch, self.llm_patch):
            p.start()
            self.addCleanup(p.stop)

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.doc1 = os.path.join(self.tmpdir.name, "a.txt")
        with open(self.doc1, "w") as f:
            f.write("first document")
        self.doc2 = os.path.join(self.tmpdir.name, "b.txt")
        with open(self.doc2, "w") as f:
            f.write("second document")

        self.config = load_config()

    def _run_agent(self, agent_cls):
        agent = agent_cls(self.config)
        agent.ingest_documents([self.doc1, self.doc2])
        resp = agent.query("test question")
        info = agent.get_collection_info()
        self.assertEqual(info["count"], 2)
        self.assertIn("dummy response", resp)

    def test_concordia_assistant(self):
        self._run_agent(ConcordiaAssistant)

    def test_openrouter_assistant(self):
        self._run_agent(OpenRouterAssistant)


if __name__ == "__main__":
    absltest.main()
