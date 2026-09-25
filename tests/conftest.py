import hashlib
import math
import os
import re
import tempfile

os.environ.update(
    LUMEN_API_BASE="http://llm.test/v1",
    LUMEN_CHAT_MODEL="test-model",
    LUMEN_API_KEY="test-key",
    LUMEN_CHROMA_DIR=tempfile.mkdtemp(prefix="lumen-test-"),
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from app import embeddings, llm, store  # noqa: E402


class HashEmbeddings(Embeddings):
    """Bag-of-words hashing: deterministic, offline, and similar texts get similar vectors."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    fake_llm = FakeListChatModel(responses=["Paris is the capital [1]."] * 10)
    monkeypatch.setattr(embeddings, "get_embeddings", lambda: HashEmbeddings())
    monkeypatch.setattr(store, "get_embeddings", lambda: HashEmbeddings())
    monkeypatch.setattr(llm, "chat_model", lambda: fake_llm)
    store.get_store.cache_clear()
    yield
    for doc in store.list_documents():
        store.delete_document(doc.id)
    store.get_store.cache_clear()


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)
