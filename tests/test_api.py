import json


def upload(client, name: str, content: str | bytes):
    data = content.encode() if isinstance(content, str) else content
    return client.post("/ingest/file", files={"file": (name, data)})


def test_health_reports_configured_model(client):
    body = client.get("/health").json()
    assert body["model"] == "test-model"
    assert body["documents"] == 0


def test_ingest_list_and_delete(client):
    res = upload(client, "cities.md", "# Cities\n\nParis is the capital of France.")
    assert res.status_code == 200
    doc = res.json()["document"]
    assert doc["name"] == "cities.md"
    assert doc["chunks"] >= 1

    assert [d["id"] for d in client.get("/documents").json()] == [doc["id"]]
    assert client.get(f"/documents/{doc['id']}/chunks").json()[0]["content"].startswith("# Cities")

    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_duplicate_upload_is_not_reindexed(client):
    upload(client, "a.txt", "same content")
    res = upload(client, "b.txt", "same content")
    assert res.json()["duplicate"] is True
    assert len(client.get("/documents").json()) == 1


def test_rejects_unsupported_and_empty_files(client):
    assert upload(client, "image.png", b"\x89PNG").status_code == 415
    assert upload(client, "empty.txt", b"").status_code == 422
    assert client.post("/ingest/text", json={"text": "   "}).status_code == 422


def test_query_returns_cited_sources(client):
    upload(client, "france.txt", "Paris is the capital of France.")
    upload(client, "japan.txt", "Tokyo is the capital of Japan.")
    body = client.post("/query", json={"question": "capital of France"}).json()
    assert body["answer"] == "Paris is the capital [1]."
    assert body["sources"][0]["source"] == "france.txt"
    assert 0 < body["sources"][0]["score"] <= 1


def test_query_scoped_to_documents(client):
    upload(client, "france.txt", "Paris is the capital of France.")
    japan = upload(client, "japan.txt", "Tokyo is the capital of Japan.").json()["document"]
    body = client.post(
        "/query", json={"question": "capital of France", "doc_ids": [japan["id"]]}
    ).json()
    assert {s["source"] for s in body["sources"]} == {"japan.txt"}


def test_stream_emits_sources_tokens_and_done(client):
    client.post("/ingest/text", json={"text": "Paris is the capital of France.", "title": "Note"})
    with client.stream("POST", "/query/stream", json={"question": "capital?"}) as res:
        raw = "".join(res.iter_text())
    events = [
        (block.split("\n")[0][7:], json.loads(block.split("\n")[1][6:]))
        for block in raw.strip().split("\n\n")
    ]
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert "".join(d for e, d in events if e == "token") == "Paris is the capital [1]."


def test_broken_embeddings_provider_is_reported_distinctly(client, monkeypatch):
    """Embeddings and chat are separate remote providers; a broken one should say so,
    not be folded into a generic or "language model" error."""
    from app import store

    def boom():
        raise RuntimeError("connection refused")

    # get_store() is @lru_cache'd and calls get_embeddings() eagerly to build the Chroma
    # client, so this patch must be undone before the autouse fixture's own teardown
    # (store.list_documents(), which also goes through get_store()) runs on a broken store.
    working = store.get_embeddings
    monkeypatch.setattr(store, "get_embeddings", boom)
    store.get_store.cache_clear()
    try:
        res = client.post("/ingest/text", json={"text": "will fail to embed"})
        assert res.status_code == 502
        assert "embeddings provider" in res.json()["detail"].lower()

        res = client.post("/query", json={"question": "anything"})
        assert res.status_code == 502
        assert "embeddings provider" in res.json()["detail"].lower()
    finally:
        monkeypatch.setattr(store, "get_embeddings", working)
        store.get_store.cache_clear()
