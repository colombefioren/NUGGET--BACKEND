"""Hybrid retrieval: dense vectors for meaning, BM25 for exact terms, fused with RRF."""

import math
import re
import threading
import unicodedata
from collections import Counter
from itertools import pairwise

from app import store
from app.schemas import Source

RRF_K = 60
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens: list[str] = []
    for word in _WORD.findall(text):
        if _CJK.search(word):
            # CJK scripts have no spaces; character bigrams are a robust stand-in for words.
            chars = [c for c in word if _CJK.match(c)]
            tokens.extend(chars if len(chars) < 2 else (a + b for a, b in pairwise(chars)))
            rest = _CJK.sub(" ", word).split()
            tokens.extend(rest)
        elif len(word) > 1 or word.isdigit():
            tokens.append(word)
    return tokens


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(doc) for doc in corpus]
        self.lens = [len(doc) for doc in corpus]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tfs, self.lens, strict=True):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
            for term in query:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


class _KeywordIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = -1
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._bm25 = BM25([])

    def _refresh(self) -> None:
        with self._lock:
            if self._version == store.version():
                return
            version = store.version()
            ids, texts, metas = store.all_chunks()
            self._ids, self._texts, self._metas = ids, texts, metas
            self._bm25 = BM25([tokenize(t) for t in texts])
            self._version = version

    def search(self, query: str, k: int, doc_ids: set[str] | None) -> list[tuple[str, str, dict]]:
        self._refresh()
        terms = tokenize(query)
        if not terms:
            return []
        scored = [
            (score, i)
            for i, score in enumerate(self._bm25.scores(terms))
            if score > 0 and (doc_ids is None or self._metas[i].get("doc_id") in doc_ids)
        ]
        scored.sort(reverse=True)
        return [(self._ids[i], self._texts[i], self._metas[i]) for _, i in scored[:k]]


keyword_index = _KeywordIndex()


def retrieve(query: str, k: int, doc_ids: list[str] | None = None) -> list[Source]:
    scope = set(doc_ids) if doc_ids is not None else None
    if scope is not None and not scope:
        return []
    pool = k * 3

    where = None
    if scope is not None:
        where = {"doc_id": {"$in": sorted(scope)}}
    dense = store.get_store().similarity_search_with_score(query, k=pool, filter=where)
    sparse = keyword_index.search(query, pool, scope)

    fused: dict[str, float] = {}
    payload: dict[str, tuple[str, dict]] = {}
    similarity: dict[str, float] = {}
    keyword_rank: dict[str, int] = {}

    for rank, (doc, distance) in enumerate(dense):
        cid = doc.id or f"{doc.metadata.get('doc_id')}:{doc.metadata.get('chunk')}"
        fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + rank + 1)
        payload[cid] = (doc.page_content, doc.metadata)
        similarity[cid] = max(0.0, 1.0 - float(distance))
    for rank, (cid, text, meta) in enumerate(sparse):
        fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + rank + 1)
        payload.setdefault(cid, (text, meta))
        keyword_rank[cid] = rank + 1

    best = 2 / (RRF_K + 1)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    sources = []
    for cid, score in ranked:
        text, meta = payload[cid]
        sources.append(
            Source(
                id=cid,
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", "Untitled"),
                page=meta.get("page"),
                chunk=int(meta.get("chunk", 0)),
                content=text,
                score=round(min(1.0, score / best), 4),
                vector_score=round(similarity[cid], 4) if cid in similarity else None,
                keyword_rank=keyword_rank.get(cid),
            )
        )
    return sources
