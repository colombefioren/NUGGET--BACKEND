from app.retrieval import BM25, tokenize


def test_tokenize_folds_accents_and_case():
    assert tokenize("Élève ÉCOLE") == ["eleve", "ecole"]


def test_tokenize_splits_cjk_into_bigrams():
    assert tokenize("東京都") == ["東京", "京都"]


def test_bm25_prefers_documents_with_rare_terms():
    corpus = [tokenize("the cat sat"), tokenize("the dog ran"), tokenize("the the the")]
    scores = BM25(corpus).scores(tokenize("dog"))
    assert scores.index(max(scores)) == 1
