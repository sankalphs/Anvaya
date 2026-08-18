from hh_goa_rag.chunking import chunk_corpus, fixed_word_chunks


def test_fixed_word_chunks_support_overlap() -> None:
    assert fixed_word_chunks("a b c d e", size=3, overlap=1) == ["a b c", "c d e", "e"]


def test_chunk_ids_are_stable_and_keep_parent() -> None:
    corpus = [{"passage_id": "p1", "text": "one two three"}]
    strategy = {"strategy": "fixed_words", "size": 2, "overlap": 0}
    first = chunk_corpus(corpus, strategy)
    second = chunk_corpus(corpus, strategy)
    assert first == second
    assert [chunk["parent_id"] for chunk in first] == ["p1", "p1"]

