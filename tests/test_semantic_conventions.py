from __future__ import annotations

from vifinqa.submission.semantics import (
    apply_absolute_difference_convention,
    asks_for_undirected_difference,
    normalize_absolute_difference,
)


def _row(*, question: str, answer: float, query: str = "df1['x'].iloc[0]") -> dict[str, object]:
    return {"id": 7, "question": question, "answer": answer, "pandas_query": query}


def test_absolute_difference_updates_answer_and_query_together() -> None:
    original = _row(question="Chênh lệch giữa A và B là bao nhiêu?", answer=-12.5)

    rows, changed = apply_absolute_difference_convention([original])

    assert changed == [7]
    assert rows[0]["answer"] == 12.5
    assert rows[0]["pandas_query"] == "abs(df1['x'].iloc[0])"
    assert original["answer"] == -12.5


def test_absolute_difference_is_idempotent_and_keeps_directional_change() -> None:
    positive = _row(question="Khoản chênh lệch giữa A và B là bao nhiêu?", answer=12.5)
    directional = _row(question="A giảm bao nhiêu so với B?", answer=-12.5)

    rows, changed = apply_absolute_difference_convention([positive, directional])

    assert changed == []
    assert rows == [positive, directional]


def test_undirected_difference_recognises_published_phrasings() -> None:
    assert asks_for_undirected_difference("Hai giá trị khác biệt bao nhiêu?")
    assert asks_for_undirected_difference("Hai giá trị chênh nhau bao nhiêu?")
    assert asks_for_undirected_difference("Cách biệt giữa hai tỷ lệ là bao nhiêu?")
    assert not asks_for_undirected_difference("Giá trị tăng bao nhiêu phần trăm?")


def test_single_result_normalizer_keeps_query_and_answer_in_execution_parity() -> None:
    query, answer, changed = normalize_absolute_difference(
        question="Chênh lệch giữa A và B là bao nhiêu?",
        pandas_query="df1['value'].iloc[0] - df1['value'].iloc[1]",
        answer=-3.0,
    )

    assert changed is True
    assert query == "abs(df1['value'].iloc[0] - df1['value'].iloc[1])"
    assert answer == 3.0
