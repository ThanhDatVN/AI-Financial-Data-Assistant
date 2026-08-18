from __future__ import annotations

import random
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

sampler = import_module("scripts.70_sample_programs")
renderer = import_module("scripts.71_render_questions")
backfill = import_module("scripts.74_backfill_condition")


def test_a_draw_carries_no_filter_unless_its_family_has_one() -> None:
    draw = sampler._Draw(None, "VND", "đồng", 1.0, "Doanh thu thuần", "2024")
    assert draw.condition is None
    assert draw.row_label == "Doanh thu thuần"


def test_the_hard_family_is_phrased_as_a_rank_not_a_raw_threshold() -> None:
    """A threshold in dong is not a question anyone asks, and the two describe the same years.

    Survivors are exactly the years whose gate value beats a threshold drawn from the gate series
    itself, so every non-survivor sits at or below it. "The N years where this item was highest"
    is therefore the same set, said the way the paper says it.
    """
    clause = renderer._condition_clause(
        {"row_label": "Doanh thu thuần", "top_n": 3, "of_years": 5, "threshold": 9.13e10}
    )
    assert "Doanh thu thuần" in clause
    assert "3 năm cao nhất" in clause
    assert "91" not in clause


def test_a_hard_sample_states_its_filter_in_the_prompt() -> None:
    sample = {
        "id": 1,
        "family": "conditional",
        "ticker": "VJC",
        "report_years": [2015, 2016, 2017, 2018],
        "scope": "separate",
        "row_label": "Các khoản tương đương tiền",
        "column_label": "31/12/2018 VND",
        "section_title": "",
        "target_unit_text": "đồng",
        "condition": {"row_label": "Tiền mặt", "top_n": 2, "of_years": 4},
    }
    prompt = renderer._prompt(sample, {"VJC": "CTCP Hàng không Vietjet"}, random.Random(0))
    assert "Tiền mặt" in prompt
    assert "2 năm cao nhất" in prompt
    # And the family brief has to ask for both halves, or the model writes only the answer half.
    assert renderer._FAMILY_BRIEF["conditional"] in prompt


def test_every_family_the_sampler_draws_has_a_brief() -> None:
    """A family with no brief renders as an empty instruction line and a plainer question.

    The Hard family was 21.2% of the dev set and had no brief: those questions would have been
    written as plain extremums, whose answers the sampler had already proved differ.
    """
    families = {name for name, _weight in sampler.FAMILY_WEIGHTS}
    assert families <= set(renderer._FAMILY_BRIEF), families - set(renderer._FAMILY_BRIEF)


def test_a_hard_sample_missing_its_filter_is_refused_rather_than_guessed() -> None:
    assert renderer._condition_clause(None) == ""
    assert renderer._condition_clause({"row_label": "Tiền mặt"}) == ""
    assert renderer._condition_clause({"top_n": 2}) == ""


def _gate_frame(label: str, values: list[float]) -> dict[str, pd.DataFrame]:
    """One frame per year, each holding the gate row at (2, 1) and an answer row at (0, 1)."""
    frames = {}
    for index, value in enumerate(values, start=1):
        frames[f"df{index}"] = pd.DataFrame(
            [
                {"row_index": 0, "column_index": 1, "row_label": "Doanh thu", "base_value": 1.0},
                {"row_index": 2, "column_index": 1, "row_label": label, "base_value": value},
            ]
        )
    return frames


def _row(labels: list[str], threshold: float) -> dict[str, object]:
    return {
        "id": 1,
        "family": "conditional",
        "table_refs": [f"D{index}|table_1" for index in range(1, len(labels) + 1)],
        "program": {
            "kind": "select",
            "operator": "argmax",
            "members": [],
            "keys": [],
            "conditions": [
                {
                    "comparator": ">",
                    "left": [
                        {
                            "kind": "cell",
                            "variable": f"df{index}",
                            "row_index": 2,
                            "column_index": 1,
                        }
                        for index in range(1, len(labels) + 1)
                    ],
                    "right": {"kind": "literal", "value": threshold},
                }
            ],
        },
    }


def test_the_filter_is_read_back_out_of_the_program_the_sample_already_carries() -> None:
    """Regenerating would swap a set that mirrors the paper for one that does not.

    The dev and train sets were drawn on an earlier revision: `--split dev --count 499` now
    returns 416 samples with 23 Hard ones against the 106 on disk. Everything needed is already
    in the file, so the field is recovered rather than the set replaced.
    """
    label = "Tiền và tương đương tiền"
    frames = _gate_frame(label, [10.0, 30.0, 20.0, 40.0])
    condition = backfill.recover(_row([label] * 4, threshold=15.0), frames)
    assert condition["row_label"] == label
    # Three of the four years clear 15.0, and they are exactly the three highest.
    assert condition["top_n"] == 3
    assert condition["of_years"] == 4
    assert condition["threshold"] == 15.0


def test_a_bullet_is_table_furniture_not_a_different_line_item() -> None:
    """The sampler matched these rows folded, so refusing them on surface form is the wrong gate.

    Three of the 106 dev samples differ only by a bullet or a footnote marker, and the question
    should name the tidy form of the two.
    """
    frames = _gate_frame("x", [10.0, 30.0, 20.0])
    frames["df1"].loc[1, "row_label"] = "▪ Phải trả nhà cung cấp"
    frames["df2"].loc[1, "row_label"] = "Phải trả nhà cung cấp"
    frames["df3"].loc[1, "row_label"] = "Phải trả nhà cung cấp (*)"
    condition = backfill.recover(_row(["x"] * 3, threshold=15.0), frames)
    assert condition["row_label"] == "Phải trả nhà cung cấp"


def test_coordinates_that_stopped_pointing_at_the_same_row_are_refused() -> None:
    frames = _gate_frame("x", [10.0, 30.0, 20.0])
    frames["df2"].loc[1, "row_label"] = "Một khoản mục hoàn toàn khác"
    with pytest.raises(backfill.Unrecoverable):
        backfill.recover(_row(["x"] * 3, threshold=15.0), frames)


def test_a_vacuous_filter_is_refused_because_the_sampler_never_emitted_one() -> None:
    """Fewer than two survivors makes the ranking meaningless, and `_sample_conditional` said so."""
    frames = _gate_frame("x", [10.0, 30.0, 20.0])
    with pytest.raises(backfill.Unrecoverable):
        backfill.recover(_row(["x"] * 3, threshold=25.0), frames)


def _schema(variable: str) -> object:
    """A stand-in: `_fit` only ever reads `.variable` and hands the list back to `render`."""

    class _Stub:
        def __init__(self, name: str) -> None:
            self.variable = name

    return _Stub(variable)


def test_padding_is_cut_back_until_the_answer_fits_and_gold_is_never_cut() -> None:
    """At 19 distractors the longest dev prompt reaches 77,000 tokens in a 16,384 context.

    Counting those as wrong answers would push X down by several points, and X decides whether to
    rent a GPU. Gold cannot be among what is dropped: a sample missing its gold table is not a
    harder question, it is a different one.
    """
    filterer = import_module("scripts.73_filter_synthetic")
    schemas = [_schema(f"df{index}") for index in range(1, 6)]
    gold = {"df2"}

    # 4,000 tokens per candidate: only one can fit beside a 6,144-token answer in 16,384.
    def measure(_messages: list[dict[str, str]], counter: list[int] = []) -> int:  # noqa: B006
        return 4000 * measure.width

    def render(candidates: list[object]) -> tuple[str, str]:
        measure.width = len(candidates)
        return "system", f"{len(candidates)} candidates"

    measure.width = len(schemas)
    _, _, kept = filterer._fit(
        schemas, gold, render=render, measure=measure, context_limit=16384, max_tokens=6144
    )
    assert [schema.variable for schema in kept] == ["df1", "df2"]

    # Gold alone still over budget: return it rather than drop the table the answer needs.
    _, _, only_gold = filterer._fit(
        [_schema("df1")],
        {"df1"},
        render=render,
        measure=measure,
        context_limit=4096,
        max_tokens=6144,
    )
    assert [schema.variable for schema in only_gold] == ["df1"]


def test_an_unmeasurable_prompt_is_left_alone_rather_than_guessed_at() -> None:
    """A server with no tokenizer route returns 0, and guessing is what the route replaced."""
    filterer = import_module("scripts.73_filter_synthetic")
    schemas = [_schema(f"df{index}") for index in range(1, 4)]
    _, _, kept = filterer._fit(
        schemas,
        {"df1"},
        render=lambda candidates: ("system", "user"),
        measure=lambda messages: 0,
        context_limit=16384,
        max_tokens=6144,
    )
    assert len(kept) == 3


def test_the_paraphrase_rule_selects_against_the_paper_so_it_is_off_by_default() -> None:
    """It rejected the way the exam actually asks, and that cost half a render session.

    Applied to the 1,012 real questions against the row labels of their own retrieved tables, the
    same test flags 590 -- 58.3% -- starting with question 1, the project's regression anchor:
    "Lãi tiền gửi năm 2018 của công ty mẹ ..." over a row labelled "Lãi tiền gửi". So it does not
    measure a bad question, it measures a question phrased the way the paper phrases it.

    Enforcing it built a dev set that mirrors the paper less, not more: it kept only the harder
    half and would have made X read worse than production.
    """
    # The rule still works; what changed is that nothing calls it unless asked. Question 1
    # is itself a flagged case, which is the whole point.
    assert renderer._quotes_label(
        "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu?",
        "Lãi tiền gửi",
    )
    assert renderer._quotes_label(
        "Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023 là bao nhiêu tỷ đồng?",
        "Lợi nhuận sau thuế",
    )
    # A label of one or two words is too generic to be evidence either way.
    assert not renderer._quotes_label("Tiền mặt cuối kỳ là bao nhiêu?", "Tiền mặt")

    body = Path(renderer.__file__).read_text(encoding="utf-8")
    assert "args.require_paraphrase and _quotes_label(" in body
    assert '"--require-paraphrase"' in body
    # And the prompt must stop forbidding it, or the model keeps paying for a rule nobody applies.
    assert "KHÔNG chép nguyên văn" not in body


def test_a_refused_generation_is_kept_not_just_counted() -> None:
    """A count says a gate fired; only the text says why.

    The first render came back at 50%, and with counts alone the diagnosis had to be rebuilt on a
    laptop while two GPU sessions sat idle.
    """
    body = Path(renderer.__file__).read_text(encoding="utf-8")
    assert '"--rejected"' in body
    assert "def refuse(" in body
    assert 'refuse(sample, attempt, "no_question_mark", candidate)' in body
    assert '"candidate": candidate,' in body


def test_a_question_buried_under_trailing_prose_is_salvaged_not_discarded() -> None:
    """7% of the 544 refused generations held a good question with commentary bolted on."""
    assert (
        renderer._trim_to_question(
            "Doanh thu thuần của CTCP ABC năm 2022 là bao nhiêu tỷ đồng? Hãy giải thích thêm."
        )
        == "Doanh thu thuần của CTCP ABC năm 2022 là bao nhiêu tỷ đồng?"
    )
    # The mark has to be far enough in to be the end of a question rather than part of one.
    assert renderer._trim_to_question("Năm 2020? Hãy") == "Năm 2020? Hãy"
    assert renderer._trim_to_question("Không có dấu hỏi nào ở đây cả.").endswith(".")


def test_a_postal_address_is_not_a_report_section() -> None:
    """16% of section titles are addresses, digit runs or LaTeX, and they went into the question.

    One rendered question asked about cash equivalents "được trình bày tại mục 10 $^{th}$ Floor,
    Sun Wah Tower, 115 Nguyen Hue Street, Ben Nghe Ward, District 1, Ho Chi Minh City, Vietnam".
    The section was always optional context, so dropping a bad one costs nothing.
    """
    assert (
        renderer._usable_section("02. CÁC KHOẢN ĐẦU TƯ TÀI CHÍNH")
        == "02. CÁC KHOẢN ĐẦU TƯ TÀI CHÍNH"
    )
    assert renderer._usable_section("10 $^{th}$ Floor, Sun Wah Tower, 115 Nguyen Hue Street") == ""
    assert renderer._usable_section("1121 2231 1411") == ""
    assert renderer._usable_section("") == ""


def test_every_family_shows_the_model_a_question_shaped_example() -> None:
    """89% of refusals ended in a full stop -- a fluent instruction, not what the paper writes.

    Measured on the 1,012 real questions: 965 end in a question mark, 44 in a full stop. So the
    rule is right and the model was wrong, which is the opposite of the paraphrase rule. Asking
    for the shape had not worked twice; every family now gets shown one.
    """
    families = {name for name, _weight in sampler.FAMILY_WEIGHTS}
    assert families <= set(renderer._EXAMPLE), families - set(renderer._EXAMPLE)
    for family, example in renderer._EXAMPLE.items():
        assert example.endswith("?"), family
