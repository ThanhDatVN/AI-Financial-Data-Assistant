"""Write the Vietnamese question for a program whose answer is already known.

`scripts/70_sample_programs.py` produces the half that has to be exact: a program over real
tables, executed, so the answer is right by construction. This script produces the half that has
to sound like the competition: the question a person would have asked to get that number.

The model never sees a value and never computes anything. It is given the company, the period,
the line item and the unit asked for, and it phrases them. Nothing it returns can change an
answer, so the worst a bad generation costs is one rejected sample.

Only open weights, per the competition rules -- the same served model the run itself uses.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED = 20260811

# Measured over the 1.012 real questions: 366 name the separate report, 13 name the consolidated
# one, and 633 name neither. Submissions 2807 and 2808 then showed that reading an unstated
# question as consolidated is right about 93% of the time, which puts roughly 600 consolidated
# and 410 separate behind those counts. So the corpus does not hide scope at random: a separate
# report is nearly always named, and a consolidated one nearly never is. Mirroring that is what
# makes the synthetic set exercise the same ambiguity as the real one.
P_STATE_SCOPE = {"separate": 0.89, "consolidated": 0.02}

_SCOPE_PHRASES = {
    "separate": ("công ty mẹ", "riêng lẻ", "báo cáo riêng của công ty mẹ"),
    "consolidated": ("hợp nhất", "báo cáo hợp nhất"),
}

# The ceiling is a safety valve, not a style rule. At 400 it became a quality filter by accident:
# a Hard question names the filter line item, the answer line item, the company and four or five
# years, which measures about 292 characters against 132-166 for every other family, and 24 of the
# 544 refused generations were cut off at exactly 400 with the sentence unfinished.
QUESTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"question": {"type": "string", "minLength": 20, "maxLength": 700}},
    "required": ["question"],
    "additionalProperties": False,
}

SYSTEM = (
    "Bạn viết câu hỏi thi bằng tiếng Việt về báo cáo tài chính doanh nghiệp. "
    "Bạn nhận mô tả một phép tính đã được thực hiện trên một bảng có thật, và viết lại thành "
    "đúng một câu hỏi mà người ra đề sẽ hỏi để nhận được kết quả đó. "
    "Không nêu đáp án. Không giải thích. Chỉ trả về câu hỏi."
)

_FAMILY_BRIEF = {
    "lookup": "Hỏi thẳng giá trị của khoản mục đó.",
    "ratio": "Hỏi tỷ trọng của khoản mục đó trên khoản mục lớn nhất cùng cột, tính bằng phần trăm.",
    "change": "Hỏi khoản mục đó thay đổi bao nhiêu phần trăm giữa hai năm.",
    "extremum": "Hỏi năm nào khoản mục đó đạt giá trị lớn nhất (hoặc nhỏ nhất) trong các năm nêu.",
    # The Hard tier, and the one family whose question is not a function of `row_label` alone: a
    # different line item decides which years are eligible, and the sampler discards any draw
    # whose filtered answer matches the unfiltered one. Phrased without the filter, the question
    # has a provably different answer, so the whole 21.3% tier would be rendered wrong and then
    # dropped by the solver filter for being unanswerable.
    "conditional": (
        "Hỏi giá trị của khoản mục đó, nhưng CHỈ xét trong nhóm năm thoả điều kiện nêu bên dưới. "
        "Câu hỏi bắt buộc phải nêu cả điều kiện lọc lẫn khoản mục cần trả lời."
    ),
}


def _condition_clause(condition: dict[str, object] | None) -> str:
    """Say the filter the way a question would: a rank among the years, not a raw threshold."""
    if not condition:
        return ""
    label = str(condition.get("row_label") or "").strip()
    top_n = condition.get("top_n")
    if not label or not isinstance(top_n, int):
        return ""
    return (
        f"Điều kiện lọc: chỉ xét những năm mà '{label}' nằm trong nhóm {top_n} năm cao nhất "
        "(so với chính các năm nêu trong câu hỏi). "
        "Diễn đạt điều kiện này bằng lời, đừng nêu con số ngưỡng."
    )


def _normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c for c in folded if c.isalnum() or c.isspace()).split())


def _company_names(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    return {
        str(row["Mã CK"]).strip().upper(): str(row["Tên công ty"]).strip()
        for _, row in frame.iterrows()
    }


def _name_form(ticker: str, full_name: str | None, rng: random.Random) -> str:
    """Real questions name a company three ways; keep all three in play."""
    if not full_name:
        return ticker
    draw = rng.random()
    if draw < 0.40:
        return f"{full_name} ({ticker})"
    if draw < 0.70:
        return full_name
    return ticker


def _scope_clause(scope: str, rng: random.Random) -> str:
    if rng.random() >= P_STATE_SCOPE.get(scope, 0.5):
        return ""
    phrases = _SCOPE_PHRASES.get(scope, ())
    return phrases[rng.randrange(len(phrases))] if phrases else ""


_EXAMPLE = {
    "lookup": "Doanh thu thuần của CTCP ABC năm 2022 là bao nhiêu tỷ đồng?",
    "ratio": "Hàng tồn kho của CTCP ABC chiếm bao nhiêu phần trăm tổng tài sản năm 2022?",
    "change": "Doanh thu thuần của CTCP ABC thay đổi bao nhiêu phần trăm từ năm 2021 sang 2022?",
    "extremum": "Trong các năm 2020, 2021 và 2022, năm nào CTCP ABC có doanh thu thuần cao nhất?",
    # The Hard form is the one the model keeps writing as an instruction, so its example carries
    # both clauses and still lands on a question mark.
    "conditional": (
        "Trong các năm 2020, 2021 và 2022, chỉ xét những năm mà tiền và tương đương tiền của "
        "CTCP ABC nằm trong nhóm 2 năm cao nhất, năm nào công ty có hàng tồn kho lớn nhất và "
        "giá trị là bao nhiêu tỷ đồng?"
    ),
}


_ADDRESS_RE = re.compile(
    r"(Street|Floor|Tower|District|Ward|City|Vietnam|Chi Minh|Hanoi|Ha Noi)", re.IGNORECASE
)
_DIGITS_ONLY_RE = re.compile(r"^[0-9 .,/-]+$")


def _usable_section(section: str) -> str:
    """Keep a report section only when it names one, which 84% of them do.

    The other 16% are the company's postal address, a bare run of digits off a page header, or a
    LaTeX fragment the parser kept. Passing those through put "được trình bày tại mục 10 $^{th}$
    Floor, Sun Wah Tower, 115 Nguyen Hue Street, Ben Nghe Ward, District 1, Ho Chi Minh City,
    Vietnam" inside a question about cash equivalents -- longer, stranger, and no help to anyone.
    An omitted section costs nothing: it was always optional context.
    """
    stripped = section.strip()
    if not stripped or len(stripped) > 90:
        return ""
    if _ADDRESS_RE.search(stripped) or _DIGITS_ONLY_RE.match(stripped):
        return ""
    if chr(92) in stripped or "$" in stripped or "^{" in stripped:
        return ""
    return stripped


_CORRECTION = {
    "no_question_mark": (
        "Câu vừa rồi bị loại vì KHÔNG kết thúc bằng dấu hỏi -- nó là câu trần thuật hoặc câu "
        "mệnh lệnh kết thúc bằng dấu chấm. Giữ nguyên nội dung, viết lại thành MỘT câu hỏi kết "
        "thúc bằng `?`."
    ),
    "copied_row_label": (
        "Câu vừa rồi bị loại vì chép nguyên văn nhãn dòng. Diễn đạt lại khoản mục bằng thuật ngữ "
        "khác, giữ nguyên phần còn lại."
    ),
}


def _retry_prompt(base: str, reason: str, candidate: str) -> str:
    """Say what was wrong with the last try, instead of drawing again at a warmer temperature.

    Attempt two used to differ only by temperature, and the model duly repeated itself: 85 of the
    217 refusals in one split were second attempts. The generator has fed its validator's message
    back into the retry since the start; this loop had no reason not to.
    """
    note = _CORRECTION.get(reason)
    if not note:
        return base
    return f'{base}\n\nLần trước bạn trả lời: "{candidate[:300]}"\n{note}'


def _prompt(sample: dict[str, object], names: dict[str, str], rng: random.Random) -> str:
    ticker = str(sample["ticker"])
    raw_years = sample["report_years"]
    years = [int(year) for year in raw_years] if isinstance(raw_years, list) else []
    company = _name_form(ticker, names.get(ticker), rng)
    scope = _scope_clause(str(sample["scope"]), rng)
    period = (
        f"năm {years[0]}"
        if len(years) == 1
        else f"các năm {', '.join(str(year) for year in years)}"
    )
    section = _usable_section(str(sample.get("section_title") or ""))
    lines = [
        f"Doanh nghiệp: {company}",
        f"Kỳ báo cáo: {period}",
        f"Khoản mục trong bảng: {sample['row_label']}",
        f"Nhãn cột: {sample['column_label']}",
        # The phrase, not the enum: a question that asks for "BILLION_VND" is not Vietnamese.
        f"Đơn vị câu trả lời phải dùng: {sample['target_unit_text']}",
        f"Loại câu hỏi: {_FAMILY_BRIEF.get(str(sample['family']), '')}",
    ]
    condition_raw = sample.get("condition")
    condition = _condition_clause(condition_raw if isinstance(condition_raw, dict) else None)
    if condition:
        lines.append(condition)
    if section:
        lines.insert(3, f"Mục của báo cáo: {section}")
    if scope:
        lines.insert(1, f"Phạm vi báo cáo phải nêu rõ trong câu hỏi: {scope}")
    else:
        lines.insert(1, "Phạm vi báo cáo: KHÔNG được nhắc tới trong câu hỏi.")
    lines.append(
        "\nYêu cầu bắt buộc:\n"
        "- Kết thúc bằng dấu hỏi `?`. Đây là điều kiện cứng: một câu trần thuật kết thúc bằng "
        "dấu chấm sẽ bị loại, dù nội dung đúng.\n"
        f"- Viết đúng MỘT câu hỏi. Ví dụ đúng dạng: \"{_EXAMPLE.get(str(sample['family']), '')}\"\n"
        "- Nêu rõ doanh nghiệp, kỳ báo cáo và đơn vị của đáp án.\n"
        "- Gọi tên khoản mục theo cách người đọc báo cáo tài chính thường gọi; dùng đúng "
        "thuật ngữ của bảng cũng được, đề thi thật làm vậy ở 58% số câu.\n"
        "- KHÔNG nêu con số đáp án.\n"
        "- Dùng lối viết tự nhiên của đề thi tiếng Việt."
    )
    return "\n".join(lines)


def _trim_to_question(candidate: str) -> str:
    """Drop anything the model added after its question mark.

    7% of refused generations held a perfectly good question with commentary bolted on. Throwing
    those away spends a generation to punish a suffix. The mark has to be far enough in to be the
    end of a question rather than part of one.
    """
    cut = candidate.rfind("?")
    return candidate[: cut + 1].strip() if cut >= 30 else candidate


def _is_proper_noun(row_label: str) -> bool:
    """Does this label name a specific entity rather than a financial concept?

    Vietnamese statement lines run lower case after the first word -- "Phải trả người bán ngắn
    hạn" -- while a named project or subsidiary keeps its capitals: "KĐT Mỹ Đình Nam Từ Liêm".
    """
    words = row_label.split()[1:]
    return sum(1 for word in words if word[:1].isupper()) >= 2


def _quotes_label(question: str, row_label: str) -> bool:
    """Does this question use the row label's own words?

    It was a rejection rule, on the reasoning that mapping everyday wording onto a statement's own
    line label is the hardest part of the real task, so a question that copies the label teaches
    the easy half and hides the hard one.

    The reasoning was never checked against the paper, and the paper disagrees. Applying this test
    to the 1,012 real questions against the row labels of their own retrieved tables flags
    **590 of them -- 58.3%**, starting with question 1, which is the project's regression anchor:
    "Lãi tiền gửi năm 2018 của công ty mẹ ..." over a row labelled "Lãi tiền gửi".

    So it does not measure "a bad question". It measures "a question phrased the way the exam
    phrases it", and rejecting those built a dev set that mirrors the paper less, not more -- it
    would have kept the hard half only and made X read worse than production. Now off by default;
    `--require-paraphrase` restores the old behaviour for anyone who wants the harder subset.
    """
    folded_label = _normalise(row_label)
    if len(folded_label.split()) < 3 or _is_proper_noun(row_label):
        return False
    return folded_label in _normalise(question)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("programs", type=Path, help="JSONL from 70_sample_programs.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--companies", type=Path, default=ROOT / "data/raw/ViFinQA/code_stock.csv")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="VIFINQA_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        # Raised alongside the schema ceiling so neither bound is the one that decides what a
        # Hard question may say. Vietnamese with diacritics costs roughly one token per 1.2-1.8
        # characters, so a 700-character allowance needs well over 256 tokens behind it.
        default=640,
    )
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--require-paraphrase",
        action="store_true",
        help=(
            "Refuse a question that uses the row label's own words. Off by default: the same "
            "test flags 58.3 percent of the real questions, so it selects against the paper "
            "rather than against bad questions"
        ),
    )
    parser.add_argument(
        "--rejected",
        type=Path,
        help=(
            "Where to write the refused generations, with the reason. Counting them says a gate "
            "fired; only the text says why. Without it the first render came back at 50 percent "
            "and the diagnosis had to be reconstructed from a laptop while two GPU sessions "
            "sat idle"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    names = _company_names(args.companies)
    raw = args.programs.read_text(encoding="utf-8").splitlines()
    samples = [json.loads(line) for line in raw if line.strip()]
    if args.limit:
        samples = samples[: args.limit]

    # Rendering a few thousand questions outlasts more than one Kaggle session, so resume from
    # whatever the last one wrote rather than paying for it twice.
    done: dict[int, dict[str, object]] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["id"])] = row
    print(f"rendering {len(samples)} programs, {len(done)} already done")

    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "local-vllm"),
        timeout=args.request_timeout,
        max_retries=0,
    )

    rejected: dict[str, int] = defaultdict(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejected_handle = (
        args.rejected.open("a", encoding="utf-8") if args.rejected else None  # noqa: SIM115
    )

    def refuse(sample: dict[str, object], attempt: int, reason: str, candidate: str) -> None:
        """Count it, and keep the text. The count says a gate fired; the text says why."""
        rejected[reason] += 1
        if rejected_handle is None:
            return
        rejected_handle.write(
            json.dumps(
                {
                    "id": sample.get("id"),
                    "family": sample.get("family"),
                    "attempt": attempt,
                    "reason": reason,
                    "row_label": sample.get("row_label"),
                    "candidate": candidate,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        rejected_handle.flush()

    with args.output.open("a", encoding="utf-8") as handle:
        for position, sample in enumerate(samples, start=1):
            sample_id = int(sample["id"])
            if sample_id in done:
                continue
            # A Hard sample without its filter recorded cannot be phrased: the question would ask
            # the plainer thing, whose answer the sampler guarantees is different. Refusing here
            # is loud; rendering it would look like a fluent question and fail at the solver.
            if str(sample.get("family")) == "conditional" and not _condition_clause(
                sample.get("condition") if isinstance(sample.get("condition"), dict) else None
            ):
                rejected["conditional_without_condition"] += 1
                continue
            question = None
            # The prompt is drawn once: `_name_form` and `_scope_clause` consume the rng, so
            # rebuilding it per attempt would silently rename the company between tries.
            base_prompt = _prompt(sample, names, rng)
            last_reason = ""
            last_candidate = ""
            for attempt in range(1, args.attempts + 1):
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": _retry_prompt(base_prompt, last_reason, last_candidate),
                            },
                        ],
                        temperature=0.3 if attempt == 1 else 0.8,
                        seed=args.seed + attempt,
                        max_tokens=args.max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "question",
                                "strict": True,
                                "schema": QUESTION_SCHEMA,
                            },
                        },
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    content = response.choices[0].message.content
                    if not content:
                        # Keep the empty one too. Every other refusal path here writes its text,
                        # and these two are exactly the paths a generation truncated by the token
                        # budget takes -- so a diagnosis would have been back to reading counts,
                        # which is the mistake that cost a session on 18/08.
                        refuse(sample, attempt, "empty", "")
                        continue
                    candidate = str(json.loads(content)["question"]).strip()
                except Exception as error:  # noqa: BLE001 - one lost sample, not a lost run
                    refuse(sample, attempt, type(error).__name__, str(error)[:400])
                    continue
                candidate = _trim_to_question(candidate)
                if not candidate.endswith("?"):
                    refuse(sample, attempt, "no_question_mark", candidate)
                    last_reason, last_candidate = "no_question_mark", candidate
                    continue
                if args.require_paraphrase and _quotes_label(candidate, str(sample["row_label"])):
                    refuse(sample, attempt, "copied_row_label", candidate)
                    last_reason, last_candidate = "copied_row_label", candidate
                    continue
                question = candidate
                break
            if question is None:
                continue
            handle.write(json.dumps({**sample, "question": question}, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 100 == 0:
                print(f"  {position}/{len(samples)}", flush=True)
    if rejected_handle is not None:
        rejected_handle.close()

    total = sum(1 for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"rendered {total} questions -> {args.output}")
    if rejected:
        print("  rejected: " + ", ".join(f"{k} {v}" for k, v in sorted(rejected.items())))
    if args.rejected:
        print(f"  refused generations written to {args.rejected}")


if __name__ == "__main__":
    main()
