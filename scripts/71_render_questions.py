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

QUESTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"question": {"type": "string", "minLength": 20, "maxLength": 400}},
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
}


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
    section = str(sample.get("section_title") or "").strip()
    lines = [
        f"Doanh nghiệp: {company}",
        f"Kỳ báo cáo: {period}",
        f"Khoản mục trong bảng: {sample['row_label']}",
        f"Nhãn cột: {sample['column_label']}",
        f"Đơn vị câu trả lời phải dùng: {sample['target_unit']}",
        f"Loại câu hỏi: {_FAMILY_BRIEF.get(str(sample['family']), '')}",
    ]
    if section:
        lines.insert(3, f"Mục của báo cáo: {section}")
    if scope:
        lines.insert(1, f"Phạm vi báo cáo phải nêu rõ trong câu hỏi: {scope}")
    else:
        lines.insert(1, "Phạm vi báo cáo: KHÔNG được nhắc tới trong câu hỏi.")
    lines.append(
        "\nYêu cầu bắt buộc:\n"
        "- Viết đúng MỘT câu hỏi, kết thúc bằng dấu hỏi.\n"
        "- Nêu rõ doanh nghiệp, kỳ báo cáo và đơn vị của đáp án.\n"
        "- KHÔNG chép nguyên văn cụm từ ở dòng 'Khoản mục trong bảng'. Hãy diễn đạt lại bằng "
        "thuật ngữ tài chính thông dụng mà người đọc báo cáo sẽ dùng.\n"
        "- KHÔNG nêu con số đáp án.\n"
        "- Dùng lối viết tự nhiên của đề thi tiếng Việt."
    )
    return "\n".join(lines)


def _is_proper_noun(row_label: str) -> bool:
    """Does this label name a specific entity rather than a financial concept?

    Vietnamese statement lines run lower case after the first word -- "Phải trả người bán ngắn
    hạn" -- while a named project or subsidiary keeps its capitals: "KĐT Mỹ Đình Nam Từ Liêm".
    """
    words = row_label.split()[1:]
    return sum(1 for word in words if word[:1].isupper()) >= 2


def _quotes_label(question: str, row_label: str) -> bool:
    """Reject a question that lifted the row label instead of paraphrasing it.

    Mapping everyday financial wording onto a statement's own line label is the hardest part of
    the real task. A question that copies the label teaches the easy half and hides the hard one.

    Names are the exception, and the real paper agrees: question 75 asks about the "Quỹ Đầu tư
    Giá trị Bảo Việt" by name, because there is no other way to say it.
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
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=2)
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
    with args.output.open("a", encoding="utf-8") as handle:
        for position, sample in enumerate(samples, start=1):
            sample_id = int(sample["id"])
            if sample_id in done:
                continue
            question = None
            for attempt in range(1, args.attempts + 1):
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": _prompt(sample, names, rng)},
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
                        rejected["empty"] += 1
                        continue
                    candidate = str(json.loads(content)["question"]).strip()
                except Exception as error:  # noqa: BLE001 - one lost sample, not a lost run
                    rejected[type(error).__name__] += 1
                    continue
                if not candidate.endswith("?"):
                    rejected["no_question_mark"] += 1
                    continue
                if _quotes_label(candidate, str(sample["row_label"])):
                    rejected["copied_row_label"] += 1
                    continue
                question = candidate
                break
            if question is None:
                continue
            handle.write(json.dumps({**sample, "question": question}, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 100 == 0:
                print(f"  {position}/{len(samples)}", flush=True)

    total = sum(1 for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"rendered {total} questions -> {args.output}")
    if rejected:
        print("  rejected: " + ", ".join(f"{k} {v}" for k, v in sorted(rejected.items())))


if __name__ == "__main__":
    main()
