from __future__ import annotations

import math
import re
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

_VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable: str
    csv_path: str

    @field_validator("variable")
    @classmethod
    def validate_variable(cls, value: str) -> str:
        if not _VARIABLE_RE.fullmatch(value):
            raise ValueError("must be a valid Python variable name")
        return value

    @field_validator("csv_path")
    @classmethod
    def validate_csv_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("must use POSIX '/' separators")
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or path.parts[0] != "data":
            raise ValueError("must be relative and start with data/")
        if ".." in path.parts or path.suffix.lower() != ".csv":
            raise ValueError("must be a traversal-free CSV path")
        return value


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    question: str
    answer: float
    relevant_docs: list[str]
    relevant_tables: list[str]
    evidence: list[Evidence]
    pandas_query: str

    @field_validator("answer")
    @classmethod
    def finite_answer(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @field_validator("relevant_docs", "relevant_tables")
    @classmethod
    def unique_non_empty_strings(cls, values: list[str]) -> list[str]:
        if not values or any(not value.strip() for value in values):
            raise ValueError("must contain non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError("must not contain duplicates")
        return values

    @model_validator(mode="after")
    def cross_validate(self, info: ValidationInfo) -> Prediction:
        variables = [item.variable for item in self.evidence]
        if not variables or len(variables) != len(set(variables)):
            raise ValueError("evidence variables must be non-empty and unique")
        # The two reference lists are scored apart: five submissions that rewrote every table
        # reference — including to strings naming no real table — left the document score at
        # exactly 0.4833. So the depth that suits one list need not suit the other, and a
        # submission may deliberately cite fewer documents than its tables span. Requiring
        # containment is still right for anything built normally, so it stays on by default.
        context = info.context if isinstance(info.context, dict) else {}
        require_doc_coverage = not context.get("allow_partial_docs", False)
        docs = set(self.relevant_docs)
        for table_ref in self.relevant_tables:
            doc_id, separator, position = table_ref.partition("|")
            if not separator or not doc_id or not position:
                raise ValueError(f"invalid relevant_tables item: {table_ref!r}")
            if require_doc_coverage and doc_id not in docs:
                raise ValueError(f"table document {doc_id!r} missing from relevant_docs")
        if not self.pandas_query.strip():
            raise ValueError("pandas_query must not be empty")
        return self
