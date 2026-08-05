from __future__ import annotations

import ast
import math
import multiprocessing as mp
import re
from collections.abc import Mapping
from multiprocessing.connection import Connection

import pandas as pd

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.IfExp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.List,
    ast.Tuple,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.BitAnd,
    ast.BitOr,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.And,
    ast.Or,
    # A cohort program reads the same threshold and the same membership test many times
    # over. Without a way to name a value once, the compiler has to inline it at every use
    # and a nine-company selection expands to megabytes. Binding is confined to the reserved
    # names below, which cannot shadow evidence or an allowlisted function.
    ast.NamedExpr,
    ast.Store,
)
_BINDING_RE = re.compile(r"^_v[0-9]+$")
_SAFE_FUNCTIONS: dict[str, object] = {
    "abs": abs,
    "float": float,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sum": sum,
}


def _validate(tree: ast.AST, variables: set[str]) -> None:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.NamedExpr):
            if not isinstance(node.target, ast.Name) or not _BINDING_RE.fullmatch(node.target.id):
                raise ValueError("Only reserved _vN names may be bound")
            if node.target.id in variables | _SAFE_FUNCTIONS.keys():
                raise ValueError(f"Binding shadows an existing name: {node.target.id}")
            bound.add(node.target.id)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in variables | _SAFE_FUNCTIONS.keys() | bound:
            raise ValueError(f"Unknown name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"Private/dunder attribute is forbidden: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError(f"Function is not allowlisted: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr not in {
                "astype",
                "fillna",
                "iloc",
                "item",
                "max",
                "mean",
                "median",
                "min",
                "round",
                "sum",
            }:
                raise ValueError(f"Method is not allowlisted: {node.func.attr}")


def execute_expression(expression: str, frames: Mapping[str, pd.DataFrame]) -> float:
    tree = ast.parse(expression, mode="eval")
    _validate(tree, set(frames))
    environment: dict[str, object] = {**_SAFE_FUNCTIONS, **frames}
    result = eval(compile(tree, "<pandas_query>", "eval"), {"__builtins__": {}}, environment)
    if isinstance(result, bool):
        raise ValueError("Boolean result is not a numeric answer")
    try:
        value = float(result)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Query result is not scalar numeric: {result!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Query result is not finite: {value}")
    return value


def _isolated_worker(
    connection: Connection,
    expression: str,
    frames: dict[str, pd.DataFrame],
    memory_limit_mb: int | None,
) -> None:
    try:
        if memory_limit_mb is not None:
            try:
                import resource
            except ImportError as exc:
                raise RuntimeError("Memory limits require a POSIX runtime") from exc
            memory_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(  # type: ignore[attr-defined]
                resource.RLIMIT_AS,  # type: ignore[attr-defined]
                (memory_bytes, memory_bytes),
            )
        connection.send(("ok", execute_expression(expression, frames)))
    except BaseException as exc:  # noqa: BLE001 - transport remote worker failures
        connection.send(("error", type(exc).__name__, str(exc)))
    finally:
        connection.close()


def execute_expression_isolated(
    expression: str,
    frames: Mapping[str, pd.DataFrame],
    *,
    timeout_seconds: float = 10.0,
    memory_limit_mb: int | None = None,
) -> float:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    if memory_limit_mb is not None and memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive")
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_worker,
        args=(writer, expression, dict(frames), memory_limit_mb),
        daemon=True,
    )
    process.start()
    writer.close()
    try:
        if not reader.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=1.0)
            raise TimeoutError(f"Expression exceeded {timeout_seconds:g} seconds")
        message = reader.recv()
    except EOFError as exc:
        raise RuntimeError(
            f"Isolated executor exited without a result; exitcode={process.exitcode}"
        ) from exc
    finally:
        reader.close()
        if process.is_alive():
            process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
    if message[0] == "ok":
        return float(message[1])
    raise ValueError(f"Isolated executor {message[1]}: {message[2]}")
