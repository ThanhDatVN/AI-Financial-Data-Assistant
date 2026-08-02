"""Typed scalar IR, Pandas compiler, and constrained execution."""

from vifinqa.programs.compiler import compile_expression
from vifinqa.programs.executor import execute_expression
from vifinqa.programs.ir import BinaryExpr, CellExpr, LiteralExpr, ScalarExpr

__all__ = [
    "BinaryExpr",
    "CellExpr",
    "LiteralExpr",
    "ScalarExpr",
    "compile_expression",
    "execute_expression",
]
