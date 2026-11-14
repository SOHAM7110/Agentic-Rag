"""Agent tools: calculator, structured lookup (facts table), compare-two-documents."""
from __future__ import annotations
import ast, operator as op
from dataclasses import dataclass, field
from ingest.facts import FACT_SPEC

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}


@dataclass
class ToolResult:
    tool: str
    ok: bool
    args: dict
    output: str = ""
    data: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)   # chunk ids that back this result
    error: str = ""

    def brief(self) -> dict:
        return {"tool": self.tool, "ok": self.ok, "args": self.args, "output": self.output[:300], "error": self.error}


def _eval(node):
    if isinstance(node, ast.Expression): return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS: return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS: return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> ToolResult:
    """Safe arithmetic evaluator (no names, no calls)."""
    try:
        val = _eval(ast.parse(expression.replace(",", ""), mode="eval"))
        if isinstance(val, float) and val.is_integer(): val = int(val)
        elif isinstance(val, float): val = round(val, 4)
        return ToolResult("calculator", True, {"expression": expression}, str(val), {"value": val})
    except Exception as e:
        return ToolResult("calculator", False, {"expression": expression}, error=str(e))


def lookup(retriever, doc_id: str, key: str) -> ToolResult:
    """Exact answer from the facts table extracted at ingestion."""
    f = retriever.facts.get(doc_id, {}).get(key)
    args = {"doc": doc_id, "key": key}
    if not f:
        return ToolResult("lookup", False, args, error="fact not in table")
    return ToolResult("lookup", True, args, f"{f['doc_title']}: {f['label']} = {f['raw']}", dict(f, doc=doc_id), [f["chunk_id"]])


def compare(retriever, doc_ids: list[str], key: str) -> ToolResult:
    """Retrieve the same aspect from several documents and contrast them."""
    rows, srcs = [], []
    for d in doc_ids:
        r = lookup(retriever, d, key)
        if not r.ok:
            return ToolResult("compare", False, {"docs": doc_ids, "key": key}, error=f"{d}: {r.error}")
        rows.append(r.data); srcs += r.sources
    return ToolResult("compare", True, {"docs": doc_ids, "key": key},
                      "; ".join(f"{r['doc_title']}: {r['raw']}" for r in rows), {"rows": rows, "key": key}, srcs)
