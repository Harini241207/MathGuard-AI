"""Parse transportation and assignment problems written as text.

Accepted shapes (whitespace, commas, tabs or Markdown pipes):

    Transportation (supply as the last column, demand as the last row):
             D1  D2  D3  D4  Supply
        S1   19  30  50  10   7
        S2   70  30  40  60   9
        S3   40   8  70  20  18
        Demand 5  8   7  14

    or the cost rows followed by "Supply: 7, 9, 18" and "Demand: 5, 8, 7, 14".

    Assignment (a square or rectangular cost matrix, optional labels):
             M1  M2  M3
        J1    9  11  14
        J2    6  15  13
        J3   12  13   6

    A '-' / 'x' / 'M' / 'inf' cell means the cell is not allowed (prohibited route / assignment).
"""
from __future__ import annotations

import re
from fractions import Fraction

from .exact import MNum, clean_math_text, to_q

FORBID = MNum(1, 0)  # a prohibited cell gets cost M (symbolic, larger than any real cost)


class MatrixParseError(ValueError):
    pass


_NUM = r"-?\d+(?:\.\d+)?(?:/\d+)?"


def _tokens(line):
    line = clean_math_text(line).replace("|", " ").replace(",", " ").replace(";", " ").replace("\t", " ")
    return [t for t in line.split() if t]


def _is_num(t):
    return re.fullmatch(_NUM, t) is not None


def _is_forbid(t):
    return t.lower() in ("-", "x", "m", "inf", "∞", "--", "na", "n/a", "×")


def _row_values(toks):
    vals = []
    for t in toks:
        if _is_num(t):
            vals.append(to_q(t) if "/" not in t else Fraction(t))
        elif _is_forbid(t):
            vals.append(FORBID)
    return vals


def sense_of(text):
    low = text.lower()
    if re.search(r"\bmaxim|\bprofit|\brevenue|\bsales\b|\bscore|\bperformance", low) and not re.search(r"\bminim", low):
        return "max"
    return "min"


def parse_transportation(text):
    if not text or not text.strip():
        raise MatrixParseError("The question is empty.")
    lines = [l for l in text.replace("\r", "").split("\n")]
    supply, demand, rows, row_labels, col_labels = None, None, [], [], None
    supply_col = False
    for ln in lines:
        low = ln.lower()
        toks = _tokens(ln)
        if not toks:
            continue
        nums = _row_values(toks)
        if re.match(r"\s*\|?\s*(demand|requirement|required|req\.?|destination\s+requirement|b_?j)\b", low):
            demand = [v for v in nums if not isinstance(v, MNum)]
            continue
        m_sup = re.match(r"\s*\|?\s*(supply|availability|available|capacity|a_?i)\b", low)
        if m_sup and len(nums) >= 1 and not re.search(r"[A-Za-z]\d", " ".join(toks[1:])):
            supply = [v for v in nums if not isinstance(v, MNum)]
            continue
        if re.search(r"(?i)\b(supply|availability|capacity)\b", ln) and sum(1 for t in toks if _is_num(t)) == 0:
            supply_col = True
            heads = [t for t in toks if not re.match(r"(?i)supply|availability|capacity|source|from|to|cost", t)]
            col_labels = heads or None
            continue
        if sum(1 for t in toks if _is_num(t)) == 0 and len(toks) >= 2:
            if col_labels is None:
                col_labels = [t for t in toks if not re.match(r"(?i)source|from|to|cost|destination|origin", t)]
            continue
        if len(nums) >= 2:
            lab = toks[0] if not (_is_num(toks[0]) or _is_forbid(toks[0])) else None
            rows.append(nums)
            row_labels.append(lab)
    if not rows:
        raise MatrixParseError("No cost matrix found. Put one row of costs per source on its own line.")
    width = max(len(r) for r in rows)
    if demand is None:
        # the last numeric row may be the demand row (without a label)
        raise MatrixParseError("No demand row found. Add a line starting with 'Demand' (for example 'Demand 5 8 7 14').")
    n = len(demand)
    if supply is None:
        if all(len(r) == n + 1 for r in rows):
            supply = [r[-1] for r in rows]
            rows = [r[:-1] for r in rows]
        else:
            raise MatrixParseError("No supply values found. Add them as the last column or as a line 'Supply: ...'.")
    elif any(len(r) == n + 1 for r in rows) and supply_col:
        rows = [r[:n] for r in rows]
    if any(len(r) != n for r in rows):
        raise MatrixParseError(f"Every cost row needs {n} entries (one per destination); got row lengths "
                               f"{[len(r) for r in rows]}.")
    if len(supply) != len(rows):
        raise MatrixParseError(f"{len(rows)} cost rows but {len(supply)} supply values.")
    if any(isinstance(v, MNum) for v in supply + demand) or any(v < 0 for v in supply + demand):
        raise MatrixParseError("Supply and demand must be non-negative numbers.")
    rl = [l or f"S{i + 1}" for i, l in enumerate(row_labels)]
    cl = (col_labels or [])[:n]
    if len(cl) != n:
        cl = [f"D{j + 1}" for j in range(n)]
    return {"costs": rows, "supply": supply, "demand": demand, "row_labels": rl, "col_labels": cl, "sense": sense_of(text)}


def parse_assignment(text):
    if not text or not text.strip():
        raise MatrixParseError("The question is empty.")
    rows, labels, col_labels = [], [], None
    for ln in text.replace("\r", "").split("\n"):
        toks = _tokens(ln)
        if not toks:
            continue
        nums = _row_values(toks)
        if len(nums) >= 2 and len(nums) >= len(toks) - 1:
            lab = toks[0] if not (_is_num(toks[0]) or _is_forbid(toks[0])) else None
            rows.append(nums)
            labels.append(lab)
        elif len(nums) == 0 and len(toks) >= 2 and not rows:
            col_labels = [t for t in toks if not re.match(r"(?i)^(cost|time|jobs?|workers?|machines?|persons?|tasks?|"
                                                         r"assignment|matrix|minimi[sz]e|maximi[sz]e|the|of|and|to)$", t)]
    if not rows:
        raise MatrixParseError("No cost matrix found. Put one row of costs per worker / job on its own line.")
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise MatrixParseError(f"All rows must have the same number of entries; got {[len(r) for r in rows]}.")
    rl = [l or f"R{i + 1}" for i, l in enumerate(labels)]
    cl = (col_labels or [])
    if len(cl) != n:
        cl = [f"C{j + 1}" for j in range(n)]
    return {"costs": rows, "row_labels": rl, "col_labels": cl, "sense": sense_of(text)}
