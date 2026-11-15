"""Assignment problem by the Hungarian method (exact arithmetic, every intermediate matrix recorded).

1. Balance a rectangular matrix with dummy rows / columns of zeros.
2. Maximisation: subtract every entry from the largest entry (opportunity-loss matrix).
3. Row reduction: subtract each row's minimum.   4. Column reduction: subtract each column's minimum.
5. Cover all zeros with the minimum number of lines (found from a maximum matching on zeros, Koenig's
   theorem).  If the number of lines equals n, an optimal assignment exists among the zeros.
6. Otherwise subtract the smallest uncovered value from every uncovered entry, add it to entries covered
   twice, and repeat step 5.
7. Make the assignment on zeros; the total comes from the original matrix.  If more than one complete
   assignment exists on the zeros, the optimum is not unique.
"""
from __future__ import annotations

import re
from copy import deepcopy
from fractions import Fraction

from .exact import MNum, fmt_q, json_num
from .matrix_parser import MatrixParseError, parse_assignment


def _max_matching(Z):
    n = len(Z)
    match_col = [-1] * n

    def try_row(i, seen):
        for j in range(n):
            if Z[i][j] and not seen[j]:
                seen[j] = True
                if match_col[j] == -1 or try_row(match_col[j], seen):
                    match_col[j] = i
                    return True
        return False
    for i in range(n):
        try_row(i, [False] * n)
    row_of = {j: i for j, i in enumerate(match_col) if i != -1}
    return {i: j for j, i in row_of.items()}


def _cover(Z, match):
    """Minimum line cover of the zeros (Koenig).  Returns (covered_rows, covered_cols)."""
    n = len(Z)
    col_match = {j: i for i, j in match.items()}
    vis_r, vis_c = set(), set()
    stack = [i for i in range(n) if i not in match]
    vis_r |= set(stack)
    while stack:
        i = stack.pop()
        for j in range(n):
            if Z[i][j] and j not in vis_c:
                vis_c.add(j)
                k = col_match.get(j)
                if k is not None and k not in vis_r:
                    vis_r.add(k)
                    stack.append(k)
    rows = {i for i in range(n) if i not in vis_r}
    return rows, vis_c


def _count_perfect(Z, limit=2):
    n = len(Z)
    count = 0

    def rec(i, used):
        nonlocal count
        if count >= limit:
            return
        if i == n:
            count += 1
            return
        for j in range(n):
            if Z[i][j] and j not in used:
                rec(i + 1, used | {j})
    rec(0, frozenset())
    return count


def solve(P0):
    P = deepcopy(P0)
    expl = []
    m, n = len(P["costs"]), len(P["costs"][0])
    if m != n:
        k = max(m, n)
        if m < k:
            for r in range(k - m):
                P["costs"].append([Fraction(0)] * n)
                P["row_labels"].append(f"Dummy{r + 1}" if k - m > 1 else "Dummy")
        if n < k:
            for row in P["costs"]:
                row += [Fraction(0)] * (k - n)
            P["col_labels"] += [f"Dummy{c + 1}" if k - n > 1 else "Dummy" for c in range(k - n)]
        expl.append(f"The matrix is {m} x {n}; dummy {'rows' if m < n else 'columns'} with zero cost make it square ({k} x {k}).")
    N = len(P["costs"])
    C = [list(r) for r in P["costs"]]
    if P["sense"] == "max":
        mx = max(c for r in C for c in r if not isinstance(c, MNum))
        C = [[(mx - c) if not isinstance(c, MNum) else c for c in r] for r in C]
        expl.append(f"Maximisation problem: every entry is subtracted from the largest value {fmt_q(mx)} to get an "
                    "opportunity-loss matrix, which is then minimised.")
    stages = [{"name": "Cost matrix" if P["sense"] == "min" else "Opportunity-loss matrix", "matrix": deepcopy(C)}]
    rmins = [min(r) for r in C]
    R = [[c - rmins[i] for c in C[i]] for i in range(N)]
    stages.append({"name": "Row reduction", "matrix": deepcopy(R), "subtracted": rmins})
    expl.append("Row reduction: subtract the row minima " + ", ".join(fmt_q(x) for x in rmins) + ".")
    cmins = [min(R[i][j] for i in range(N)) for j in range(N)]
    K = [[R[i][j] - cmins[j] for j in range(N)] for i in range(N)]
    stages.append({"name": "Column reduction", "matrix": deepcopy(K), "subtracted": cmins})
    expl.append("Column reduction: subtract the column minima " + ", ".join(fmt_q(x) for x in cmins) + ".")
    for it in range(4 * N + 5):
        Z = [[K[i][j] == 0 for j in range(N)] for i in range(N)]
        match = _max_matching(Z)
        rows, cols = _cover(Z, match)
        lines = len(rows) + len(cols)
        stages[-1]["lines"] = {"rows": sorted(rows), "cols": sorted(cols), "count": lines}
        if lines >= N:
            expl.append(f"All zeros can be covered with no fewer than {N} lines, so an optimal assignment exists.")
            break
        unc = [K[i][j] for i in range(N) for j in range(N) if i not in rows and j not in cols]
        theta = min(unc)
        if isinstance(theta, MNum) and theta.a != 0:
            raise MatrixParseError("Every remaining choice is a prohibited assignment: no feasible assignment exists.")
        expl.append(f"Zeros are covered by {lines} lines (< {N}). Smallest uncovered value = {fmt_q(theta)}: subtract it from "
                    "every uncovered entry and add it to the entries covered by two lines.")
        for i in range(N):
            for j in range(N):
                if i not in rows and j not in cols:
                    K[i][j] = K[i][j] - theta
                elif i in rows and j in cols:
                    K[i][j] = K[i][j] + theta
        stages.append({"name": f"Adjustment {it + 1}", "matrix": deepcopy(K), "theta": theta})
    Z = [[K[i][j] == 0 for j in range(N)] for i in range(N)]
    match = _max_matching(Z)
    assign = sorted(match.items())
    total = sum((P["costs"][i][j] for i, j in assign), Fraction(0))
    multiple = _count_perfect(Z) > 1
    infeasible = any(isinstance(P["costs"][i][j], MNum) for i, j in assign)
    expl.append("Optimal assignment: " + ", ".join(f"{P['row_labels'][i]} -> {P['col_labels'][j]} ({fmt_q(P['costs'][i][j])})"
                                                   for i, j in assign) + f". Total = {fmt_q(total)}.")
    if multiple:
        expl.append("The final matrix allows more than one complete assignment on zeros: alternative optimal assignments exist.")
    return {"method": "assignment", "problem": P, "sense": P["sense"], "stages": stages, "assignment": assign,
            "total": total, "status": "infeasible" if infeasible else ("multiple_optima" if multiple else "optimal"),
            "explanation": expl}


def to_json(res):
    P = res["problem"]
    return {"method": "assignment", "sense": res["sense"], "status": res["status"], "row_labels": P["row_labels"],
            "col_labels": P["col_labels"], "costs": [[json_num(x) for x in r] for r in P["costs"]],
            "stages": [{"name": s["name"], "matrix": [[json_num(x) for x in r] for r in s["matrix"]],
                        "lines": s.get("lines"), "theta": json_num(s.get("theta")),
                        "subtracted": [json_num(x) for x in s["subtracted"]] if s.get("subtracted") else None}
                       for s in res["stages"]],
            "assignment": [[i, j] for i, j in res["assignment"]], "total": json_num(res["total"]),
            "explanation": res["explanation"]}


# ----------------------------------------------------------------------------- verification
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
_TOK = r"([A-Za-z]{0,12}\s?(?:\d{1,2}|[IVX]{1,4})\b|\b[A-Z]\b)"
_PAIR = re.compile(_TOK + r"\s*(?:->|→|⟶|=>|-|–|—|:|to|assigned\s+to|gets|does|on|with)\s*" + _TOK)
_TOTAL = re.compile(r"(?i)(?:total|minimum|min\.?|optimal|optimum|maximum|max\.?)\s+(?:assignment\s+)?"
                    r"(?:cost|time|profit|value|sales|score)?[^=\n:]{0,30}?(?:=|:|is)\s*(?:Rs\.?|₹|\$)?\s*(-?[\d,]+(?:\.\d+)?)")


def _idx(labels, tok, prefer_letters=False):
    t = tok.strip().lower().replace(" ", "")
    for k, l in enumerate(labels):
        if l.lower().replace(" ", "") == t:
            return k
    m = re.fullmatch(r"[a-z]*?(\d{1,2})", t)
    if m and 1 <= int(m.group(1)) <= len(labels):
        return int(m.group(1)) - 1
    m = re.fullmatch(r"[a-z]*?([ivx]{1,4})", t)
    if m and m.group(1) in _ROMAN and _ROMAN[m.group(1)] <= len(labels):
        return _ROMAN[m.group(1)] - 1
    if len(t) == 1 and t.isalpha():
        k = ord(t) - ord("a")
        if 0 <= k < len(labels):
            return k
    return None


def parse_pairs(text, P):
    from .exact import clean_math_text
    t = clean_math_text(text)
    found = []
    for m in _PAIR.finditer(t):
        i, j = _idx(P["row_labels"], m.group(1)), _idx(P["col_labels"], m.group(2))
        if i is None or j is None:
            continue
        found.append((m.start(), i, j))
    groups, cur, last = [], {}, -10**9
    for pos, i, j in found:
        if cur and (pos - last > 300 or i in cur):
            groups.append(cur)
            cur = {}
        cur[i] = j
        last = pos
    if cur:
        groups.append(cur)
    full = [g for g in groups if len(g) >= min(len(P["row_labels"]), len(P["col_labels"])) - 1]
    return full[-1] if full else (groups[-1] if groups else {})


def _matrix_after(text, keyword_re, n):
    """First block of n consecutive lines with n numbers each, after a heading matching keyword_re."""
    from .exact import clean_math_text
    lines = clean_math_text(text).split("\n")
    for k, ln in enumerate(lines):
        if not re.search(keyword_re, ln, re.I):
            continue
        block = []
        for ln2 in lines[k + 1: k + 4 + 2 * n]:
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", ln2.replace("|", " "))
            if re.match(r"\s*\|?\s*[A-Za-z]", ln2) and len(nums) == n + 1:
                nums = nums[1:]
            if len(nums) == n:
                block.append([Fraction(x) for x in nums])
                if len(block) == n:
                    return block
            elif block:
                break
    return None


def analyze(question, answer, ai_source, refs=None):
    from .error_catalog import CATALOG
    from .step_verifier import Report, summarize
    try:
        P0 = parse_assignment(question)
    except MatrixParseError as e:
        from .report_generator import ReportError
        raise ReportError(str(e), "Write the cost matrix with one row per worker / job, numbers separated by spaces.")
    res = solve(P0)
    P = res["problem"]
    N = len(P["costs"])
    rep = Report()
    for key, stage_name, pat in (("row", "Row reduction", r"row[^\n]{0,40}(reduc|subtract|minim)"),
                                 ("col", "Column reduction", r"col(umn)?[^\n]{0,40}(reduc|subtract|minim)")):
        mat = _matrix_after(answer, pat, N)
        if mat is None:
            continue
        ours = next(s["matrix"] for s in res["stages"] if s["name"] == stage_name)
        diffs = [(i, j) for i in range(N) for j in range(N) if mat[i][j] != ours[i][j]]
        if diffs:
            i, j = diffs[0]
            rep.add("Hungarian method", stage_name, "incorrect", "H01",
                    f"After {stage_name.lower()}, the entry ({P['row_labels'][i]}, {P['col_labels'][j]}) should be "
                    f"{fmt_q(ours[i][j])}" + (f" ({len(diffs) - 1} more entries differ)" if len(diffs) > 1 else "") + ".",
                    ai=mat[i][j], correct=ours[i][j])
        else:
            rep.add("Hungarian method", stage_name, "correct", message=f"The {stage_name.lower()} matrix is correct.")
    pairs = parse_pairs(answer, P)
    totals = [Fraction(m.group(1).replace(",", "")) for m in _TOTAL.finditer(answer)]
    claim_total = totals[-1] if totals else None
    label = "unclear"
    real_rows = [i for i, l in enumerate(P["row_labels"]) if not l.startswith("Dummy")]
    if pairs:
        cols_used = list(pairs.values())
        if len(set(cols_used)) != len(cols_used):
            rep.add("Final assignment", "One-to-one assignment", "incorrect", "H01",
                    "Two rows are assigned to the same column; each column can be used only once.")
            label = "infeasible_solution"
        elif not all(i in pairs for i in real_rows):
            rep.add("Final assignment", "Complete assignment", "incomplete",
                    message="Not every row is assigned in the answer: missing " +
                            ", ".join(P["row_labels"][i] for i in real_rows if i not in pairs) + ".")
        else:
            tot = sum((P["costs"][i][j] for i, j in pairs.items()), Fraction(0))
            if claim_total is not None and tot != claim_total:
                rep.add("Final assignment", "Total arithmetic", "incorrect", "H01",
                        f"The assignment listed adds up to {fmt_q(tot)}.", ai=claim_total, correct=tot)
            worse = (tot < res["total"]) if res["sense"] == "max" else (tot > res["total"])
            if worse:
                rep.add("Final assignment", "Optimality", "incorrect", "H01",
                        f"This assignment totals {fmt_q(tot)}; the optimum is {fmt_q(res['total'])}.", ai=tot, correct=res["total"])
                label = "suboptimal_solution"
            else:
                rep.add("Final assignment", "Optimality", "correct", message="The assignment is optimal.")
                label = "correct"
    if claim_total is not None:
        if claim_total == res["total"]:
            rep.add("Final answer", "Optimal total", "correct", message=f"The optimal total {fmt_q(res['total'])} is correct.")
            label = "correct" if label in ("unclear", "correct") else label
        else:
            rep.add("Final answer", "Optimal total", "incorrect", "E12", f"The optimal total is {fmt_q(res['total'])}.",
                    ai=claim_total, correct=res["total"])
            label = "wrong_value" if label in ("unclear", "correct") else label
    elif label == "unclear":
        rep.add("Final answer", "Final answer", "unverifiable", message="No assignment or total could be read from the answer.")
    if res["status"] == "multiple_optima":
        rep.notes.append("Alternative optimal assignments exist; a different assignment with the same total is also correct.")
    summary = summarize(rep, label)
    codes = sorted({s["code"] for s in rep.steps if s["status"] == "incorrect" and s.get("code")})
    tb = {"loaded": bool(refs and refs.status()["loaded"]), "method_passages": [], "by_error": {}}
    if tb["loaded"]:
        tb["method_passages"] = refs.method_passages("assignment", question)
        tb["by_error"] = refs.for_errors("assignment", codes, question)
    LEG = {"verified": "valid", "partially_verified": "partial", "incorrect": "invalid", "incorrect_working": "invalid",
           "unverifiable": "unverifiable"}
    return {"method": "assignment", "method_name": "Assignment problem (Hungarian method)", "method_note": None,
            "ai_source": ai_source,
            "report": {"summary": summary, "steps": rep.steps, "tableaus": [], "notes": rep.notes,
                       "parsed": {"pairs": len(pairs), "totals": len(totals)}},
            "solution": to_json(res), "textbook": tb, "cross_check": _scipy_assign(P, res), "ml": None,
            "general_notes": {c: {"title": CATALOG[c]["title"], "note": CATALOG[c]["note"]} for c in codes},
            "verdict": LEG[summary["verdict"]], "label_text": summary["verdict_text"],
            "error_type": None if label == "unclear" else label,
            "truth": {"text": f"Optimal total = {fmt_q(res['total'])}", "status": res["status"]},
            "claim": {"text": f"Total = {fmt_q(claim_total)}" if claim_total is not None else "(no total found)"},
            "reasons": [], "references": [p for ps in tb["by_error"].values() for p in ps][:3], "plot": None, "checks": []}


def _scipy_assign(P, res):
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        C = np.array([[float(x) if not isinstance(x, MNum) else 1e9 for x in r] for r in P["costs"]])
        r, c = linear_sum_assignment(C, maximize=res["sense"] == "max")
        val = float(C[r, c].sum())
        ok = abs(val - float(res["total"])) < 1e-6
        return {"available": True, "agrees": bool(ok), "scipy_value": val,
                "note": ("SciPy (linear_sum_assignment) reaches the same optimal total." if ok else
                         "SciPy disagrees; please report this problem.") + " Final-answer cross-check only."}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "agrees": None, "note": f"SciPy cross-check unavailable: {e}"}


def solve_text(question):
    return {"method": "assignment", "method_name": "Assignment problem (Hungarian method)",
            "solution": to_json(solve(parse_assignment(question)))}
