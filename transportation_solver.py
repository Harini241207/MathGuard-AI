"""Transportation problem: initial basic feasible solution (NWC, LCM, VAM) and MODI optimisation.

Textbook procedure, exact arithmetic:

* Balance the problem (dummy source / destination with zero cost) if total supply != total demand.
* Maximisation (profit) problems are converted to an opportunity-loss matrix (max profit - profit).
* Initial solution by North-West Corner, Least Cost (matrix minimum) or Vogel's Approximation.
* Degeneracy: if fewer than m + n - 1 cells are occupied, epsilon (zero) allocations are placed in the
  cheapest independent cells (cells that do not close a loop) so the basic cells form a spanning tree.
* MODI (u-v) method: ui + vj = cij on basic cells (u = 0 for the row with most allocations), opportunity
  costs dij = cij - (ui + vj) on empty cells; the most negative dij enters, theta is found on the closed
  loop, and the iteration repeats until every dij >= 0.  dij = 0 on an empty cell signals alternative optima.
"""
from __future__ import annotations

import re
from collections import deque
from copy import deepcopy
from fractions import Fraction

from .exact import MNum, fmt_q, json_num, sign
from .matrix_parser import FORBID, MatrixParseError, parse_transportation

EPS = "eps"


def _cell_name(P, i, j):
    return f"({P['row_labels'][i]}, {P['col_labels'][j]})"


def balance(P):
    S, D = sum(P["supply"]), sum(P["demand"])
    notes = []
    P = deepcopy(P)
    if S > D:
        P["costs"] = [r + [Fraction(0)] for r in P["costs"]]
        P["demand"].append(S - D)
        P["col_labels"].append("Dummy")
        notes.append(f"Total supply {fmt_q(S)} exceeds total demand {fmt_q(D)}: a dummy destination with demand "
                     f"{fmt_q(S - D)} and zero costs is added.")
    elif D > S:
        P["costs"].append([Fraction(0)] * len(P["demand"]))
        P["supply"].append(D - S)
        P["row_labels"].append("Dummy")
        notes.append(f"Total demand {fmt_q(D)} exceeds total supply {fmt_q(S)}: a dummy source with supply "
                     f"{fmt_q(D - S)} and zero costs is added.")
    else:
        notes.append(f"Total supply = total demand = {fmt_q(S)}: the problem is balanced.")
    return P, notes


def to_cost_problem(P):
    """Profit maximisation -> opportunity-loss minimisation."""
    if P["sense"] != "max":
        return P, None
    vals = [c for r in P["costs"] for c in r if not isinstance(c, MNum)]
    mx = max(vals)
    Q = deepcopy(P)
    Q["costs"] = [[(mx - c) if not isinstance(c, MNum) else c for c in r] for r in P["costs"]]
    return Q, (f"This is a maximisation (profit) problem: each profit is subtracted from the largest profit {fmt_q(mx)} "
               "to get an opportunity-loss matrix, which is then minimised.")


# ----------------------------------------------------------------------------- initial solutions
def northwest(P):
    m, n = len(P["supply"]), len(P["demand"])
    s, d = list(P["supply"]), list(P["demand"])
    alloc = [[None] * n for _ in range(m)]
    steps = []
    i = j = 0
    while i < m and j < n:
        q = min(s[i], d[j])
        alloc[i][j] = q
        s[i] -= q
        d[j] -= q
        steps.append({"cell": [i, j], "amount": q,
                      "text": f"North-west corner cell {_cell_name(P, i, j)}: allocate min({fmt_q(q + s[i])}, {fmt_q(q + d[j])}) = {fmt_q(q)}."})
        if s[i] == 0 and d[j] == 0:
            i += 1
            j += 1
        elif s[i] == 0:
            i += 1
        else:
            j += 1
    return alloc, steps


def least_cost(P):
    m, n = len(P["supply"]), len(P["demand"])
    s, d = list(P["supply"]), list(P["demand"])
    alloc = [[None] * n for _ in range(m)]
    rows, cols = set(range(m)), set(range(n))
    steps = []
    while rows and cols:
        cells = [(i, j) for i in rows for j in cols]
        best = min(cells, key=lambda c: (P["costs"][c[0]][c[1]], -min(s[c[0]], d[c[1]]), c))
        i, j = best
        q = min(s[i], d[j])
        alloc[i][j] = q
        steps.append({"cell": [i, j], "amount": q,
                      "text": f"Lowest remaining cost {fmt_q(P['costs'][i][j])} at {_cell_name(P, i, j)}: allocate {fmt_q(q)}."})
        s[i] -= q
        d[j] -= q
        if s[i] == 0 and d[j] == 0:
            rows.discard(i)
            cols.discard(j)
        elif s[i] == 0:
            rows.discard(i)
        else:
            cols.discard(j)
    return alloc, steps


def vogel(P):
    m, n = len(P["supply"]), len(P["demand"])
    C = P["costs"]
    s, d = list(P["supply"]), list(P["demand"])
    alloc = [[None] * n for _ in range(m)]
    rows, cols = set(range(m)), set(range(n))
    steps = []

    def pen(vals):
        v = sorted(vals)
        return v[1] - v[0] if len(v) >= 2 else v[0]

    while rows and cols:
        if len(rows) == 1 or len(cols) == 1:
            # only one line left: allocate in order of cost
            for i in sorted(rows):
                for j in sorted(cols, key=lambda jj: C[i][jj]):
                    if s[i] == 0 or d[j] == 0:
                        continue
                    q = min(s[i], d[j])
                    alloc[i][j] = q
                    s[i] -= q
                    d[j] -= q
                    steps.append({"cell": [i, j], "amount": q,
                                  "text": f"Only one line remains: allocate {fmt_q(q)} to {_cell_name(P, i, j)}."})
            break
        rp = {i: pen([C[i][j] for j in cols]) for i in rows}
        cp = {j: pen([C[i][j] for i in rows]) for j in cols}
        lines = [("row", i, rp[i]) for i in rows] + [("col", j, cp[j]) for j in cols]
        top = max(p for _, _, p in lines)
        tied = [l for l in lines if l[2] == top]

        def line_min(l):
            if l[0] == "row":
                return min(C[l[1]][j] for j in cols)
            return min(C[i][l[1]] for i in rows)
        kind, k, p = min(tied, key=lambda l: (line_min(l), 0 if l[0] == "row" else 1, l[1]))
        if kind == "row":
            i = k
            j = min(cols, key=lambda jj: (C[i][jj], jj))
        else:
            j = k
            i = min(rows, key=lambda ii: (C[ii][j], ii))
        q = min(s[i], d[j])
        alloc[i][j] = q
        steps.append({"cell": [i, j], "amount": q, "row_penalties": {P["row_labels"][r]: rp[r] for r in sorted(rows)},
                      "col_penalties": {P["col_labels"][c]: cp[c] for c in sorted(cols)},
                      "text": _vam_text(P, rows, cols, rp, cp, p, kind, k, i, j, q)})
        s[i] -= q
        d[j] -= q
        if s[i] == 0 and d[j] == 0:
            rows.discard(i)
            cols.discard(j)
        elif s[i] == 0:
            rows.discard(i)
        else:
            cols.discard(j)
    return alloc, steps


def _vam_text(P, rows, cols, rp, cp, p, kind, k, i, j, q):
    rtxt = ", ".join("%s=%s" % (P["row_labels"][r], fmt_q(rp[r])) for r in sorted(rows))
    ctxt = ", ".join("%s=%s" % (P["col_labels"][c], fmt_q(cp[c])) for c in sorted(cols))
    where = ("row " + P["row_labels"][k]) if kind == "row" else ("column " + P["col_labels"][k])
    return (f"Penalties: rows {rtxt}; columns {ctxt}. Largest penalty {fmt_q(p)} in {where}; "
            f"its lowest cost cell is {_cell_name(P, i, j)}: allocate {fmt_q(q)}.")


INITIAL = {"nwc": ("North-West Corner method", northwest), "lcm": ("Least Cost method", least_cost),
           "vam": ("Vogel's Approximation method", vogel)}


# ----------------------------------------------------------------------------- degeneracy + MODI
class _DSU:
    def __init__(self, k):
        self.p = list(range(k))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def fix_degeneracy(P, alloc):
    m, n = len(alloc), len(alloc[0])
    basic = [(i, j) for i in range(m) for j in range(n) if alloc[i][j] is not None]
    dsu = _DSU(m + n)
    for i, j in basic:
        dsu.union(i, m + j)
    added = []
    if len(basic) < m + n - 1:
        cand = sorted(((P["costs"][i][j], i, j) for i in range(m) for j in range(n) if alloc[i][j] is None),
                      key=lambda t: (t[0], t[1], t[2]))
        for _, i, j in cand:
            if len(basic) + len(added) >= m + n - 1:
                break
            if dsu.union(i, m + j):
                alloc[i][j] = Fraction(0)
                added.append((i, j))
    return added


def _tree_path(basic, m, n, start_row, end_col):
    """Path of basic cells from row node start_row to column node end_col in the basis tree."""
    adj = {}
    for i, j in basic:
        adj.setdefault(("r", i), []).append(("c", j))
        adj.setdefault(("c", j), []).append(("r", i))
    prev = {("r", start_row): None}
    q = deque([("r", start_row)])
    while q:
        u = q.popleft()
        if u == ("c", end_col):
            break
        for w in adj.get(u, []):
            if w not in prev:
                prev[w] = u
                q.append(w)
    if ("c", end_col) not in prev:
        return None
    nodes = []
    u = ("c", end_col)
    while u is not None:
        nodes.append(u)
        u = prev[u]
    nodes.reverse()  # r_start ... c_end
    cells = []
    for a, b in zip(nodes, nodes[1:]):
        if a[0] == "r":
            cells.append((a[1], b[1]))
        else:
            cells.append((b[1], a[1]))
    return cells


def uv(P, basic):
    m, n = len(P["supply"]), len(P["demand"])
    count = [sum(1 for (i, _) in basic if i == r) for r in range(m)]
    r0 = max(range(m), key=lambda r: (count[r], -r))
    u, v = [None] * m, [None] * n
    u[r0] = Fraction(0)
    changed = True
    while changed:
        changed = False
        for i, j in basic:
            c = P["costs"][i][j]
            if u[i] is not None and v[j] is None:
                v[j] = c - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = c - v[j]
                changed = True
    return u, v, r0


def modi(P, alloc, max_iters=50):
    m, n = len(alloc), len(alloc[0])
    iters = []
    for _ in range(max_iters):
        basic = [(i, j) for i in range(m) for j in range(n) if alloc[i][j] is not None]
        u, v, r0 = uv(P, basic)
        d = [[None] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if alloc[i][j] is None:
                    d[i][j] = P["costs"][i][j] - (u[i] + v[j])
        cost = total_cost(P, alloc)
        rec = {"alloc": deepcopy(alloc), "u": list(u), "v": list(v), "u0_row": r0, "d": deepcopy(d), "cost": cost}
        neg = [(d[i][j], i, j) for i in range(m) for j in range(n) if d[i][j] is not None and sign(d[i][j]) < 0]
        if not neg:
            rec["optimal"] = True
            rec["alternative"] = any(d[i][j] is not None and sign(d[i][j]) == 0 for i in range(m) for j in range(n))
            iters.append(rec)
            return iters
        _, ei, ej = min(neg, key=lambda t: (t[0], t[1], t[2]))
        path = _tree_path(basic, m, n, ei, ej)
        loop = [((ei, ej), "+")] + [(c, "-" if k % 2 == 0 else "+") for k, c in enumerate(path)]
        minus = [c for c, sgn in loop if sgn == "-"]
        theta = min(alloc[i][j] for i, j in minus)
        leave = next(c for c in minus if alloc[c[0]][c[1]] == theta)
        rec.update(optimal=False, entering=[ei, ej], loop=[{"cell": list(c), "sign": sgn} for c, sgn in loop],
                   theta=theta, leaving=list(leave))
        iters.append(rec)
        alloc[ei][ej] = Fraction(0)
        for (i, j), sgn in loop:
            alloc[i][j] = alloc[i][j] + theta if sgn == "+" else alloc[i][j] - theta
        alloc[leave[0]][leave[1]] = None
    raise RuntimeError("MODI did not converge within the iteration limit.")


def total_cost(P, alloc):
    t = Fraction(0)
    for i, row in enumerate(alloc):
        for j, a in enumerate(row):
            if a:
                t = t + P["costs"][i][j] * a
    return t


def solve(P0, initial="vam"):
    explanation = []
    Pb, bnotes = balance(P0)
    explanation += bnotes
    P, conv = to_cost_problem(Pb)
    if conv:
        explanation.append(conv)
    name, fn = INITIAL[initial]
    alloc, steps = fn(P)
    init_cost = total_cost(Pb, alloc)
    explanation.append(f"Initial basic feasible solution by the {name}: " + " ".join(s["text"] for s in steps))
    explanation.append(f"Initial {'profit' if P0['sense'] == 'max' else 'cost'} = {fmt_q(init_cost)}.")
    m, n = len(alloc), len(alloc[0])
    occupied = sum(1 for r in alloc for a in r if a is not None)
    added = fix_degeneracy(P, alloc)
    if added:
        explanation.append(f"Only {occupied} cells are occupied but m + n - 1 = {m + n - 1} are needed: the solution is "
                           "degenerate. An epsilon (zero) allocation is placed in " +
                           ", ".join(_cell_name(P, i, j) for i, j in added) + " (cheapest cells that do not form a loop).")
    initial_alloc = deepcopy(alloc)
    iters = modi(P, alloc)
    for k, it in enumerate(iters):
        us = ", ".join(f"u{i + 1} = {fmt_q(x)}" for i, x in enumerate(it["u"]))
        vs = ", ".join(f"v{j + 1} = {fmt_q(x)}" for j, x in enumerate(it["v"]))
        neg = [(i, j) for i in range(m) for j in range(n) if it["d"][i][j] is not None and sign(it["d"][i][j]) < 0]
        txt = f"MODI iteration {k + 1}: from ui + vj = cij on occupied cells: {us}; {vs}. "
        if it["optimal"]:
            txt += "All opportunity costs dij = cij - (ui + vj) are >= 0, so the solution is optimal."
            if it["alternative"]:
                txt += " Some empty cell has dij = 0, so an alternative optimal solution exists."
        else:
            ei, ej = it["entering"]
            txt += (f"Negative opportunity costs at {', '.join(_cell_name(P, i, j) for i, j in neg)}; the most negative, "
                    f"d = {fmt_q(it['d'][ei][ej])} at {_cell_name(P, ei, ej)}, enters. Closed loop: " +
                    " -> ".join(f"{_cell_name(P, *c['cell'])}({c['sign']})" for c in it["loop"]) +
                    f". Theta = {fmt_q(it['theta'])}; {_cell_name(P, *it['leaving'])} leaves.")
        explanation.append(txt)
    final = iters[-1]
    fcost = total_cost(Pb, final["alloc"])
    forbidden_used = any(final["alloc"][i][j] and isinstance(P["costs"][i][j], MNum) for i in range(m) for j in range(n))
    status = "infeasible" if forbidden_used else ("multiple_optima" if final.get("alternative") else "optimal")
    if forbidden_used:
        explanation.append("The optimum still uses a prohibited route (cost M): there is no feasible schedule that avoids it.")
    explanation.append(f"Optimal {'profit' if P0['sense'] == 'max' else 'cost'} = {fmt_q(fcost)}.")
    return {"method": "transportation", "initial_method": initial, "initial_method_name": name, "problem": Pb,
            "cost_problem": P, "sense": P0["sense"], "balance_notes": bnotes, "initial_steps": steps,
            "initial_alloc": initial_alloc, "initial_cost": init_cost, "epsilon_cells": added, "iterations": iters,
            "final_alloc": final["alloc"], "optimal_cost": fcost, "status": status, "explanation": explanation}


def to_json(res):
    P = res["problem"]

    def mat(a):
        return [[None if x is None else json_num(x) for x in r] for r in a]
    return {
        "method": "transportation", "sense": res["sense"], "status": res["status"],
        "row_labels": P["row_labels"], "col_labels": P["col_labels"],
        "costs": mat(P["costs"]), "supply": [json_num(x) for x in P["supply"]], "demand": [json_num(x) for x in P["demand"]],
        "initial_method": res["initial_method_name"], "initial_steps": [s["text"] for s in res["initial_steps"]],
        "initial_alloc": mat(res["initial_alloc"]), "initial_cost": json_num(res["initial_cost"]),
        "epsilon_cells": [list(c) for c in res["epsilon_cells"]],
        "iterations": [{
            "alloc": mat(it["alloc"]), "u": [json_num(x) for x in it["u"]], "v": [json_num(x) for x in it["v"]],
            "d": mat(it["d"]), "optimal": it["optimal"], "entering": it.get("entering"), "leaving": it.get("leaving"),
            "theta": json_num(it.get("theta")), "loop": it.get("loop"), "cost": json_num(it["cost"])} for it in res["iterations"]],
        "final_alloc": mat(res["final_alloc"]), "optimal_cost": json_num(res["optimal_cost"]),
        "explanation": res["explanation"],
    }


# ----------------------------------------------------------------------------- verification of an AI answer
_COST = re.compile(r"(?i)(?:total|minimum|min\.?|optimal|optimum|final|maximum|max\.?|net)\s+(?:transportation\s+|shipping\s+)?"
                   r"(?:cost|profit)[^=\n:]{0,40}?(?:=|:|is)\s*(?:Rs\.?|₹|\$|INR)?\s*([\d,]+(?:\.\d+)?)")
_INIT_COST = re.compile(r"(?i)(?:initial|ibfs|north|nwc|least|lcm|vogel|vam|matrix minimum)[^\n]{0,80}?(?:cost|profit)[^=\n:\d]{0,30}?"
                        r"(?:=|:|is|of)?\s*(?:Rs\.?|₹|\$|INR)?\s*(\d[\d,]*(?:\.\d+)?)")


def _label_idx(lbls, tok):
    t = tok.lower().strip()
    for k, l in enumerate(lbls):
        if l.lower() == t:
            return k
    m = re.fullmatch(r"[a-z]*?(\d+)", t)
    if m and 1 <= int(m.group(1)) <= len(lbls):
        return int(m.group(1)) - 1
    return None


def parse_allocations(text, P):
    """Allocation statements in the answer -> list of clusters [{(i, j): amount}] in order of appearance."""
    from .exact import clean_math_text
    t = clean_math_text(text)
    t = re.sub(r"\b([A-Za-z])_\{?(\d)\}?\{?(\d)\}?", r"\1\2\3", t)
    R, Cc = P["row_labels"], P["col_labels"]
    rl = "|".join(re.escape(x) for x in sorted(R, key=len, reverse=True))
    cl = "|".join(re.escape(x) for x in sorted(Cc, key=len, reverse=True))
    pats = [
        re.compile(r"(?i)\b(" + rl + r"|[A-Za-z]{1,8}\s?\d)\s*(?:->|→|to|-|,)\s*(" + cl + r"|[A-Za-z]{1,8}\s?\d)\s*\)?\s*(?:=|:|-)?\s*\(?\s*([\d.]+)\s*(?:units?)?"),
        re.compile(r"(?i)\bx\s?(\d)(\d)\s*=\s*([\d.]+)"),
        re.compile(r"(?i)\(\s*(\d)\s*,\s*(\d)\s*\)\s*(?:=|:)\s*([\d.]+)"),
    ]
    found = []
    for k, p in enumerate(pats):
        for m in p.finditer(t):
            if k == 0:
                i, j = _label_idx(R, m.group(1)), _label_idx(Cc, m.group(2))
            else:
                i, j = int(m.group(1)) - 1, int(m.group(2)) - 1
            if i is None or j is None or not (0 <= i < len(R) and 0 <= j < len(Cc)):
                continue
            try:
                q = Fraction(m.group(3))
            except (ValueError, ZeroDivisionError):
                continue
            found.append((m.start(), i, j, q))
    found.sort()
    clusters, cur, last = [], {}, -10**9
    for pos, i, j, q in found:
        if cur and (pos - last > 350 or (i, j) in cur):
            clusters.append(cur)
            cur = {}
        cur[(i, j)] = q
        last = pos
    if cur:
        clusters.append(cur)
    return clusters


def _num(s):
    return Fraction(s.replace(",", ""))


def analyze(question, answer, ai_source, refs=None):
    from .error_catalog import CATALOG
    from .step_verifier import Report, summarize
    try:
        P0 = parse_transportation(question)
    except MatrixParseError as e:
        from .report_generator import ReportError
        raise ReportError(str(e), "Write the costs one source per line, with Supply as the last column and a 'Demand' row.")
    low = answer.lower()
    init = "vam" if re.search(r"vogel|\bvam\b", low) else "lcm" if re.search(r"least\s+cost|\blcm\b|matrix\s+minimum", low) \
        else "nwc" if re.search(r"north[\s-]*west|\bnwc\b|\bnwcm\b", low) else "vam"
    res = solve(P0, init)
    P = res["problem"]
    rep = Report()
    opt = res["optimal_cost"]
    # 1. balancing
    if res["balance_notes"] and "dummy" in res["balance_notes"][0].lower():
        if "dummy" in low:
            rep.add("Setup", "Balancing", "correct", message=res["balance_notes"][0])
        else:
            rep.add("Setup", "Balancing", "incomplete", message="The problem is unbalanced; the answer does not mention adding a "
                                                                 "dummy row or column. " + res["balance_notes"][0])
    # 2. initial solution cost
    init_claims = [m for m in _INIT_COST.finditer(answer)]
    if init_claims:
        v = _num(init_claims[0].group(1))
        if v == res["initial_cost"]:
            rep.add("Initial solution", f"Initial cost ({res['initial_method_name']})", "correct",
                    message=f"The initial basic feasible solution cost {fmt_q(v)} matches the {res['initial_method_name']}.")
        else:
            rep.add("Initial solution", f"Initial cost ({res['initial_method_name']})", "incorrect", "T01",
                    f"The {res['initial_method_name']} gives an initial cost of {fmt_q(res['initial_cost'])}.",
                    ai=v, correct=res["initial_cost"])
    # 3. allocations
    clusters = parse_allocations(answer, P)
    final_alloc = clusters[-1] if clusters else None
    if clusters and len(clusters) > 1:
        first = clusters[0]
        ours = {(i, j): a for i, r in enumerate(res["initial_alloc"]) for j, a in enumerate(r) if a}
        if {k: v for k, v in first.items() if v} == ours:
            rep.add("Initial solution", "Initial allocations", "correct",
                    message=f"The initial allocations match the {res['initial_method_name']}.")
        elif sum(first.values()) == sum(P["supply"]):
            diff = [k for k in set(first) | set(ours) if first.get(k, 0) != ours.get(k, 0)]
            rep.add("Initial solution", "Initial allocations", "incorrect", "T01",
                    f"The {res['initial_method_name']} allocates differently, e.g. at " +
                    ", ".join(_cell_name(P, i, j) + f": {fmt_q(ours.get((i, j), 0))}" for i, j in diff[:3]) + ".")
    claimed = [_num(m.group(1)) for m in _COST.finditer(answer)]
    cost_of = None
    claim_cost = claimed[-1] if claimed else None
    label = "unclear"
    if final_alloc:
        m_, n_ = len(P["supply"]), len(P["demand"])
        rows_ok = all(sum(final_alloc.get((i, j), 0) for j in range(n_)) == P["supply"][i] for i in range(m_))
        cols_ok = all(sum(final_alloc.get((i, j), 0) for i in range(m_)) == P["demand"][j] for j in range(n_))
        real_rows = [i for i, l in enumerate(P["row_labels"]) if l != "Dummy"]
        real_cols = [j for j, l in enumerate(P["col_labels"]) if l != "Dummy"]
        if not (rows_ok and cols_ok):
            # answers often omit dummy allocations; check the real rows / columns only
            rows_ok = all(sum(final_alloc.get((i, j), 0) for j in real_cols) <= P["supply"][i] for i in real_rows)
            cols_ok = all(sum(final_alloc.get((i, j), 0) for i in real_rows) <= P["demand"][j] for j in real_cols)
        cost_of = sum((P["costs"][i][j] * a for (i, j), a in final_alloc.items()), Fraction(0))
        if not (rows_ok and cols_ok):
            rep.add("Final allocation", "Supply and demand totals", "incorrect", "T02",
                    "The final allocations do not add up to the supplies and demands.")
            label = "infeasible_solution"
        else:
            rep.add("Final allocation", "Supply and demand totals", "correct",
                    message="Every source ships its supply and every destination receives its demand.")
            if claim_cost is not None and cost_of != claim_cost:
                rep.add("Final allocation", "Total cost arithmetic", "incorrect", "T02",
                        f"The allocations listed cost {fmt_q(cost_of)} in total.", ai=claim_cost, correct=cost_of)
            better = (cost_of < opt) if res["sense"] == "max" else (cost_of > opt)
            if better:
                rep.add("Final allocation", "Optimality", "incorrect", "T03",
                        f"This allocation is feasible but not optimal: it gives {fmt_q(cost_of)}, the optimum is {fmt_q(opt)}.",
                        ai=cost_of, correct=opt)
                label = "suboptimal_solution"
            else:
                rep.add("Final allocation", "Optimality", "correct", message="The final allocation is optimal.")
                label = "correct"
    if claim_cost is not None:
        if claim_cost == opt:
            rep.add("Final answer", "Optimal total", "correct", message=f"Optimal total {fmt_q(opt)} is correct.")
            label = "correct" if label in ("unclear", "correct") else label
        elif final_alloc and label == "suboptimal_solution" and claim_cost == cost_of:
            rep.add("Final answer", "Optimal total", "consequential", "T03",
                    f"The total {fmt_q(claim_cost)} is computed correctly for the answer's allocation, but that allocation is "
                    f"not optimal; the optimal total is {fmt_q(opt)}.", ai=claim_cost, correct=opt,
                    consequence="Final allocation: Optimality")
        else:
            rep.add("Final answer", "Optimal total", "incorrect", "T03" if not final_alloc else "T02",
                    f"The optimal total is {fmt_q(opt)}.", ai=claim_cost, correct=opt)
            label = "wrong_value" if label in ("unclear", "correct") else label
    elif label == "unclear":
        rep.add("Final answer", "Optimal total", "unverifiable",
                message="No total cost or final allocation could be read from the answer.")
    if res["status"] == "multiple_optima":
        rep.notes.append("This problem has alternative optimal allocations (an empty cell has opportunity cost 0); a different "
                         "allocation with the same total is also correct.")
    if re.search(r"(?i)\bmodi\b|u-?v\s+method|ui\s*\+\s*vj|opportunity\s+cost", answer):
        rep.add("Optimality test", "MODI working", "unverifiable",
                message="The answer uses the MODI method; individual ui / vj values are not checked, only the resulting allocation.")
    summary = summarize(rep, label)
    codes = []
    for s in rep.steps:
        if s["status"] == "incorrect" and s.get("code") and s["code"] not in codes:
            codes.append(s["code"])
    tb = {"loaded": bool(refs and refs.status()["loaded"]), "method_passages": [], "by_error": {}}
    if tb["loaded"]:
        tb["method_passages"] = refs.method_passages("transportation", question)
        tb["by_error"] = refs.for_errors("transportation", codes, question)
    LEG = {"verified": "valid", "partially_verified": "partial", "incorrect": "invalid", "incorrect_working": "invalid",
           "unverifiable": "unverifiable"}
    return {"method": "transportation", "method_name": "Transportation problem", "method_note":
            f"Initial solution method detected in the answer: {res['initial_method_name']}.", "ai_source": ai_source,
            "report": {"summary": summary, "steps": rep.steps, "tableaus": [], "notes": rep.notes, "parsed": {
                "allocation_groups": len(clusters), "cost_claims": len(claimed)}},
            "solution": to_json(res), "textbook": tb,
            "general_notes": {c: {"title": CATALOG[c]["title"], "note": CATALOG[c]["note"]} for c in codes},
            "cross_check": _scipy_transport(P, res), "ml": None,
            "verdict": LEG[summary["verdict"]], "label_text": summary["verdict_text"], "error_type": None if label == "unclear" else label,
            "truth": {"text": f"Optimal {'profit' if res['sense'] == 'max' else 'cost'} = {fmt_q(opt)}", "status": res["status"]},
            "claim": {"text": f"Total = {fmt_q(claim_cost)}" if claim_cost is not None else "(no total found)"},
            "reasons": [], "references": [p for ps in tb["by_error"].values() for p in ps][:3], "plot": None, "checks": []}


def _scipy_transport(P, res):
    try:
        import numpy as np
        from scipy.optimize import linprog
        m, n = len(P["supply"]), len(P["demand"])
        big = 1e9
        c = [float(P["costs"][i][j]) if not isinstance(P["costs"][i][j], MNum) else big for i in range(m) for j in range(n)]
        if res["sense"] == "max":
            c = [-x if x != big else big for x in c]
        A, b = [], []
        for i in range(m):
            A.append([1.0 if k // n == i else 0.0 for k in range(m * n)]); b.append(float(P["supply"][i]))
        for j in range(n):
            A.append([1.0 if k % n == j else 0.0 for k in range(m * n)]); b.append(float(P["demand"][j]))
        r = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * (m * n), method="highs")
        val = -r.fun if res["sense"] == "max" else r.fun
        ok = abs(val - float(res["optimal_cost"])) < 1e-6
        return {"available": True, "agrees": bool(ok), "scipy_value": float(val),
                "note": ("SciPy (HiGHS) reaches the same optimal total." if ok else "SciPy disagrees; please report this problem.") +
                        " Final-answer cross-check only."}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "agrees": None, "note": f"SciPy cross-check unavailable: {e}"}


def solve_text(question, initial="vam"):
    P0 = parse_transportation(question)
    return {"method": "transportation", "method_name": "Transportation problem", "solution": to_json(solve(P0, initial))}
