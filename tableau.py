"""Shared simplex-tableau machinery (standard form, Cj / Zj / Cj - Zj, ratio test, pivoting).

The Simplex, Big M and Two-Phase solvers are thin wrappers around this module.
Everything is exact: body entries and RHS are Fractions; costs (Cj) are
Fractions or ``MNum`` (symbolic Big-M values).

Conventions (as in most Operations Research textbooks):

* Columns are ordered: decision variables, then slack/surplus variables
  (s1, s2, ... numbered by constraint), then artificial variables (A1, A2, ...).
* Net evaluation row:  Cj - Zj.  For a maximisation problem the tableau is
  optimal when every Cj - Zj <= 0 and the entering variable is the one with
  the largest positive Cj - Zj.  For minimisation the signs are reversed.
* Ties for the entering variable are broken by the lowest column index; ties
  in the minimum-ratio test by the lowest row index.  Both ties are recorded
  (a ratio tie means the next basic solution is degenerate).
* If more than ``BLAND_AFTER`` iterations are needed, Bland's rule is used to
  rule out cycling.  This switch is recorded in the iteration notes.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

from .exact import MNum, fmt_q, json_num, sign
from .lp_parser import exact_view, fmt_linear

BLAND_AFTER = 50
MAX_ITERS = 200


class UnsupportedForMethod(ValueError):
    """The selected method cannot be applied to this problem as stated (e.g. Simplex with >= constraints)."""


# ----------------------------------------------------------------------------- standard form
def standard_form(prob, mode):
    """Convert a parsed LP to standard (equality) form.

    mode: "simplex"   -> only slack variables allowed (all <= with b >= 0)
          "bigm"      -> artificial variables penalised with M
          "two_phase" -> artificial variables handled by Phase I

    Returns dict with columns, A, b, costs, initial basis and the conversion steps (text)."""
    ev = exact_view(prob)
    vs, sense, rows = ev["variables"], ev["sense"], ev["rows"]
    steps = []
    norm_rows = []
    for i, r in enumerate(rows):
        coefs, op, rhs = dict(r["coefs"]), r["op"], r["rhs"]
        if rhs < 0:
            coefs = {v: -k for v, k in coefs.items()}
            rhs = -rhs
            op = {"<=": ">=", ">=": "<=", "=": "="}[op]
            steps.append(f"Constraint {i + 1} has a negative right-hand side, so it is multiplied by -1: "
                         f"{fmt_linear(coefs, vs)} {op} {fmt_q(rhs)}.")
        norm_rows.append({"coefs": coefs, "op": op, "rhs": rhs, "orig": r["text"]})

    need_art = [i for i, r in enumerate(norm_rows) if r["op"] in (">=", "=")]
    if mode == "simplex" and need_art:
        bad = ", ".join(f"constraint {i + 1} ({norm_rows[i]['orig']})" for i in need_art)
        raise UnsupportedForMethod(
            "The standard Simplex method needs every constraint in the form '<=' with a non-negative right-hand side, "
            f"so that slack variables give a starting basis. {bad} would need an artificial variable. "
            "Use the Big M or Two-Phase method for this problem.")

    cols, kinds, owner = list(vs), ["decision"] * len(vs), [None] * len(vs)
    m = len(norm_rows)
    A = [[r["coefs"][v] for v in vs] for r in norm_rows]
    b = [r["rhs"] for r in norm_rows]
    basis = [None] * m
    eq_text = []
    for i, r in enumerate(norm_rows):
        if r["op"] in ("<=", ">="):
            name = f"s{i + 1}"
            cols.append(name)
            kinds.append("slack" if r["op"] == "<=" else "surplus")
            owner.append(i)
            for k in range(m):
                A[k].append(Fraction(1 if r["op"] == "<=" else -1) if k == i else Fraction(0))
            if r["op"] == "<=":
                basis[i] = len(cols) - 1
    for i, r in enumerate(norm_rows):
        if r["op"] in (">=", "="):
            name = f"A{i + 1}"
            cols.append(name)
            kinds.append("artificial")
            owner.append(i)
            for k in range(m):
                A[k].append(Fraction(1) if k == i else Fraction(0))
            basis[i] = len(cols) - 1
    for i, r in enumerate(norm_rows):
        terms = fmt_linear(r["coefs"], vs)
        extra = []
        for j, nm in enumerate(cols):
            if kinds[j] != "decision" and owner[j] == i:
                extra.append((" + " if A[i][j] > 0 else " - ") + nm)
        eq_text.append(terms + "".join(extra) + f" = {fmt_q(r['rhs'])}")
        if r["op"] == "<=":
            steps.append(f"Constraint {i + 1} ({r['orig']}) is '<=': add slack variable s{i + 1} >= 0.")
        elif r["op"] == ">=":
            steps.append(f"Constraint {i + 1} ({r['orig']}) is '>=': subtract surplus variable s{i + 1} >= 0 and add "
                         f"artificial variable A{i + 1} >= 0.")
        else:
            steps.append(f"Constraint {i + 1} ({r['orig']}) is an equation: add artificial variable A{i + 1} >= 0.")
    return {
        "variables": vs, "sense": sense, "c": ev["c"], "columns": cols, "kinds": kinds, "owner": owner,
        "A": A, "b": b, "basis": basis, "steps": steps, "equations": eq_text,
        "artificials": [j for j, k in enumerate(kinds) if k == "artificial"],
    }


# ----------------------------------------------------------------------------- tableau
class Tableau:
    def __init__(self, cols, kinds, cost, A, b, basis, sense, label=""):
        self.cols = list(cols)
        self.kinds = list(kinds)
        self.cost = list(cost)
        self.A = [list(r) for r in A]
        self.b = list(b)
        self.basis = list(basis)
        self.sense = sense
        self.label = label
        self.excluded = set()  # columns not allowed to enter (dropped artificials)

    # ---- net evaluations
    def cb(self):
        return [self.cost[j] for j in self.basis]

    def zj(self):
        cb = self.cb()
        return [sum((cb[i] * self.A[i][j] for i in range(len(self.A))), Fraction(0)) for j in range(len(self.cols))]

    def delta(self):
        z = self.zj()
        return [self.cost[j] - z[j] for j in range(len(self.cols))]

    def z(self):
        cb = self.cb()
        return sum((cb[i] * self.b[i] for i in range(len(self.b))), Fraction(0))

    def improving(self, d):
        return sign(d) > 0 if self.sense == "max" else sign(d) < 0

    def candidates(self):
        d = self.delta()
        return [j for j in range(len(self.cols)) if j not in self.basis and j not in self.excluded and self.improving(d[j])]

    def choose_entering(self, bland=False):
        cands = self.candidates()
        if not cands:
            return None, []
        if bland:
            return cands[0], []
        d = self.delta()
        best = max(cands, key=lambda j: (d[j] if self.sense == "max" else -d[j], -j))
        bv = d[best]
        ties = [j for j in cands if d[j] == bv and j != best]
        return best, ties

    def ratios(self, e):
        out = []
        for i in range(len(self.A)):
            a = self.A[i][e]
            out.append(self.b[i] / a if a > 0 else None)
        return out

    def choose_leaving(self, e, bland=False):
        rs = self.ratios(e)
        valid = [(r, i) for i, r in enumerate(rs) if r is not None]
        if not valid:
            return None, rs, []
        mn = min(r for r, _ in valid)
        tied = [i for r, i in valid if r == mn]
        if bland:
            leave = min(tied, key=lambda i: self.basis[i])
        else:
            leave = tied[0]
        return leave, rs, [i for i in tied if i != leave]

    def pivot(self, r, e):
        """Pivot on (row r, column e).  Returns the row operations as data + text."""
        p = self.A[r][e]
        ops = []
        self.A[r] = [x / p for x in self.A[r]]
        self.b[r] = self.b[r] / p
        ops.append({"row": r, "kind": "normalise", "factor": p,
                    "text": f"R{r + 1}(new) = R{r + 1} / {fmt_q(p)}" if p != 1 else f"R{r + 1}(new) = R{r + 1} (pivot is already 1)"})
        for i in range(len(self.A)):
            if i == r:
                continue
            f = self.A[i][e]
            if f == 0:
                ops.append({"row": i, "kind": "unchanged", "factor": Fraction(0),
                            "text": f"R{i + 1}(new) = R{i + 1} (entry in the pivot column is already 0)"})
                continue
            self.A[i] = [x - f * y for x, y in zip(self.A[i], self.A[r])]
            self.b[i] = self.b[i] - f * self.b[r]
            if f > 0:
                txt = f"R{i + 1}(new) = R{i + 1} - {fmt_q(f) + ' x ' if f != 1 else ''}R{r + 1}(new)"
            else:
                txt = f"R{i + 1}(new) = R{i + 1} + {fmt_q(-f) + ' x ' if f != -1 else ''}R{r + 1}(new)"
            ops.append({"row": i, "kind": "eliminate", "factor": f, "text": txt})
        self.basis[r] = e
        return ops

    def snapshot(self):
        d = self.delta()
        return {
            "label": self.label, "columns": list(self.cols), "kinds": list(self.kinds), "cost": list(self.cost),
            "basis": [self.cols[j] for j in self.basis], "basis_idx": list(self.basis), "cb": self.cb(),
            "A": deepcopy(self.A), "b": list(self.b), "zj": self.zj(), "delta": d, "z": self.z(),
            "sense": self.sense, "excluded": sorted(self.excluded),
        }

    def solution(self, names):
        vals = {n: Fraction(0) for n in names}
        for i, j in enumerate(self.basis):
            if self.cols[j] in vals:
                vals[self.cols[j]] = self.b[i]
        return vals


def run(tab: Tableau, phase_label="", max_iters=MAX_ITERS):
    """Iterate until optimal / unbounded.  Returns (status, iterations).

    Each iteration dict = tableau snapshot + the decision taken from it:
      entering, entering_ties, ratios, leaving, leaving_ties, pivot, row_ops, notes, conclusion."""
    iters = []
    k = 0
    bland = False
    while True:
        snap = tab.snapshot()
        snap["phase"] = phase_label
        snap["iteration"] = k
        snap["notes"] = []
        if any(bv == 0 for bv in tab.b):
            snap["degenerate"] = True
            snap["notes"].append("Degenerate basic solution: a basic variable has value 0.")
        else:
            snap["degenerate"] = False
        if k >= BLAND_AFTER and not bland:
            bland = True
            snap["notes"].append(f"More than {BLAND_AFTER} iterations: switching to Bland's rule to prevent cycling.")
        e, eties = tab.choose_entering(bland)
        if e is None:
            snap["conclusion"] = "optimal"
            snap["notes"].append(_optimality_note(tab))
            iters.append(snap)
            return "optimal", iters
        leave, rs, lties = tab.choose_leaving(e, bland)
        snap.update(entering=tab.cols[e], entering_idx=e, entering_ties=[tab.cols[j] for j in eties], ratios=rs)
        if leave is None:
            snap["conclusion"] = "unbounded"
            snap["notes"].append(
                f"{tab.cols[e]} can enter (Cj - Zj = {fmt_q(snap['delta'][e])}), but no entry in its column is positive, so no "
                "variable limits its increase: the objective is unbounded.")
            iters.append(snap)
            return "unbounded", iters
        piv = tab.A[leave][e]
        snap.update(leaving=tab.cols[tab.basis[leave]], leaving_row=leave, leaving_ties=[tab.cols[tab.basis[i]] for i in lties],
                    pivot={"row": leave, "col": e, "value": piv})
        if eties:
            snap["notes"].append("Tie for the entering variable between " + ", ".join([tab.cols[e]] + snap["entering_ties"]) +
                                 f"; {tab.cols[e]} chosen (lowest index).")
        if lties:
            snap["notes"].append("Tie in the minimum-ratio test (" + ", ".join([snap["leaving"]] + snap["leaving_ties"]) +
                                 "): the next basic solution will be degenerate.")
        snap["row_ops"] = tab.pivot(leave, e)
        snap["conclusion"] = "pivot"
        iters.append(snap)
        k += 1
        if k > max_iters:
            raise RuntimeError("Simplex did not terminate within the iteration limit.")


def _optimality_note(tab):
    if tab.sense == "max":
        return "All Cj - Zj <= 0, so the current solution is optimal (maximisation rule)."
    return "All Cj - Zj >= 0, so the current solution is optimal (minimisation rule)."


# ----------------------------------------------------------------------------- after optimality
def alternative_optima(tab: Tableau, decision):
    """Explore optimal bases reachable through zero Cj - Zj columns.

    Returns (points, rays, alt_snapshot): distinct optimal decision-variable points,
    optimal rays (directions in decision space), and one alternative optimal tableau
    for display (or None)."""
    def point(t):
        return tuple(t.solution(decision)[v] for v in decision)

    seen_bases = {tuple(sorted(tab.basis))}
    pts = [point(tab)]
    rays = []
    alt_snapshot = None
    queue = [deepcopy(tab)]
    while queue and len(seen_bases) < 64:
        t = queue.pop(0)
        d = t.delta()
        for j in range(len(t.cols)):
            if j in t.basis or j in t.excluded or t.kinds[j] == "artificial" or sign(d[j]) != 0:
                continue
            leave, _, _ = t.choose_leaving(j)
            if leave is None:
                # moving along column j forever keeps Z the same: an optimal ray (if it changes decision values)
                dirn = {v: Fraction(0) for v in decision}
                if t.cols[j] in dirn:
                    dirn[t.cols[j]] = Fraction(1)
                for i, bj in enumerate(t.basis):
                    if t.cols[bj] in dirn:
                        dirn[t.cols[bj]] -= t.A[i][j]
                vec = tuple(dirn[v] for v in decision)
                if any(x != 0 for x in vec) and vec not in rays:
                    rays.append(vec)
                continue
            t2 = deepcopy(t)
            ops = t2.pivot(leave, j)
            key = tuple(sorted(t2.basis))
            if key in seen_bases:
                continue
            seen_bases.add(key)
            p = point(t2)
            if p not in pts:
                pts.append(p)
                if alt_snapshot is None:
                    alt_snapshot = t2.snapshot()
                    alt_snapshot["via"] = {"entering": t.cols[j], "leaving": t.cols[t.basis[leave]], "row_ops": ops}
            queue.append(t2)
    return pts, rays, alt_snapshot


# ----------------------------------------------------------------------------- serialisation
def snap_json(s):
    out = {
        "label": s.get("label", ""), "phase": s.get("phase", ""), "iteration": s.get("iteration"),
        "columns": s["columns"], "kinds": s["kinds"], "cost": [json_num(x) for x in s["cost"]],
        "basis": s["basis"], "cb": [json_num(x) for x in s["cb"]],
        "rows": [[json_num(x) for x in row] for row in s["A"]], "rhs": [json_num(x) for x in s["b"]],
        "zj": [json_num(x) for x in s["zj"]], "delta": [json_num(x) for x in s["delta"]], "z": json_num(s["z"]),
        "excluded": [s["columns"][j] for j in s.get("excluded", [])],
        "conclusion": s.get("conclusion"), "notes": s.get("notes", []), "degenerate": s.get("degenerate", False),
        "entering": s.get("entering"), "entering_ties": s.get("entering_ties", []),
        "ratios": [json_num(r) for r in s["ratios"]] if s.get("ratios") is not None else None,
        "leaving": s.get("leaving"), "leaving_ties": s.get("leaving_ties", []),
        "pivot": None if not s.get("pivot") else {"row": s["pivot"]["row"], "col": s["pivot"]["col"],
                                                  "value": json_num(s["pivot"]["value"])},
        "row_ops": [o["text"] for o in s.get("row_ops", [])],
    }
    if s.get("via"):
        out["via"] = {"entering": s["via"]["entering"], "leaving": s["via"]["leaving"],
                      "row_ops": [o["text"] for o in s["via"]["row_ops"]]}
    return out


def tableau_text(s):
    """Plain-text rendering of a snapshot (used in explanations and tests)."""
    cols = s["columns"]
    head = ["Basis", "CB"] + cols + ["RHS"]
    lines = [" | ".join(head)]
    for i, bn in enumerate(s["basis"]):
        lines.append(" | ".join([bn, fmt_q(s["cb"][i])] + [fmt_q(x) for x in s["A"][i]] + [fmt_q(s["b"][i])]))
    lines.append(" | ".join(["Zj", ""] + [fmt_q(x) for x in s["zj"]] + [fmt_q(s["z"])]))
    lines.append(" | ".join(["Cj-Zj", ""] + [fmt_q(x) for x in s["delta"]] + [""]))
    return "\n".join(lines)
