"""Two-Phase simplex method.

Phase I  : Maximise W = -(sum of artificial variables) over the standard-form
           constraints.  If the optimum has W < 0 the original problem is
           infeasible.  Artificial variables that are still basic at value 0 are
           driven out (pivot on any non-zero non-artificial entry) or, if the row
           has none, the row is redundant and is removed.
Phase II : Remove the artificial columns, restore the original objective and
           continue the simplex iterations from the Phase I basis.

If the problem needs no artificial variables, Phase I is skipped and the
result is the ordinary simplex method.
"""
from __future__ import annotations

from fractions import Fraction

from .exact import fmt_q
from .lp_common import finish, to_json  # noqa: F401
from .tableau import Tableau, run, standard_form


def solve(prob):
    sf = standard_form(prob, "two_phase")
    vs = sf["variables"]
    cols, kinds = sf["columns"], sf["kinds"]
    arts = sf["artificials"]
    phases = []
    intro = ["Standard form: " + "; ".join(sf["equations"]) + "."]

    if not arts:
        intro.append("No artificial variables are needed, so Phase I is not required; the ordinary simplex method is applied.")
        cost = [sf["c"][v] for v in vs] + [Fraction(0)] * (len(cols) - len(vs))
        tab = Tableau(cols, kinds, cost, sf["A"], sf["b"], sf["basis"], sf["sense"], "Phase II")
        status, iters = run(tab, "Phase II")
        phases.append({"name": "Phase II", "status": status, "iterations": iters})
        return finish("two_phase", sf, phases, tab if status == "optimal" else None, status, vs, intro)

    # ---------------- Phase I
    cost1 = [Fraction(-1) if k == "artificial" else Fraction(0) for k in kinds]
    t1 = Tableau(cols, kinds, cost1, sf["A"], sf["b"], sf["basis"], "max", "Phase I")
    intro.append("Phase I: Maximise W = -(" + " + ".join(cols[j] for j in arts) +
                 ") subject to the standard-form constraints; the artificial variables form the initial basis for their rows.")
    st1, it1 = run(t1, "Phase I")
    w = t1.z()
    note = f"Phase I ends with W = {fmt_q(w)}."
    phases.append({"name": "Phase I", "status": st1, "iterations": it1, "note": note})
    if w != 0:
        phases[-1]["note"] += (" Because the maximum of W is negative, at least one artificial variable is still positive: "
                               "the original problem has no feasible solution.")
        intro.append(phases[-1]["note"])
        return finish("two_phase", sf, phases, None, "infeasible", vs, intro)

    # drive out artificial variables still basic at zero level
    extra_ops = []
    drop_rows = []
    for i in range(len(t1.basis)):
        if kinds[t1.basis[i]] != "artificial":
            continue
        piv = next((j for j in range(len(cols)) if kinds[j] != "artificial" and j not in t1.basis and t1.A[i][j] != 0), None)
        if piv is None:
            drop_rows.append(i)
            extra_ops.append(f"Row {i + 1} ({cols[t1.basis[i]]} = 0) has no non-zero entry outside the artificial columns: "
                             "the constraint is redundant and the row is removed.")
        else:
            old = cols[t1.basis[i]]
            t1.pivot(i, piv)
            extra_ops.append(f"Artificial variable {old} is basic at zero level; pivot on row {i + 1}, column {cols[piv]} to remove it.")
    keep_rows = [i for i in range(len(t1.basis)) if i not in drop_rows]
    keep_cols = [j for j in range(len(cols)) if kinds[j] != "artificial"]
    remap = {j: k for k, j in enumerate(keep_cols)}
    A2 = [[t1.A[i][j] for j in keep_cols] for i in keep_rows]
    b2 = [t1.b[i] for i in keep_rows]
    basis2 = [remap[t1.basis[i]] for i in keep_rows]
    cols2 = [cols[j] for j in keep_cols]
    kinds2 = [kinds[j] for j in keep_cols]
    cost2 = [sf["c"][c] if k == "decision" else Fraction(0) for c, k in zip(cols2, kinds2)]
    phases[-1]["note"] += (" W = 0, so a feasible basic solution has been found. The artificial columns are removed and the "
                           "original objective is restored for Phase II.")
    if extra_ops:
        phases[-1]["note"] += " " + " ".join(extra_ops)
    intro.append(phases[-1]["note"])

    # ---------------- Phase II
    t2 = Tableau(cols2, kinds2, cost2, A2, b2, basis2, sf["sense"], "Phase II")
    st2, it2 = run(t2, "Phase II")
    phases.append({"name": "Phase II", "status": st2, "iterations": it2})
    return finish("two_phase", sf, phases, t2 if st2 == "optimal" else None, st2, vs, intro)


def feasible(prob) -> bool:
    """Phase I only: does the problem have a feasible solution?"""
    sf = standard_form(prob, "two_phase")
    if not sf["artificials"]:
        return True
    kinds = sf["kinds"]
    cost1 = [Fraction(-1) if k == "artificial" else Fraction(0) for k in kinds]
    t1 = Tableau(sf["columns"], kinds, cost1, sf["A"], sf["b"], sf["basis"], "max", "Phase I")
    run(t1, "Phase I")
    return t1.z() == 0
