"""Big M (penalty) method with a symbolic M.

Each artificial variable gets the objective coefficient -M (maximisation) or +M
(minimisation).  M is kept symbolic as ``MNum(a, b)`` = a*M + b and compared
"M first", which is exactly the textbook assumption that M is larger than any
other number in the problem.  This avoids the classic numerical pitfalls of
substituting a large constant such as 10**6 (rounding, or a constant that is not
actually large enough for the data).

Outcome rules:
* optimal tableau with an artificial variable basic at a positive level
  -> the original problem is infeasible;
* the penalised problem is unbounded -> the original problem is unbounded if
  it is feasible (checked with Phase I of the two-phase method; this check is
  recorded in the explanation).
"""
from __future__ import annotations

from fractions import Fraction

from .exact import MNum
from .lp_common import finish, to_json  # noqa: F401
from .tableau import Tableau, run, standard_form
from . import two_phase_solver


def solve(prob):
    sf = standard_form(prob, "bigm")
    vs = sf["variables"]
    cols, kinds = sf["columns"], sf["kinds"]
    sense = sf["sense"]
    pen = MNum(-1, 0) if sense == "max" else MNum(1, 0)
    cost = []
    for c, k in zip(cols, kinds):
        if k == "decision":
            cost.append(MNum.lift(sf["c"][c]))
        elif k == "artificial":
            cost.append(pen)
        else:
            cost.append(MNum(0, 0))
    intro = ["Standard form: " + "; ".join(sf["equations"]) + "."]
    if sf["artificials"]:
        intro.append(
            f"Each artificial variable gets the penalty {'-M' if sense == 'max' else '+M'} in the objective "
            f"({'Maximise' if sense == 'max' else 'Minimise'} Z = ... {'-' if sense == 'max' else '+'} M(" +
            " + ".join(cols[j] for j in sf["artificials"]) + ")), where M is a very large positive number. "
            "M is kept as a symbol, so Cj - Zj values are of the form aM + b and are compared on the M part first.")
    else:
        intro.append("No artificial variables are needed, so the Big M method reduces to the ordinary simplex method.")
    tab = Tableau(cols, kinds, cost, sf["A"], sf["b"], sf["basis"], sense, "Big M")
    status, iters = run(tab, "Big M")
    phases = [{"name": "Big M", "status": status, "iterations": iters}]
    if status == "optimal":
        pos_art = [cols[j] for i, j in enumerate(tab.basis) if kinds[j] == "artificial" and tab.b[i] > 0]
        if pos_art:
            phases[0]["note"] = (f"The optimality condition holds but artificial variable(s) {', '.join(pos_art)} remain in the "
                                 "basis at a positive level, so the original problem has no feasible solution.")
            intro.append(phases[0]["note"])
            return finish("bigm", sf, phases, None, "infeasible", vs, intro)
        zero_art = [cols[j] for i, j in enumerate(tab.basis) if kinds[j] == "artificial"]
        if zero_art:
            phases[0]["note"] = (f"Artificial variable(s) {', '.join(zero_art)} remain basic at zero level: the solution is "
                                 "feasible (and degenerate).")
        tab.excluded |= set(sf["artificials"])
        return finish("bigm", sf, phases, tab, "optimal", vs, intro)
    # unbounded penalised problem
    art_in_basis = [cols[j] for i, j in enumerate(tab.basis) if kinds[j] == "artificial" and tab.b[i] > 0]
    if art_in_basis and not two_phase_solver.feasible(prob):
        phases[0]["note"] = ("The penalised problem looks unbounded while artificial variables are still positive; "
                             "a Phase I feasibility check shows the original constraints have no feasible solution.")
        intro.append(phases[0]["note"])
        return finish("bigm", sf, phases, None, "infeasible", vs, intro)
    return finish("bigm", sf, phases, None, "unbounded", vs, intro)
