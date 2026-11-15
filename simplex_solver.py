"""Standard (primal) Simplex method with a full tableau history.

Applies to problems whose constraints are all '<=' with non-negative right-hand
sides (after the textbook sign normalisation), so that the slack variables form
the initial basis.  Both maximisation and minimisation are supported:

* maximise: enter the largest positive Cj - Zj, optimal when all Cj - Zj <= 0;
* minimise: enter the most negative Cj - Zj, optimal when all Cj - Zj >= 0.

Problems that need artificial variables raise ``UnsupportedForMethod`` with an
explanation; use ``bigm_solver`` or ``two_phase_solver`` for those.
"""
from __future__ import annotations

from fractions import Fraction

from .lp_common import finish, to_json  # noqa: F401  (to_json re-exported)
from .tableau import Tableau, UnsupportedForMethod, run, standard_form  # noqa: F401


def initial_tableau(prob):
    sf = standard_form(prob, "simplex")
    vs = sf["variables"]
    cost = [sf["c"][v] for v in vs] + [Fraction(0)] * (len(sf["columns"]) - len(vs))
    tab = Tableau(sf["columns"], sf["kinds"], cost, sf["A"], sf["b"], sf["basis"], sf["sense"], "Simplex")
    return sf, tab


def solve(prob):
    sf, tab = initial_tableau(prob)
    vs = sf["variables"]
    status, iters = run(tab)
    phases = [{"name": "Simplex", "status": status, "iterations": iters}]
    intro = ["Standard form: " + "; ".join(sf["equations"]) + ". The slack variables " +
             ", ".join(c for c, k in zip(sf["columns"], sf["kinds"]) if k == "slack") +
             " form the initial basis (all decision variables = 0)."]
    return finish("simplex", sf, phases, tab if status == "optimal" else None, status, vs, intro)
