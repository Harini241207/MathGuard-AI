"""Big M (symbolic M) and Two-Phase tests."""
import random
from fractions import Fraction as F

import pytest

from backend import bigm_solver as B, two_phase_solver as T, graphical_solver as G, simplex_solver as S
from backend.exact import MNum, parse_number
from backend.lp_parser import parse_problem

MIN_GE = "Minimize Z = 2x + 3y subject to x + y >= 4, x + 2y >= 6, x, y >= 0"


def test_mnum_ordering_and_arithmetic():
    M = MNum(1, 0)
    assert MNum(1, -1000) > MNum(0, 10**9)          # M dominates any constant
    assert MNum(-2, 5) < MNum(-1, -100)
    assert (3 * M + 2) - M == MNum(2, 2)
    assert str(MNum(-3, 2)) == "-3M + 2" and str(MNum(1, 0)) == "M"
    assert parse_number("3 - 2M") == MNum(-2, 3)
    assert parse_number("-M") == MNum(-1, 0)
    assert parse_number("M/2 + 1") == MNum(F(1, 2), 1)


@pytest.mark.parametrize("mod", [B, T])
def test_min_with_ge_constraints(mod):
    r = mod.solve(parse_problem(MIN_GE))
    assert r["status"] == "optimal" and r["value"] == 10 and r["solution"] == {"x": 2, "y": 2}


def test_bigm_initial_tableau_has_symbolic_m():
    r = B.solve(parse_problem(MIN_GE))
    t0 = r["phases"][0]["iterations"][0]
    assert t0["columns"] == ["x", "y", "s1", "s2", "A1", "A2"]
    assert t0["basis"] == ["A1", "A2"]
    assert t0["cb"] == [MNum(1, 0), MNum(1, 0)]
    # Cj - Zj for x = 2 - 2M, for y = 3 - 3M (minimisation: most negative enters -> y)
    assert t0["delta"][0] == MNum(-2, 2) and t0["delta"][1] == MNum(-3, 3)
    assert t0["entering"] == "y"


def test_two_phase_structure():
    r = T.solve(parse_problem(MIN_GE))
    names = [p["name"] for p in r["phases"]]
    assert names == ["Phase I", "Phase II"]
    p1_last = r["phases"][0]["iterations"][-1]
    assert p1_last["z"] == 0
    assert "A1" not in r["phases"][1]["iterations"][0]["columns"]


@pytest.mark.parametrize("mod", [B, T])
def test_infeasible(mod):
    r = mod.solve(parse_problem("Maximize Z = 3x + 2y subject to x + y <= 2, x + y >= 5, x, y >= 0"))
    assert r["status"] == "infeasible"


@pytest.mark.parametrize("mod", [B, T])
def test_unbounded(mod):
    r = mod.solve(parse_problem("Maximize Z = x + y subject to x - y >= 1, x, y >= 0"))
    assert r["status"] == "unbounded"


@pytest.mark.parametrize("mod", [B, T])
def test_equality_and_redundant_equality(mod):
    r = mod.solve(parse_problem("Maximize Z = x + 2y subject to x + y = 2, 2x + 2y = 4, y <= 1, x, y >= 0"))
    assert r["status"] == "optimal" and r["value"] == 3 and r["solution"] == {"x": 1, "y": 1}


def test_textbook_three_constraint_min():
    # Min Z = 3x + 2y, x + y >= 5, 2x + y >= 8, x + 3y >= 9  -> (3, 2), Z = 13
    t = "Minimize Z = 3x + 2y subject to x + y >= 5, 2x + y >= 8, x + 3y >= 9, x, y >= 0"
    for mod in (B, T):
        r = mod.solve(parse_problem(t))
        assert r["value"] == 13 and r["solution"] == {"x": 3, "y": 2}


@pytest.mark.parametrize("seed", range(150))
def test_random_against_scipy_and_each_other(seed):
    from scipy.optimize import linprog
    rng = random.Random(1000 + seed)
    n = rng.randint(2, 4)
    names = ["x1", "x2", "x3", "x4"][:n]
    sense = rng.choice(["max", "min"])
    cons = [([rng.randint(-2, 6) for _ in range(n)], rng.choice(["<=", ">=", "<=", "="]), rng.randint(0, 20))
            for _ in range(rng.randint(1, 4))]
    cons = [(a, op, b) for a, op, b in cons if any(a)] or [([1] * n, "<=", 10)]
    obj = [rng.randint(-3, 9) for _ in range(n)]

    def lin(a):
        return " + ".join(f"{k}{v}" for k, v in zip(a, names)).replace("+ -", "- ")
    txt = (f"{'Maximize' if sense == 'max' else 'Minimize'} Z = {lin(obj)} subject to " +
           ", ".join(f"{lin(a)} {op} {b}" for a, op, b in cons) + ", " + ", ".join(names) + " >= 0")
    prob = parse_problem(txt)
    rb, rt = B.solve(prob), T.solve(prob)
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for a, op, b in cons:
        if op == "<=":
            A_ub.append(a); b_ub.append(b)
        elif op == ">=":
            A_ub.append([-k for k in a]); b_ub.append(-b)
        else:
            A_eq.append(a); b_eq.append(b)
    cm = [-k for k in obj] if sense == "max" else obj
    s = linprog(cm, A_ub=A_ub or None, b_ub=b_ub or None, A_eq=A_eq or None, b_eq=b_eq or None,
                bounds=[(0, None)] * n, method="highs")
    want = {0: "opt", 2: "infeasible", 3: "unbounded"}[s.status]
    for r in (rb, rt):
        got = "opt" if r["status"] in ("optimal", "multiple_optima") else r["status"]
        assert got == want, (txt, r["method"], r["status"], s.status)
        if want == "opt":
            assert abs(float(r["value"]) - (-s.fun if sense == "max" else s.fun)) < 1e-7, txt
    if n == 2:
        g = G.solve(prob)
        assert (g["status"] in ("optimal", "multiple_optima")) == (want == "opt")
        if want == "opt":
            assert g["optimum"]["value"] == rt["value"]
        assert (g["status"] == "multiple_optima") == (rt["status"] == "multiple_optima"), txt
