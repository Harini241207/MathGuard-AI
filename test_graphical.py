"""Graphical-method solver tests.  Expected answers were worked out by hand; SciPy is only a secondary cross-check."""
import random
from fractions import Fraction as F

import pytest

from backend import graphical_solver as g
from backend.lp_parser import parse_problem


def solve(text):
    return g.solve(parse_problem(text))


def corners(res):
    return {cn["point"] for cn in res["corners"]}


def test_bounded_maximisation_full_working():
    r = solve("Maximize Z = 3x + 2y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0")
    assert r["status"] == "optimal"
    assert r["optimum"]["point"] == (2, 2) and r["optimum"]["value"] == 10
    assert corners(r) == {(0, 0), (2, 0), (2, 2), (1, 3), (0, 3)}
    # boundary equations and intercepts
    l1 = r["boundary_lines"][0]
    assert l1["equation"] == "x + y = 4"
    assert l1["x_intercept"] == (4, 0) and l1["y_intercept"] == (0, 4)
    # infeasible intersections are reported with the constraint they break
    it = next(i for i in r["intersections"] if i["point"] == (4, 0))
    assert not it["feasible"] and "x <= 2" in it["violated"]
    # objective at every corner
    zs = {cn["point"]: cn["z"] for cn in r["corners"]}
    assert zs == {(0, 0): 0, (2, 0): 6, (2, 2): 10, (1, 3): 9, (0, 3): 6}
    # textbook ordering around the polygon starting at the origin
    assert r["corners"][0]["point"] == (0, 0)


def test_bounded_minimisation():
    r = solve("Minimize Z = 2x + 3y subject to x + y >= 4, x + 2y >= 6, x, y >= 0")
    assert r["status"] == "optimal"
    assert r["optimum"]["point"] == (2, 2) and r["optimum"]["value"] == 10
    assert corners(r) == {(0, 4), (2, 2), (6, 0)}
    assert r["region"] == "unbounded"  # region is unbounded but the minimum is finite


def test_infeasible():
    r = solve("Maximize Z = 3x + 2y subject to x + y <= 2, x + y >= 5, x, y >= 0")
    assert r["status"] == "infeasible" and r["region"] == "empty" and r["corners"] == []


def test_unbounded():
    r = solve("Maximize Z = x + y subject to x - y >= 1, x, y >= 0")
    assert r["status"] == "unbounded"
    assert r["optimum"] is None


def test_unbounded_region_but_finite_max():
    r = solve("Maximize Z = -x - y subject to x + y >= 2, x, y >= 0")
    assert r["status"] == "multiple_optima"  # every point of the segment x + y = 2 gives Z = -2
    assert r["optimum"]["value"] == -2


def test_multiple_optima_segment():
    r = solve("Maximize Z = 2x + 4y subject to x + 2y <= 8, x <= 6, y <= 5, x, y >= 0")
    assert r["status"] == "multiple_optima"
    assert set(r["optimal_corners"]) == {(0, 4), (6, 1)}
    assert r["optimum"]["value"] == 16


def test_multiple_optima_along_ray():
    # min x subject to x >= 1, y free upward: optimal along the ray x = 1, y >= 0
    r = solve("Minimize Z = x subject to x >= 1, x, y >= 0")
    assert r["status"] == "multiple_optima"
    assert r["optimal_ray"] == (0, 1)


def test_degenerate_corner():
    # three lines through (1, 1): x + y = 2, x = 1, y = 1
    r = solve("Maximize Z = 2x + y subject to x + y <= 2, x <= 1, y <= 1, x, y >= 0")
    assert r["status"] == "optimal" and r["optimum"]["point"] == (1, 1) and r["optimum"]["value"] == 3
    assert r["degenerate"] is True
    c = next(cn for cn in r["corners"] if cn["point"] == (1, 1))
    assert c["degenerate"] and len(c["on_lines"]) == 3


def test_equality_and_fractions():
    r = solve("Maximize Z = x + 2y subject to x + y = 3, 3x + y <= 7, 2y <= 5, x, y >= 0")
    assert r["status"] == "optimal"
    assert r["optimum"]["point"] == (F(1, 2), F(5, 2))
    assert r["optimum"]["value"] == F(11, 2)


def test_parallel_lines_reported():
    r = solve("Maximize Z = x + y subject to x + y <= 4, 2x + 2y <= 10, x, y >= 0")
    par = [i for i in r["intersections"] if i["parallel"]]
    assert par and "parallel" in par[0]["calc"]


def test_rejects_three_variables():
    with pytest.raises(g.GraphicalError):
        solve("Maximize Z = x + y + z subject to x + y + z <= 3, x, y, z >= 0")


def test_json_is_serialisable():
    import json
    r = solve("Maximize Z = 3x + 2y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0")
    s = json.dumps(g.to_json(r))
    assert '"s": "10"' in s


@pytest.mark.parametrize("seed", range(150))
def test_random_against_scipy(seed):
    """Secondary validation only: random 2-variable LPs must agree with SciPy/HiGHS on status and optimum value."""
    from scipy.optimize import linprog
    rng = random.Random(seed)
    sense = rng.choice(["max", "min"])
    cons = []
    for _ in range(rng.randint(1, 4)):
        a = [rng.randint(-3, 6), rng.randint(-3, 6)]
        if a == [0, 0]:
            a = [1, 1]
        cons.append((a, rng.choice(["<=", "<=", ">="]), rng.randint(0, 20)))
    obj = [rng.randint(-4, 9), rng.randint(-4, 9)]
    txt = (f"{'Maximize' if sense == 'max' else 'Minimize'} Z = {obj[0]}x + {obj[1]}y subject to "
           + ", ".join(f"{a[0]}x + {a[1]}y {op} {b}" for a, op, b in cons) + ", x, y >= 0").replace("+ -", "- ")
    r = solve(txt)
    A_ub, b_ub = [], []
    for a, op, b in cons:
        if op == "<=":
            A_ub.append(a); b_ub.append(b)
        else:
            A_ub.append([-a[0], -a[1]]); b_ub.append(-b)
    cm = [-obj[0], -obj[1]] if sense == "max" else obj
    s = linprog(cm, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 2, method="highs")
    if s.status == 2:
        assert r["status"] == "infeasible", txt
    elif s.status == 3:
        assert r["status"] == "unbounded", txt
    else:
        assert r["status"] in ("optimal", "multiple_optima"), txt
        z = -s.fun if sense == "max" else s.fun
        assert abs(float(r["optimum"]["value"]) - z) < 1e-6, txt
