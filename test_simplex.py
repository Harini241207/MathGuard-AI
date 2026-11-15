"""Simplex-method tests: every tableau of hand-checked examples, plus edge cases."""
import random
from fractions import Fraction as F

import pytest

from backend import simplex_solver as S
from backend.lp_parser import parse_problem
from backend.tableau import UnsupportedForMethod


def solve(t):
    return S.solve(parse_problem(t))


def iters(res):
    return res["phases"][0]["iterations"]


def test_standard_max_every_tableau():
    r = solve("Maximize Z = 3x + 5y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0")
    it = iters(r)
    assert len(it) == 3
    t0 = it[0]
    assert t0["columns"] == ["x", "y", "s1", "s2", "s3"]
    assert t0["basis"] == ["s1", "s2", "s3"]
    assert t0["delta"] == [3, 5, 0, 0, 0]
    assert t0["entering"] == "y" and t0["leaving"] == "s3"
    assert t0["ratios"] == [4, None, 3]
    assert t0["pivot"]["value"] == 1
    assert [o["text"] for o in t0["row_ops"]][1] == "R1(new) = R1 - R3(new)"
    t1 = it[1]
    assert t1["basis"] == ["s1", "s2", "y"]
    assert t1["A"][0] == [1, 0, 1, 0, -1] and t1["b"] == [1, 2, 3]
    assert t1["zj"] == [0, 5, 0, 0, 5] and t1["z"] == 15
    assert t1["entering"] == "x" and t1["leaving"] == "s1"
    t2 = it[2]
    assert t2["conclusion"] == "optimal"
    assert t2["delta"] == [0, 0, -3, 0, -2]
    assert r["status"] == "optimal"
    assert r["solution"] == {"x": 1, "y": 3} and r["value"] == 18


def test_three_variables_multiple_iterations():
    r = solve("Maximize Z = 3x + 2y + 4z subject to x + y + z <= 6, 2x + y <= 8, y + 2z <= 8, x, y, z >= 0")
    assert r["status"] == "optimal" and r["value"] == 22
    assert r["solution"] == {"x": 2, "y": 0, "z": 4}
    assert len(iters(r)) >= 3


def test_degenerate():
    r = solve("Maximize Z = 2x + y subject to x + y <= 2, x <= 1, y <= 1, x, y >= 0")
    assert r["value"] == 3 and r["solution"] == {"x": 1, "y": 1}
    assert r["degenerate"] is True
    assert any("degenerate" in n.lower() for s in iters(r) for n in s["notes"])


def test_alternative_optima():
    r = solve("Maximize Z = 2x + 4y subject to x + 2y <= 8, x + y <= 6, x, y >= 0")
    assert r["status"] == "multiple_optima" and r["value"] == 16
    assert set(r["optimal_points"]) == {(0, 4), (4, 2)}
    assert r["alternative"] is not None


def test_unbounded():
    r = solve("Maximize Z = 3x + 2y subject to x - y <= 1, x, y >= 0")
    assert r["status"] == "unbounded"
    last = iters(r)[-1]
    assert last["conclusion"] == "unbounded" and last["entering"] == "y"


def test_minimisation_with_le_constraints():
    r = solve("Minimize Z = x - 3y subject to x + y <= 4, y <= 3, x, y >= 0")
    assert r["status"] == "optimal" and r["value"] == -9
    assert r["solution"] == {"x": 0, "y": 3}
    assert iters(r)[0]["entering"] == "y"  # most negative Cj - Zj


def test_negative_rhs_is_normalised():
    # -x + y >= -2  ->  x - y <= 2 : still a standard-simplex problem
    r = solve("Maximize Z = x + y subject to -x + y >= -2, y <= 3, x, y >= 0")
    assert r["status"] == "optimal" and r["value"] == 8
    assert any("multiplied by -1" in s for s in r["standard_form"]["steps"])


def test_ge_constraint_is_rejected_with_advice():
    with pytest.raises(UnsupportedForMethod) as e:
        solve("Minimize Z = 2x + 3y subject to x + y >= 4, x + 2y >= 6, x, y >= 0")
    assert "Big M" in str(e.value) and "Two-Phase" in str(e.value)


def test_beale_cycling_example_terminates():
    # Beale (1955): cycles under the largest-coefficient rule with naive tie-breaking; Bland's rule fallback guarantees termination.
    t = ("Maximize Z = 3/4x1 - 20x2 + 1/2x3 - 6x4 subject to "
         "1/4x1 - 8x2 - x3 + 9x4 <= 0, 1/2x1 - 12x2 - 1/2x3 + 3x4 <= 0, x3 <= 1, x1, x2, x3, x4 >= 0")
    r = solve(t)
    assert r["status"] == "optimal" and r["value"] == F(5, 4)


def test_explanation_mentions_each_decision():
    r = solve("Maximize Z = 3x + 5y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0")
    txt = " ".join(r["explanation"])
    assert "y enters" in txt and "s3 leaves" in txt and "Pivot element = 1" in txt and "maximum Z = 18" in txt


def test_json():
    import json
    r = solve("Maximize Z = 3x + 5y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0")
    j = S.to_json(r)
    json.dumps(j)
    assert j["phases"][0]["iterations"][0]["entering"] == "y"


@pytest.mark.parametrize("seed", range(120))
def test_random_against_scipy(seed):
    from scipy.optimize import linprog
    rng = random.Random(seed)
    n = rng.randint(2, 4)
    names = ["x1", "x2", "x3", "x4"][:n]
    sense = rng.choice(["max", "min"])
    cons = [([rng.randint(-2, 6) for _ in range(n)], rng.randint(0, 25)) for _ in range(rng.randint(1, 4))]
    obj = [rng.randint(-3, 9) for _ in range(n)]

    def lin(a):
        return " + ".join(f"{k}{v}" for k, v in zip(a, names)).replace("+ -", "- ")
    txt = (f"{'Maximize' if sense == 'max' else 'Minimize'} Z = {lin(obj)} subject to " +
           ", ".join(f"{lin(a)} <= {b}" for a, b in cons) + ", " + ", ".join(names) + " >= 0")
    prob = parse_problem(txt)
    r = S.solve(prob)
    cm = [-k for k in obj] if sense == "max" else obj
    s = linprog(cm, A_ub=[a for a, _ in cons], b_ub=[b for _, b in cons], bounds=[(0, None)] * n, method="highs")
    if s.status == 3:
        assert r["status"] == "unbounded", txt
    else:
        assert s.status == 0
        assert r["status"] in ("optimal", "multiple_optima"), txt
        assert abs(float(r["value"]) - (-s.fun if sense == "max" else s.fun)) < 1e-7, txt
