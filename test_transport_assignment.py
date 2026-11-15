"""Transportation (NWC / LCM / VAM + MODI) and assignment (Hungarian) tests."""
import random
from fractions import Fraction as F

import pytest

from backend import assignment_solver as A, transportation_solver as T
from backend.matrix_parser import parse_assignment, parse_transportation

CLASSIC = """       D1  D2  D3  D4  Supply
S1     19  30  50  10   7
S2     70  30  40  60   9
S3     40   8  70  20  18
Demand  5   8   7  14"""


@pytest.mark.parametrize("method,init", [("nwc", 1015), ("lcm", 814), ("vam", 779)])
def test_classic_textbook_values(method, init):
    r = T.solve(parse_transportation(CLASSIC), method)
    assert r["initial_cost"] == init
    assert r["optimal_cost"] == 743
    last = r["iterations"][-1]
    assert last["optimal"] and all(d is None or d >= 0 for row in last["d"] for d in row)


def test_supply_demand_as_lists_and_markdown():
    q = """| | A | B | C |
|---|---|---|---|
| P | 2 | 7 | 4 |
| Q | 3 | 3 | 1 |
| R | 5 | 4 | 7 |
| S | 1 | 6 | 2 |
Supply: 5, 8, 7, 14
Demand: 7, 9, 18"""
    P = parse_transportation(q)
    assert P["supply"] == [5, 8, 7, 14] and P["demand"] == [7, 9, 18]
    assert P["col_labels"] == ["A", "B", "C"] and P["row_labels"] == ["P", "Q", "R", "S"]
    r = T.solve(P, "vam")
    assert r["optimal_cost"] == 76


def test_unbalanced_gets_dummy():
    q = "10 2 20 | 15\n12 7 9 | 25\nDemand 5 15 15"
    q = q.replace("|", " ")
    r = T.solve(parse_transportation(q), "lcm")
    assert r["problem"]["col_labels"][-1] == "Dummy" and r["problem"]["demand"][-1] == 5
    assert "dummy" in r["explanation"][0].lower()


def test_degenerate_epsilon_cells():
    q = "1 4 5 10\n6 2 7 15\n8 9 3 25\nDemand 10 15 25"
    r = T.solve(parse_transportation(q), "nwc")
    assert len(r["epsilon_cells"]) == 2
    assert r["optimal_cost"] == 10 + 30 + 75


def test_profit_maximisation():
    q = "Maximize total profit\n 12 18 6 | 10\n 10 7 15 | 20\nDemand 15 5 10".replace("|", " ")
    r = T.solve(parse_transportation(q), "vam")
    assert r["sense"] == "max"
    # best: S1->D1 5, S1->D2 5, S2->D1 10, S2->D3 10 : 60 + 90 + 100 + 150 = 400
    assert r["optimal_cost"] == 400


@pytest.mark.parametrize("seed", range(80))
@pytest.mark.parametrize("init", ["nwc", "vam"])
def test_random_transport_against_scipy(seed, init):
    from scipy.optimize import linprog
    rng = random.Random(seed)
    m, n = rng.randint(2, 4), rng.randint(2, 5)
    C = [[rng.randint(1, 20) for _ in range(n)] for _ in range(m)]
    s = [rng.randint(1, 30) for _ in range(m)]
    d = [rng.randint(1, 30) for _ in range(n)]
    q = "\n".join(" ".join(map(str, row)) + f" {s[i]}" for i, row in enumerate(C)) + "\nDemand " + " ".join(map(str, d))
    r = T.solve(parse_transportation(q), init)
    S, D = sum(s), sum(d)
    Cb = [row + [0] for row in C] if S > D else C + ([[0] * n] if D > S else [])
    sb = s + ([D - S] if D > S else [])
    db = d + ([S - D] if S > D else [])
    mm, nn = len(sb), len(db)
    A_eq, b_eq = [], []
    for i in range(mm):
        A_eq.append([1 if k // nn == i else 0 for k in range(mm * nn)]); b_eq.append(sb[i])
    for j in range(nn):
        A_eq.append([1 if k % nn == j else 0 for k in range(mm * nn)]); b_eq.append(db[j])
    res = linprog([Cb[i][j] for i in range(mm) for j in range(nn)], A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (mm * nn),
                  method="highs")
    assert abs(float(r["optimal_cost"]) - res.fun) < 1e-6
    # allocations respect totals
    fa = r["final_alloc"]
    assert all(sum(a or 0 for a in fa[i]) == r["problem"]["supply"][i] for i in range(mm))


def test_transport_verification():
    ok = """Using Vogel's Approximation Method the initial cost is 779.
After MODI: S1 -> D1 = 5, S1 -> D4 = 2, S2 -> D2 = 2, S2 -> D3 = 7, S3 -> D2 = 6, S3 -> D4 = 12.
Minimum total cost = 743."""
    r = T.analyze(CLASSIC, ok, "ChatGPT")
    assert r["report"]["summary"]["verdict"] in ("verified", "partially_verified")
    assert r["report"]["summary"]["incorrect_steps"] == 0
    bad = ok.replace("initial cost is 779", "initial cost is 799").replace("Minimum total cost = 743", "Minimum total cost = 779")
    r2 = T.analyze(CLASSIC, bad, "Gemini")
    codes = [s.get("code") for s in r2["report"]["steps"] if s["status"] == "incorrect"]
    assert "T01" in codes and "T02" in codes
    assert r2["cross_check"]["agrees"]


# ----------------------------------------------------------------------------- assignment
def test_hungarian_small_known():
    P = parse_assignment("     M1 M2 M3\nJ1  9 11 14\nJ2  6 15 13\nJ3 12 13  6")
    r = A.solve(P)
    assert r["total"] == 23
    assert dict(r["assignment"]) == {0: 1, 1: 0, 2: 2}
    names = [s["name"] for s in r["stages"]]
    assert names[:3] == ["Cost matrix", "Row reduction", "Column reduction"]
    assert r["stages"][1]["matrix"][0] == [0, 2, 5]


def test_hungarian_needs_adjustment():
    # Taha-style 4 x 4 example: two lines are not enough after the reductions, one adjustment is needed
    q = "1 4 6 3\n9 7 10 9\n4 5 11 7\n8 7 8 5"
    r = A.solve(parse_assignment(q))
    adj = [s for s in r["stages"] if s["name"].startswith("Adjustment")]
    assert adj and adj[0]["theta"] == 1
    assert r["stages"][2]["lines"]["count"] < 4
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    C = np.array([[1, 4, 6, 3], [9, 7, 10, 9], [4, 5, 11, 7], [8, 7, 8, 5]])
    rr, cc = linear_sum_assignment(C)
    assert r["total"] == C[rr, cc].sum()


def test_unbalanced_and_maximisation_and_forbidden():
    r = A.solve(parse_assignment("Maximize sales\n 32 38 40\n 40 24 28\n 41 27 33\n 22 38 41"))
    assert r["sense"] == "max" and r["problem"]["col_labels"][-1] == "Dummy"
    assert r["total"] == 38 + 41 + 41  # R1->C2, R3->C1, R4->C3 (R2 gets the dummy)
    rf = A.solve(parse_assignment("4 - 6\n2 5 3\n7 1 -"))
    assert rf["status"] in ("optimal", "multiple_optima")
    assert all(not hasattr(rf["problem"]["costs"][i][j], "a") for i, j in rf["assignment"])


@pytest.mark.parametrize("seed", range(120))
def test_random_assignment_against_scipy(seed):
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    rng = random.Random(seed)
    m, n = rng.randint(2, 6), rng.randint(2, 6)
    C = [[rng.randint(1, 30) for _ in range(n)] for _ in range(m)]
    sense = rng.choice(["min", "max"])
    q = ("Maximize\n" if sense == "max" else "") + "\n".join(" ".join(map(str, r)) for r in C)
    r = A.solve(parse_assignment(q))
    rr, cc = linear_sum_assignment(np.array(C), maximize=sense == "max")
    assert r["total"] == sum(C[i][j] for i, j in zip(rr, cc))


def test_assignment_verification():
    Q = "     M1 M2 M3\nJ1  9 11 14\nJ2  6 15 13\nJ3 12 13  6"
    ok = """Row reduction:
0 2 5
0 9 7
6 7 0
Column reduction:
0 0 5
0 7 7
6 5 0
Assignment: J1 -> M2, J2 -> M1, J3 -> M3. Minimum total cost = 23."""
    r = A.analyze(Q, ok, "Claude")
    assert r["report"]["summary"]["verdict"] == "verified", r["report"]["steps"]
    bad = ok.replace("0 9 7", "0 8 7").replace("J1 -> M2, J2 -> M1", "J1 -> M1, J2 -> M2").replace("= 23", "= 30")
    r2 = A.analyze(Q, bad, "Other")
    codes = [s.get("code") for s in r2["report"]["steps"] if s["status"] == "incorrect"]
    assert "H01" in codes and r2["report"]["summary"]["verdict"] == "incorrect"
