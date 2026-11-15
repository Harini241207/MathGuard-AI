"""Step-verifier tests: every scenario from the specification, using hand-written AI answers and
answers rendered with injected mistakes (backend.ai_render)."""
import pytest

from backend import bigm_solver, graphical_solver, simplex_solver, solver, two_phase_solver
from backend.ai_render import make_answer
from backend.lp_parser import parse_problem
from backend.step_verifier import verify
from tests import fixtures as F

SOLVERS = {"simplex": simplex_solver, "bigm": bigm_solver, "two_phase": two_phase_solver, "graphical": graphical_solver}


def run(q, ans, method="simplex"):
    prob = parse_problem(q)
    return verify(prob, method, ans, SOLVERS[method].solve(prob), solver.solve(prob))


def codes(r, status="incorrect"):
    return [s.get("code") for s in r["steps"] if s["status"] == status]


def test_completely_correct_markdown():
    r = run(F.S01_Q, F.S01_CORRECT_MD)
    assert r["summary"]["verdict"] == "verified"
    assert r["summary"]["incorrect_steps"] == 0 and r["summary"]["correct_steps"] >= 15
    assert len(r["tableaus"]) == 3


@pytest.mark.parametrize("ans", [F.S01_LATEX, F.S01_PLAIN])
def test_correct_in_other_formats(ans):
    r = run(F.S01_Q, ans)
    assert r["summary"]["verdict"] == "verified", [s for s in r["steps"] if s["status"] != "correct"]


def test_incorrect_row_operation_reported_once_with_downstream():
    r = run(F.S01_Q, F.S01_ROWOP_ERROR)
    assert codes(r) == ["E06"]
    bad = next(s for s in r["steps"] if s["status"] == "incorrect")
    assert bad["ai_value"] == "3" and bad["correct_value"] == "1" and bad["column"] == "RHS"
    assert any(s["status"] == "consequential" for s in r["steps"])
    assert r["summary"]["verdict"] == "incorrect_working"   # final answer itself is still right


def test_non_textbook_entering_is_a_deviation_not_an_error():
    r = run(F.S01_Q, F.S01_WRONG_ENTERING)
    assert codes(r) == []
    assert codes(r, "deviation") == ["E03"]
    assert r["summary"]["verdict"] == "verified"


def test_contradictory_answer_is_unverifiable_not_wrong():
    r = run(F.S01_Q, F.CONTRADICTORY)
    assert codes(r) == []
    assert any(s["status"] == "unverifiable" for s in r["steps"])
    assert r["summary"]["verdict"] == "partially_verified"


def test_incomplete_answer_without_tableaus():
    r = run(F.S01_Q, F.CLAIMS_ONLY)
    assert any(s["status"] == "incomplete" for s in r["steps"])
    assert r["summary"]["verdict"] == "partially_verified"
    assert codes(r) == []


def test_premature_termination_e08_dataset_case():
    r = run(F.S01_Q, F.E08_PREMATURE)
    assert "E13" in codes(r) and r["summary"]["verdict"] == "incorrect"


def test_unboundedness_misclassified_e10_dataset_case():
    r = run(F.E10_UNBOUNDED_Q, F.E10_FINITE_CLAIM)
    assert "E14" in codes(r)
    r2 = run(F.E10_UNBOUNDED_Q, F.E10_CORRECT)
    assert r2["summary"]["verdict"] == "verified"


def test_missed_alternative_optima_e09_dataset_case():
    r = run(F.E09_Q, F.E09_UNIQUE_CLAIM)
    assert "E13" in codes(r)


def test_min_problem_solved_as_max_minus_z_is_valid():
    r = run(F.MIN_LE_Q, F.MIN_AS_MAX)
    assert codes(r) == [] and r["summary"]["verdict"] == "verified"
    assert codes(r, "deviation")  # convention noted


def test_bigm_symbolic_m_correct_and_wrong_penalty():
    r = run(F.MIN_GE_Q, F.BIGM_CORRECT, "bigm")
    assert r["summary"]["verdict"] == "verified", [s for s in r["steps"] if s["status"] != "correct"]
    r2 = run(F.MIN_GE_Q, F.BIGM_WRONG_PENALTY, "bigm")
    assert set(codes(r2)) == {"E01"}


def test_two_phase_with_min_w_convention():
    r = run(F.MIN_GE_Q, F.TWO_PHASE_MIN_W, "two_phase")
    assert r["summary"]["verdict"] == "verified", [s for s in r["steps"] if s["status"] not in ("correct", "deviation")]
    groups = {s["group"].split(":")[0] for s in r["steps"]}
    assert "Phase I" in groups and "Phase II" in groups


def test_graphical_correct_and_wrong():
    r = run(F.G01_Q, F.G01_CORRECT, "graphical")
    assert r["summary"]["verdict"] == "verified"
    r2 = run(F.G01_Q, F.G01_WRONG, "graphical")
    c = codes(r2)
    assert "G01" in c and "G02" in c and r2["summary"]["verdict"] == "incorrect"
    # the infeasible corner is reported as a consequence of the wrong intersection, not as a separate mistake
    assert any(s["status"] == "consequential" and s.get("code") == "G03" for s in r2["steps"])


def test_graphical_wrong_objective_evaluation():
    ans = F.G01_CORRECT.replace("| B | (2, 2) | 10 |", "| B | (2, 2) | 12 |").replace("Z = 10 at", "Z = 12 at")
    r = run(F.G01_Q, ans, "graphical")
    assert "G04" in codes(r) and "E12" in codes(r)


def test_restated_objective_coefficient_error_e01():
    ans = "Maximize Z = 5x + 2y subject to the constraints. " + F.S01_CORRECT_MD
    r = run("Maximize Z = 3x + 5y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0", ans.replace("Maximize Z = 3x + 5y + 0s1", "Max Z = 3x + 5y + 0s1"))
    assert "E01" in codes(r)


def test_slack_sign_error_e02():
    ans = F.S01_CORRECT_MD.replace("x + y + s1 = 4", "x + y - s1 = 4")
    r = run(F.S01_Q, ans)
    assert "E02" in codes(r)


def test_variable_naming_x3_x4_for_slacks():
    q = "Maximize Z = 5x1 + 4x2 subject to 6x1 + 4x2 <= 24, x1 + 2x2 <= 6, -x1 + x2 <= 1, x2 <= 2, x1, x2 >= 0"
    prob = parse_problem(q)
    txt, _ = make_answer(prob, None, seed=1, slack_naming="x", decimals=True)
    r = verify(prob, "simplex", txt, simplex_solver.solve(prob), solver.solve(prob))
    assert r["summary"]["verdict"] == "verified"
    assert any("x3" in n for n in r["notes"])


PROBS = {
    "S01": "Maximize Z = 3x + 5y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0",
    "S06": "Maximize Z = 4x + 3y subject to 2x + y <= 8, x + y <= 5, x <= 3, x, y >= 0",
    "TAHA": "Maximize Z = 5x1 + 4x2 subject to 6x1 + 4x2 <= 24, x1 + 2x2 <= 6, -x1 + x2 <= 1, x2 <= 2, x1, x2 >= 0",
    "S09": "Maximize Z = 3x + 2y + 4z subject to x + y + z <= 6, 2x + y <= 8, y + 2z <= 8, x, y, z >= 0",
}
EXPECT = {
    None: ([], "verified"),
    "entering_not_best": ([], "verified"),
    "leaving_bad": (["E04"], "incorrect"),
    "ratio_text": (["E04"], "incorrect_working"),
    "pivot_claim": (["E11"], "incorrect_working"),
    "rowop_cell": (["E06"], None),
    "zj_cell": (["E07"], "incorrect_working"),
    "premature_stop": (["E13"], "incorrect"),
    "wrong_value": (["E12"], "incorrect"),
}


@pytest.mark.parametrize("pname", list(PROBS))
@pytest.mark.parametrize("mistake", list(EXPECT))
@pytest.mark.parametrize("style", ["md", "latex", "plain"])
def test_injected_mistake_matrix(pname, mistake, style):
    prob = parse_problem(PROBS[pname])
    txt, applied = make_answer(prob, mistake, at_iter=0, seed=3, style=style,
                               slack_naming="x" if pname == "TAHA" else "s", decimals=pname == "TAHA")
    if mistake and not applied:
        pytest.skip("mistake not applicable to this problem")
    r = verify(prob, "simplex", txt, simplex_solver.solve(prob), solver.solve(prob))
    want_codes, verdict = EXPECT[mistake]
    got = codes(r)
    for c in want_codes:
        assert c in got, (pname, mistake, style, got)
    if not want_codes:
        assert got == [], (pname, mistake, style, [s for s in r["steps"] if s["status"] == "incorrect"])
    if verdict:
        assert r["summary"]["verdict"] == verdict, (pname, mistake, style, r["summary"])
    # the first reported error is in the iteration where the mistake was made
    if want_codes and want_codes[0] in ("E04", "E11", "E06"):
        first = next(s for s in r["steps"] if s["status"] == "incorrect")
        assert first["group"] == "Iteration 1", first


def test_entering_variable_that_does_not_improve():
    prob = parse_problem(PROBS["S06"])
    txt, applied = make_answer(prob, "entering_bad", at_iter=1, seed=2)
    assert applied
    r = verify(prob, "simplex", txt, simplex_solver.solve(prob), solver.solve(prob))
    assert "E03" in codes(r)
