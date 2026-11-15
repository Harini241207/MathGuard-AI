"""End-to-end tests through the Flask API (all methods, all samples, legacy endpoints)."""
import json

import pytest

from backend.app import EXAMPLES, app
from tests import fixtures as F


@pytest.fixture(scope="module")
def client():
    return app.test_client()


def post(c, url, body):
    r = c.post(url, json=body)
    return r.status_code, r.get_json()


def test_health_and_methods(client):
    h = client.get("/api/health").get_json()
    assert h["ok"] and "simplex" in h["methods"] and "transportation" in h["methods"]
    m = client.get("/api/methods").get_json()
    assert {x["id"] for x in m["methods"]} >= {"graphical", "simplex", "bigm", "two_phase", "transportation", "assignment"}
    assert "ChatGPT" in m["ai_sources"]


@pytest.mark.parametrize("q,a,method,verdict", [
    (F.S01_Q, F.S01_CORRECT_MD, "simplex", "verified"),
    (F.S01_Q, F.S01_ROWOP_ERROR, "simplex", "incorrect_working"),
    (F.MIN_GE_Q, F.BIGM_CORRECT, "bigm", "verified"),
    (F.MIN_GE_Q, F.TWO_PHASE_MIN_W, "two_phase", "verified"),
    (F.G01_Q, F.G01_WRONG, "graphical", "incorrect"),
    (F.G01_Q, F.G01_CORRECT, "auto", "verified"),
])
def test_analyze_lp(client, q, a, method, verdict):
    code, j = post(client, "/api/analyze", {"question": q, "answer": a, "method": method, "ai_source": "ChatGPT"})
    assert code == 200, j
    assert j["report"]["summary"]["verdict"] == verdict
    assert j["ai_source"] == "ChatGPT"
    assert j["cross_check"]["agrees"] is True
    assert j["consistency"]["method_matches_general_solver"] is True
    # legacy keys kept for older clients
    for k in ("verdict", "label_text", "reasons", "truth", "claim", "plot", "checks", "problem"):
        assert k in j
    json.dumps(j)


def test_simplex_on_ge_problem_explains_alternative(client):
    code, j = post(client, "/api/analyze", {"question": F.MIN_GE_Q, "answer": "x = 2, y = 2, Z = 10", "method": "simplex"})
    assert code == 422 and "Big M" in j["error"]


def test_graphical_with_three_variables(client):
    code, j = post(client, "/api/analyze", {"question": "Maximize Z = x + y + z subject to x + y + z <= 3, x, y, z >= 0",
                                            "answer": "x = 3, Z = 3", "method": "graphical"})
    assert code == 422 and "two decision variables" in j["error"]


def test_bad_inputs(client):
    assert post(client, "/api/analyze", {"question": "", "answer": "x"})[0] == 400
    assert post(client, "/api/analyze", {"question": "Max Z = x", "answer": "x", "method": "nope"})[0] == 400
    code, j = post(client, "/api/analyze", {"question": "hello world", "answer": "x = 1"})
    assert code == 422 and "hint" in j


def test_transport_and_assignment_endpoints(client):
    code, j = post(client, "/api/analyze", {"question": "19 30 50 10 7\n70 30 40 60 9\n40 8 70 20 18\nDemand 5 8 7 14",
                                            "answer": "Minimum total cost = 743", "method": "transportation"})
    assert code == 200 and j["report"]["summary"]["verdict"] in ("verified", "partially_verified")
    code, j = post(client, "/api/solve", {"question": "9 11 14\n6 15 13\n12 13 6", "method": "assignment"})
    assert code == 200 and j["solution"]["total"]["v"] == 23


def test_solve_only_every_lp_method(client):
    for m, q in (("graphical", F.G01_Q), ("simplex", F.S01_Q), ("bigm", F.MIN_GE_Q), ("two_phase", F.MIN_GE_Q)):
        code, j = post(client, "/api/solve", {"question": q, "method": m})
        assert code == 200 and j["solution"]["status"] in ("optimal", "multiple_optima")
    code, j = post(client, "/api/solve", {"question": F.G01_Q, "method": "graphical"})
    assert j["plot"] is not None


def test_every_sample_runs(client):
    assert len(EXAMPLES) > 40
    for e in EXAMPLES:
        code, j = post(client, "/api/analyze", {"question": e["question"], "answer": e["answer"],
                                                "method": e.get("method", "auto"), "ai_source": e.get("source", "Unknown")})
        assert code in (200, 422), (e["label"], j)
        if code == 200 and e.get("group") == "Step-by-step checks":
            v = j["report"]["summary"]["verdict"]
            if e["expected"] == "correct":
                assert v in ("verified", "partially_verified"), (e["label"], v)
            else:
                assert v in ("incorrect", "incorrect_working", "partially_verified"), (e["label"], v)


def test_reference_upload_flow(client, tmp_path, monkeypatch):
    import backend.reference as R
    import backend.app as A
    monkeypatch.setattr(R, "STORE", tmp_path / "ref.pkl")
    monkeypatch.setattr(A, "refs", R.ReferenceStore())
    data = {"file": (__import__("io").BytesIO(b"The simplex method selects the leaving variable by the minimum ratio test. " * 20), "book.txt")}
    r = client.post("/api/reference/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["loaded"]
    code, j = post(client, "/api/analyze", {"question": F.S01_Q, "answer": F.S01_ROWOP_ERROR, "method": "simplex"})
    assert j["textbook"]["loaded"] and j["textbook"]["method_passages"]
    client.delete("/api/reference")
