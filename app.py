"""MathGuard AI web server.  Run from the project root:  python -m backend.app"""
import json
import random
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

from . import datagen, features, report_generator, solver, train
from .answer_parser import parse_answer
from .lp_parser import ParseError, fmt_num, parse_problem
from .reference import ReferenceStore, extract_text
from .report_generator import AI_SOURCES, METHODS, ReportError

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
refs = ReferenceStore()

if not train.MODEL_PATH.exists():
    train.main()
_bundle = joblib.load(train.MODEL_PATH)
MODEL, FEATS, CLASSES = _bundle["model"], _bundle["features"], _bundle["classes"]
ML = {"model": MODEL, "features": FEATS, "classes": CLASSES}

LABEL_TEXT = {
    "correct": "Correct",
    "wrong_status": "Wrong problem status",
    "wrong_value": "Wrong objective value",
    "suboptimal_solution": "Feasible but not optimal",
    "infeasible_solution": "Violates a constraint",
    "missed_multiple_optima": "Missed alternative optima",
}


# ------------------------------------------------------------------ helpers
def _pt_text(pt, vs):
    return ", ".join(f"{v} = {fmt_num(pt[v])}" for v in vs)


def _truth_text(prob, truth):
    vs = prob["variables"]
    st = truth["status"]
    if st == "infeasible":
        return "Infeasible: the constraints cannot all hold at once, so there is no solution."
    if st == "unbounded":
        return "Unbounded: Z can be improved forever, so there is no finite optimum."
    if st == "multiple_optima":
        ends = "; ".join("(" + ", ".join(fmt_num(p[v]) for v in vs) + ")" for p in truth["face"][:4])
        return f"Multiple optimal solutions, all with Z = {fmt_num(truth['value'])}. Corner optima include {ends}."
    extra = " (degenerate vertex)" if truth["degenerate"] else ""
    return f"Unique optimum{extra}: {_pt_text(truth['solution'], vs)}, Z = {fmt_num(truth['value'])}."


def _claim_text(claim, vs):
    st = claim["status"]
    if st == "infeasible":
        return "The problem is infeasible."
    if st == "unbounded":
        return "The problem is unbounded."
    bits = []
    if claim["point"]:
        bits.append(", ".join(f"{v} = {fmt_num(claim['point'][v].v)}" for v in vs if v in claim["point"]))
    if claim["value"] is not None:
        bits.append(f"Z = {fmt_num(claim['value'].v)}")
    prefix = "Multiple optima: " if st == "multiple" else ("Unique optimum: " if claim["unique"] else "")
    return prefix + (", ".join(bits) if bits else "(no solution or value found)")


def explain(prob, truth, claim, f, det, label):
    vs = prob["variables"]
    out = [{"kind": "info", "text": "Exact solver: " + _truth_text(prob, truth)},
           {"kind": "info", "text": "AI answer, as understood: " + _claim_text(claim, vs)}]
    chk = det["check"]
    if label == "wrong_status":
        names = {"optimal": "has a finite optimum", "multiple_optima": "has several optimal solutions",
                 "infeasible": "is infeasible", "unbounded": "is unbounded"}
        said = {"optimal": "has a single finite optimum", "multiple": "has several optimal solutions",
                "infeasible": "is infeasible", "unbounded": "is unbounded"}
        out.append({"kind": "bad", "text": f"The AI says the problem {said[claim['status']]}, but it actually {names[truth['status']]}."})
    elif label == "missed_multiple_optima":
        out.append({"kind": "bad", "text": "The AI calls the optimum unique, but the objective line is parallel to a binding constraint, so a whole edge of points is optimal."})
    elif label == "infeasible_solution":
        bad = [r for r in chk["rows"] if not r["ok"]] if chk else []
        for r in bad[:4]:
            out.append({"kind": "bad", "text": f"The AI's point breaks \"{r['text']}\" (left side evaluates to {fmt_num(round(r['lhs'], 6))})."})
        if not bad:
            out.append({"kind": "bad", "text": "The AI's point is not feasible."})
    elif label == "suboptimal_solution":
        if chk:
            out.append({"kind": "bad", "text": f"The AI's point is feasible but gives Z = {fmt_num(round(chk['objective'], 6))}, not the true optimum Z = {fmt_num(truth['value'])}."})
        else:
            out.append({"kind": "bad", "text": "The coordinates the AI gave do not match the optimal solution."})
    elif label == "wrong_value":
        out.append({"kind": "bad", "text": f"The AI reports Z = {fmt_num(claim['value'].v)}, but the true optimum is Z = {fmt_num(truth['value'])}."})
        if det["value_obj_at_point"] is not None and not f["val_consistent"]:
            out.append({"kind": "bad", "text": f"It is also internally inconsistent: its own point gives Z = {fmt_num(round(det['value_obj_at_point'], 6))}."})
    else:
        if truth["status"] in ("infeasible", "unbounded"):
            out.append({"kind": "ok", "text": f"The AI correctly identified the problem as {truth['status']}."})
        else:
            if f["sol_state"] == 2:
                out.append({"kind": "ok", "text": "The AI's point satisfies every constraint and non-negativity."})
                out.append({"kind": "ok", "text": "Its objective value is the best achievable (matches the solver)."})
            elif f["sol_state"] == 1:
                out.append({"kind": "info", "text": "Only some variables were given; the ones provided match the solver."})
            if f["val_provided"]:
                out.append({"kind": "ok", "text": "The reported objective value matches the solver."})
            elif f["sol_state"] == 2:
                out.append({"kind": "info", "text": "No objective value was stated, so only the solution point was checked."})
            if truth["status"] == "multiple_optima":
                out.append({"kind": "info", "text": "Note: this problem has alternative optima, so other optimal points are also valid."})
    return out


def _build_examples():
    rows = [json.loads(l) for l in (ROOT / "data" / "mathguard_30q_dataset_step2.jsonl").read_text().splitlines()]
    ex = []
    cycle = ["other_vertex", "wrong_value", "infeasible_point", "swap_sense", "coord_swap", "wrong_status"]
    k = 0
    for r in rows:
        if r["task_type"] != "solve_and_verify":
            continue
        p = r["problem"]
        o = p["objective"]
        q = (f"{'Maximize' if o['sense'] == 'max' else 'Minimize'} Z = {o['expression']}\nsubject to\n"
             + "\n".join("  " + c for c in p["constraints"]) + "\n  " + ", ".join(v + " >= 0" for v in p["variables"]))
        prob = parse_problem(q)
        truth = solver.solve(prob)
        rng = random.Random(1000 + k)
        spec = {"status": "optimal", "point": truth.get("solution"), "points": [], "value": truth.get("value"), "unique": False}
        if truth["status"] in ("infeasible", "unbounded"):
            spec = {"status": truth["status"], "point": None, "points": [], "value": None, "unique": False}
        elif truth["status"] == "multiple_optima" and len(truth["face"]) >= 2 and len(prob["variables"]) == 2:
            spec.update(status="multiple", points=truth["face"][:2], point=truth["face"][0])
        right = datagen.render_answer(prob, spec, rng)
        wrong = None
        for j in range(12):
            beh = cycle[(k + j) % len(cycle)]
            ws = datagen.make_claim(random.Random(5000 + k * 7 + j), prob, truth, beh)
            if ws["status"] in ("optimal", "multiple") and ws["point"] is None and ws["value"] is None:
                continue
            txt = datagen.render_answer(prob, ws, random.Random(9000 + k))
            c = parse_answer(txt, prob["variables"])
            ff, _ = features.build(prob, truth, c)
            if features.judge(ff) not in ("correct", "unclear"):
                wrong = (txt, features.judge(ff))
                break
        meta = f"{r['id']} · {r['method']}"
        ex.append({"label": f"{meta} · final answer only (correct)", "question": q, "answer": right, "expected": "correct",
                   "method": r["method"], "group": "Final-answer checks (original dataset)"})
        if wrong:
            ex.append({"label": f"{meta} · final answer only (with a mistake)", "question": q, "answer": wrong[0],
                       "expected": wrong[1], "method": r["method"], "group": "Final-answer checks (original dataset)"})
        k += 1
    for r in rows:
        if r["id"] in ("E08", "E09", "E10"):
            q = r["problem"].replace("; ", "\nsubject to\n  ", 1).replace(", ", "\n  ", 1) if False else r["problem"]
            ex.append({"label": f"{r['id']} · student error · {r['title']}", "question": q,
                       "answer": r["student_submission"], "expected": "invalid", "method": "auto",
                       "group": "Student errors (original dataset)"})
    step_file = ROOT / "data" / "step_examples.json"
    if step_file.exists():
        ex = json.loads(step_file.read_text()) + ex
    return ex


EXAMPLES = _build_examples()


# ------------------------------------------------------------------ routes
@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/")
def index():
    return send_from_directory(ROOT / "frontend", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "classes": CLASSES, "reference": refs.status(),
                    "primary_solver": "MathGuard exact textbook algorithms (Fractions)",
                    "methods": [k for k, v in METHODS.items() if v["implemented"]]})


@app.get("/api/examples")
def examples():
    return jsonify(EXAMPLES)


@app.get("/api/metrics")
def metrics():
    p = train.METRICS_PATH
    return jsonify(json.loads(p.read_text()) if p.exists() else {})


@app.get("/api/methods")
def methods():
    return jsonify({"methods": [{"id": k, **v} for k, v in METHODS.items()], "ai_sources": AI_SOURCES})


@app.post("/api/analyze")
def analyze():
    """Verify an AI-generated solution.  Body: {question, answer, method?, ai_source?}."""
    body = request.get_json(silent=True) or {}
    question, answer = (body.get("question") or "").strip(), (body.get("answer") or "").strip()
    method = (body.get("method") or "auto").strip()
    ai_source = (body.get("ai_source") or "Unknown").strip()
    if ai_source not in AI_SOURCES:
        ai_source = "Other"
    if not question or not answer:
        return jsonify({"error": "Please provide both the question and the AI-generated answer."}), 400
    if method not in METHODS:
        return jsonify({"error": f"Unknown method '{method}'.", "hint": "Choose one of: " + ", ".join(METHODS)}), 400
    try:
        kind = METHODS[method]["kind"]
        if kind == "transport":
            from . import transportation_solver
            return jsonify(transportation_solver.analyze(question, answer, ai_source, refs))
        if kind == "assignment":
            from . import assignment_solver
            return jsonify(assignment_solver.analyze(question, answer, ai_source, refs))
        return jsonify(report_generator.analyze_lp(question, answer, method, ai_source, refs, ML))
    except ReportError as e:
        return jsonify({"error": str(e), "hint": e.hint}), e.status
    except ValueError as e:
        return jsonify({"error": str(e)}), 422


@app.post("/api/solve")
def solve_route():
    """MathGuard's own step-by-step solution, without an AI answer.  Body: {question, method?}."""
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    method = (body.get("method") or "auto").strip()
    if not question:
        return jsonify({"error": "Please provide the question."}), 400
    try:
        kind = METHODS.get(method, {}).get("kind", "lp")
        if kind == "transport":
            from . import transportation_solver
            return jsonify(transportation_solver.solve_text(question))
        if kind == "assignment":
            from . import assignment_solver
            return jsonify(assignment_solver.solve_text(question))
        return jsonify(report_generator.solve_only(question, method))
    except ReportError as e:
        return jsonify({"error": str(e), "hint": e.hint}), e.status
    except (ParseError, ValueError) as e:
        return jsonify({"error": str(e)}), 422


@app.post("/api/extract")
def extract():
    fl = request.files.get("file")
    if not fl:
        return jsonify({"error": "No file uploaded."}), 400
    try:
        pages = extract_text(fl.filename, fl.read())
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Could not read the file: {e}"}), 422
    text = "\n".join(t for _, t in pages).strip()
    if not text:
        return jsonify({"error": "No text found in that file (scanned images need OCR)."}), 422
    return jsonify({"text": text[:20000], "pages": len(pages)})


@app.get("/api/reference")
def reference_status():
    return jsonify(refs.status())


@app.post("/api/reference/upload")
def reference_upload():
    fl = request.files.get("file")
    if not fl:
        return jsonify({"error": "No file uploaded."}), 400
    try:
        info = refs.add(fl.filename, fl.read())
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 422
    return jsonify({"added": info, **refs.status()})


@app.delete("/api/reference")
def reference_clear():
    refs.clear()
    return jsonify(refs.status())


if __name__ == "__main__":
    print("MathGuard AI running on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
