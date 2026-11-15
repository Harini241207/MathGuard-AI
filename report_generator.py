"""Builds the complete verification report returned by /api/analyze.

Pipeline
--------
question --lp_parser--> problem
problem  --chosen method solver (graphical / simplex / Big M / two-phase)--> MathGuard's own working
problem  --two-phase exact solver--> ground-truth summary (status, optimum, alternative optima)
answer   --solution_parser + step_verifier--> step-by-step comparison, error categories
         --scipy_validator--> independent final-answer cross-check (secondary only)
         --reference store--> textbook passages for the method and for each detected error
         --ML model--> experimental opinion on the final-answer category (never overrides the verifier)
"""
from __future__ import annotations

from . import bigm_solver, graphical_solver, lp_common, scipy_validator, simplex_solver, solver, two_phase_solver
from .error_catalog import CATALOG
from .exact import fmt_q
from .graphical_solver import GraphicalError
from .lp_parser import ParseError, fmt_num, parse_problem
from .step_verifier import verify
from .tableau import UnsupportedForMethod, standard_form

METHODS = {
    "auto": {"name": "Detect automatically", "implemented": True, "kind": "lp"},
    "graphical": {"name": "Graphical method", "implemented": True, "kind": "lp"},
    "simplex": {"name": "Simplex method", "implemented": True, "kind": "lp"},
    "bigm": {"name": "Big M method", "implemented": True, "kind": "lp"},
    "two_phase": {"name": "Two-Phase method", "implemented": True, "kind": "lp"},
    "transportation": {"name": "Transportation problem (NWC / LCM / VAM + MODI)", "implemented": True, "kind": "transport"},
    "assignment": {"name": "Assignment problem (Hungarian method)", "implemented": True, "kind": "assignment"},
}
LP_SOLVERS = {"graphical": graphical_solver, "simplex": simplex_solver, "bigm": bigm_solver, "two_phase": two_phase_solver}
AI_SOURCES = ["ChatGPT", "Gemini", "Claude", "Other", "Unknown"]


class ReportError(ValueError):
    def __init__(self, msg, hint=None, status=422):
        super().__init__(msg)
        self.hint = hint
        self.status = status


def choose_method(prob, answer, requested):
    """Resolve 'auto' to a concrete method from the problem and the style of the answer."""
    if requested and requested != "auto":
        return requested, None
    low = (answer or "").lower()
    needs_art = False
    try:
        standard_form(prob, "simplex")
    except UnsupportedForMethod:
        needs_art = True
    import re
    tabular = bool(re.search(r"tableau|\bbasis\b|\bbasic\b|c_?j\s*[-−]\s*z_?j|\bzj\b|\bc_?b\b|\bslack|\bentering|\bpivot",
                             low))
    graphical_words = bool(re.search(r"corner|vert(?:ex|ices)|graphical|feasible region|intercept|iso-?profit", low))
    if "phase i" in low or "phase 1" in low or "two-phase" in low or "two phase" in low:
        return "two_phase", "The answer mentions Phase I / Phase II, so the Two-Phase method was assumed."
    if needs_art and ("big m" in low or "big-m" in low or "penalty" in low or "+ma" in low.replace(" ", "") or "-ma" in low.replace(" ", "")):
        return "bigm", "The answer uses a Big M penalty, so the Big M method was assumed."
    if len(prob["variables"]) == 2 and (graphical_words or not tabular):
        return "graphical", "Two decision variables and a corner-point / graphical answer: the graphical method was assumed."
    if needs_art:
        return "bigm", "The problem has '>=' or '=' constraints, so the Big M method was assumed."
    return "simplex", "The answer uses tableaus, so the Simplex method was assumed."


def _truth_text(prob, truth):
    vs = prob["variables"]
    st = truth["status"]
    if st == "infeasible":
        return "Infeasible: the constraints cannot all hold at once, so there is no solution."
    if st == "unbounded":
        return "Unbounded: Z can be improved forever, so there is no finite optimum."
    if st == "multiple_optima":
        ends = "; ".join("(" + ", ".join(fmt_num(p[v]) for v in vs) + ")" for p in truth["face"][:4])
        return f"Multiple optimal solutions, all with Z = {fmt_num(truth['value'])}. Optimal corner points include {ends}."
    extra = " (degenerate vertex)" if truth["degenerate"] else ""
    return f"Unique optimum{extra}: " + ", ".join(f"{v} = {fmt_num(truth['solution'][v])}" for v in vs) + \
        f", Z = {fmt_num(truth['value'])}."


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


LEGACY = {"verified": "valid", "partially_verified": "partial", "incorrect": "invalid", "incorrect_working": "invalid",
          "unverifiable": "unverifiable"}


def analyze_lp(question, answer, method, ai_source, refs=None, model=None):
    try:
        prob = parse_problem(question)
    except ParseError as e:
        raise ReportError(f"Couldn't understand the question: {e}",
                          "Write it like: Maximize Z = 3x + 2y subject to x + y <= 4, x <= 2, x, y >= 0", 422)
    method, method_note = choose_method(prob, answer, method)
    if method not in LP_SOLVERS:
        raise ReportError(f"Unknown method '{method}'.", status=400)
    try:
        ref = LP_SOLVERS[method].solve(prob)
    except (UnsupportedForMethod, GraphicalError) as e:
        raise ReportError(str(e), "Choose another method in the list.", 422)
    exact = solver.solve_exact(prob)
    truth = solver.summarize(prob, exact)
    # internal consistency between the selected method and the general two-phase solver
    consistent = ref["status"] == truth["status"] or {ref["status"], truth["status"]} <= {"optimal", "multiple_optima"}
    if ref.get("status") in ("optimal", "multiple_optima") and truth["value"] is not None:
        val = ref["optimum"]["value"] if method == "graphical" else ref["value"]
        consistent = consistent and abs(float(val) - truth["value"]) < 1e-9
    result = verify(prob, method, answer, ref, truth)
    vs = prob["variables"]
    claim = result["claim"]
    claim_pt = {v: claim["point"][v].v for v in vs} if all(v in claim["point"] for v in vs) else None
    working = graphical_solver.to_json(ref) if method == "graphical" else lp_common.to_json(ref)
    cross = scipy_validator.cross_check(prob, truth)

    # textbook passages
    error_codes = []
    for s in result["steps"]:
        if s["status"] in ("incorrect", "consequential") and s.get("code") and s["code"] not in error_codes:
            error_codes.append(s["code"])
    textbook = {"loaded": bool(refs and refs.status()["loaded"]), "method_passages": [], "by_error": {}}
    if textbook["loaded"]:
        textbook["method_passages"] = refs.method_passages(method, question)
        textbook["by_error"] = refs.for_errors(method, error_codes, question)
    general = {c: {"title": CATALOG[c]["title"], "note": CATALOG[c]["note"]} for c in error_codes if c in CATALOG}

    # ML model: experimental opinion on the final-answer category only
    ml = None
    if model is not None and result["final"]["label"] != "unclear":
        feats = result["features"]
        proba = model["model"].predict_proba([[feats[k] for k in model["features"]]])[0]
        pred = model["classes"][int(proba.argmax())]
        ml = {"prediction": pred, "agrees": pred == result["final"]["label"],
              "probabilities": {c: float(p) for c, p in zip(model["classes"], proba)},
              "note": "Experimental. The model only re-predicts the final-answer category from solver-derived features; "
                      "the verdict always comes from the mathematical verifier."}
    summary = result["summary"]
    det = result["details"]
    reasons = [{"kind": "info", "text": "MathGuard's own solution: " + _truth_text(prob, truth)},
               {"kind": "info", "text": "AI answer, as understood: " + _claim_text(claim, vs)}]
    for s in result["steps"]:
        if s["status"] == "incorrect":
            reasons.append({"kind": "bad", "text": f"{s['group']} - {s['title']}: {s['message']}"})
    if summary["verdict"] == "verified":
        reasons.append({"kind": "ok", "text": summary["verdict_text"] + "."})
    return {
        # ---- new report
        "method": method, "method_name": METHODS[method]["name"], "method_note": method_note, "ai_source": ai_source,
        "report": {"summary": summary, "steps": result["steps"], "tableaus": result["tableaus"], "notes": result["notes"],
                   "parsed": result["parsed"]},
        "solution": working, "consistency": {"method_matches_general_solver": bool(consistent)},
        "cross_check": cross, "textbook": textbook, "general_notes": general, "ml": ml,
        # ---- fields kept from the original API
        "problem": {"objective": prob["objective_text"], "constraints": [c["text"] for c in prob["constraints"]],
                    "variables": vs, "warnings": prob["warnings"]},
        "truth": {"status": truth["status"], "text": _truth_text(prob, truth), "solution": truth["solution"],
                  "value": truth["value"], "degenerate": truth["degenerate"], "face": truth["face"]},
        "claim": {"status": claim["status"], "text": _claim_text(claim, vs), "point": claim_pt,
                  "value": claim["value"].v if claim["value"] else None, "unique": claim["unique"]},
        "plot": solver.plot_data(prob, truth, claim_pt),
        "checks": det["check"]["rows"] if det.get("check") else [],
        "verdict": LEGACY[summary["verdict"]], "label_text": summary["verdict_text"],
        "error_type": result["final"]["label"] if result["final"]["label"] != "unclear" else None,
        "reasons": reasons,
        "references": [p for ps in textbook["by_error"].values() for p in ps][:3] or textbook["method_passages"][:2],
        "confidence": max(ml["probabilities"].values()) if ml else None,
        "probabilities": ml["probabilities"] if ml else {}, "solver_agrees": ml["agrees"] if ml else None,
        "degenerate": truth["degenerate"],
    }


def solve_only(question, method):
    prob = parse_problem(question)
    method, note = choose_method(prob, "", method)
    try:
        ref = LP_SOLVERS[method].solve(prob)
    except (UnsupportedForMethod, GraphicalError) as e:
        raise ReportError(str(e), "Choose another method.", 422)
    working = graphical_solver.to_json(ref) if method == "graphical" else lp_common.to_json(ref)
    extra = {}
    if len(prob["variables"]) == 2:
        truth = solver.summarize(prob, solver.solve_exact(prob))
        extra = {"plot": solver.plot_data(prob, truth, None),
                 "truth": {"status": truth["status"], "solution": truth["solution"], "face": truth["face"]},
                 "claim": {"point": None}}
    return {**extra, "method": method, "method_name": METHODS[method]["name"], "method_note": note, "solution": working,
            "problem": {"objective": prob["objective_text"], "constraints": [c["text"] for c in prob["constraints"]],
                        "variables": prob["variables"], "warnings": prob["warnings"]}}
