"""Ground truth for a linear program, computed by MathGuard's own exact algorithms.

``solve(prob)`` runs the Two-Phase simplex method (exact Fractions), which handles
every constraint type, and returns the summary dict used by the feature builder:
status, optimal point, value, optimal face (alternative optimal vertices) and
degeneracy.  SciPy is no longer used here; see ``scipy_validator`` for the
independent cross-check.

``vertices``, ``check_point`` and ``plot_data`` are geometric helpers (NumPy only).
"""
import itertools

import numpy as np

from . import two_phase_solver


def _mats(prob):
    vs = prob["variables"]
    n = len(vs)
    idx = {v: i for i, v in enumerate(vs)}
    c = np.array([prob["objective"].get(v, 0.0) for v in vs], dtype=float)
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for con in prob["constraints"]:
        a = np.zeros(n)
        for v, k in con["coefs"].items():
            a[idx[v]] = k
        if con["op"] == "<=":
            A_ub.append(a); b_ub.append(con["rhs"])
        elif con["op"] == ">=":
            A_ub.append(-a); b_ub.append(-con["rhs"])
        else:
            A_eq.append(a); b_eq.append(con["rhs"])
    return (
        c,
        np.array(A_ub).reshape(-1, n), np.array(b_ub, dtype=float),
        np.array(A_eq).reshape(-1, n), np.array(b_eq, dtype=float),
    )


def solve_exact(prob):
    """Full exact two-phase result (with every tableau)."""
    return two_phase_solver.solve(prob)


def summarize(prob, res):
    """Convert an exact tableau-method result into the summary dict used across MathGuard."""
    vs = prob["variables"]
    out = {"status": res["status"], "degenerate": False, "solution": None, "value": None, "face": [], "exact": res}
    if prob.get("trivially_infeasible"):
        out["status"] = "infeasible"
        return out
    if res["status"] in ("infeasible", "unbounded"):
        return out
    out["solution"] = {v: float(res["solution"][v]) for v in vs}
    out["value"] = float(res["value"])
    last = res["phases"][-1]["iterations"][-1]
    out["degenerate"] = any(x == 0 for x in last["b"])
    out["face"] = [{v: float(p[i]) for i, v in enumerate(vs)} for p in res["optimal_points"]]
    return out


def solve(prob):
    return summarize(prob, solve_exact(prob))


def vertices(prob, extra_ub=None):
    """All feasible basic solutions (vertices). extra_ub = list of (coef_list, rhs) for A x <= b."""
    c, A_ub, b_ub, A_eq, b_eq = _mats(prob)
    n = len(c)
    rows, rhs = [], []
    for a, b in zip(A_ub, b_ub):
        rows.append(a); rhs.append(b)
    for a, b in zip(A_eq, b_eq):
        rows.append(a); rhs.append(b)
    for j in range(n):
        e = np.zeros(n); e[j] = 1
        rows.append(e); rhs.append(0.0)
    for a, b in (extra_ub or []):
        rows.append(np.array(a, dtype=float)); rhs.append(float(b))
    rows = np.array(rows); rhs = np.array(rhs)
    out = []
    for combo in itertools.combinations(range(len(rows)), n):
        M = rows[list(combo)]
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        x = np.linalg.solve(M, rhs[list(combo)])
        if np.any(x < -1e-7):
            continue
        ok = True
        for a, b in zip(A_ub, b_ub):
            if a @ x > b + 1e-7:
                ok = False; break
        if ok:
            for a, b in zip(A_eq, b_eq):
                if abs(a @ x - b) > 1e-7:
                    ok = False; break
        if ok:
            for a, b in (extra_ub or []):
                if np.dot(a, x) > b + 1e-7:
                    ok = False; break
        if ok and not any(np.allclose(x, y, atol=1e-7) for y in out):
            out.append(x)
    return out


def check_point(prob, pt, ptol=0.0):
    """Check a claimed point against all constraints + non-negativity."""
    rows, feasible, maxviol = [], True, 0.0
    for con in prob["constraints"]:
        lhs = sum(k * pt[v] for v, k in con["coefs"].items())
        scale = sum(abs(k) for k in con["coefs"].values())
        tol = 1e-6 + ptol * scale
        if con["op"] == "<=":
            viol = lhs - con["rhs"]
        elif con["op"] == ">=":
            viol = con["rhs"] - lhs
        else:
            viol = abs(lhs - con["rhs"])
        ok = viol <= tol
        feasible &= ok
        maxviol = max(maxviol, max(viol, 0.0) / (1 + abs(con["rhs"])))
        rows.append({"text": con["text"], "lhs": float(lhs), "op": con["op"], "rhs": con["rhs"],
                     "ok": bool(ok), "violation": float(max(viol, 0.0))})
    for v in prob["variables"]:
        ok = pt[v] >= -(1e-6 + ptol)
        feasible &= ok
        if not ok:
            maxviol = max(maxviol, -pt[v])
        rows.append({"text": f"{v} >= 0", "lhs": float(pt[v]), "op": ">=", "rhs": 0.0,
                     "ok": bool(ok), "violation": float(max(-pt[v], 0.0))})
    obj = sum(prob["objective"].get(v, 0.0) * pt[v] for v in prob["variables"])
    return {"feasible": bool(feasible), "objective": float(obj), "rows": rows, "max_violation": float(maxviol)}


def plot_data(prob, truth, claim_pt=None):
    """Geometry for the 2-variable feasible-region chart."""
    vs = prob["variables"]
    if len(vs) != 2:
        return None
    xs = [0.0]
    ys = [0.0]
    for v in vertices(prob):
        xs.append(v[0]); ys.append(v[1])
    for p in ([truth["solution"]] if truth.get("solution") else []) + ([claim_pt] if claim_pt else []):
        xs.append(p[vs[0]]); ys.append(p[vs[1]])
    for con in prob["constraints"]:
        a, b = con["coefs"].get(vs[0], 0.0), con["coefs"].get(vs[1], 0.0)
        if abs(a) > 1e-12 and abs(b) < 1e-12:
            xs.append(con["rhs"] / a)
        if abs(b) > 1e-12 and abs(a) < 1e-12:
            ys.append(con["rhs"] / b)
    M = max(max(xs), max(ys), 1.0) * 1.25
    M = float(max(M, 4.0))
    poly = vertices(prob, extra_ub=[([1, 0], M), ([0, 1], M)])
    polygon = []
    if poly:
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        poly = sorted(poly, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        polygon = [[float(p[0]), float(p[1])] for p in poly]
    lines = []
    for con in prob["constraints"]:
        a, b, r = con["coefs"].get(vs[0], 0.0), con["coefs"].get(vs[1], 0.0), con["rhs"]
        pts = []
        if abs(b) > 1e-12:
            for x in (0.0, M):
                y = (r - a * x) / b
                if -1e-9 <= y <= M + 1e-9:
                    pts.append([x, y])
        if abs(a) > 1e-12:
            for y in (0.0, M):
                x = (r - b * y) / a
                if -1e-9 <= x <= M + 1e-9:
                    pts.append([x, y])
        uniq = []
        for p in pts:
            if not any(abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9 for q in uniq):
                uniq.append(p)
        if len(uniq) >= 2:
            lines.append({"label": con["text"], "p1": uniq[0], "p2": uniq[1]})
    return {"axes": vs, "max": M, "polygon": polygon, "lines": lines}
