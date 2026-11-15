"""Independent SciPy/HiGHS cross-check.

MathGuard's displayed answers come from its own textbook algorithms (graphical,
simplex, Big M, two-phase ...).  This module solves the same LP with
``scipy.optimize.linprog`` purely as a secondary validation: agreement on the
final optimum is reported, but it is never treated as proof that the textbook
steps are right, and SciPy never overrides the primary result.
"""
import numpy as np
from scipy.optimize import linprog


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


def _lp(cm, A_ub, b_ub, A_eq, b_eq):
    n = len(cm)
    return linprog(
        cm,
        A_ub=A_ub if len(A_ub) else None, b_ub=b_ub if len(b_ub) else None,
        A_eq=A_eq if len(A_eq) else None, b_eq=b_eq if len(b_eq) else None,
        bounds=[(0, None)] * n, method="highs",
    )


def scipy_solve(prob):
    vs = prob["variables"]
    n = len(vs)
    c, A_ub, b_ub, A_eq, b_eq = _mats(prob)
    cm = -c if prob["sense"] == "max" else c
    res = _lp(cm, A_ub, b_ub, A_eq, b_eq)
    out = {"status": None, "degenerate": False, "solution": None, "value": None, "face": []}
    if res.status == 2:
        out["status"] = "infeasible"; return out
    if res.status == 3:
        out["status"] = "unbounded"; return out
    if res.status != 0:  # numerical trouble: feasibility probe
        probe = _lp(np.zeros(n), A_ub, b_ub, A_eq, b_eq)
        out["status"] = "unbounded" if probe.status == 0 else "infeasible"
        return out
    x = res.x
    z_m = float(res.fun)
    out["solution"] = {v: float(x[i]) for i, v in enumerate(vs)}
    out["value"] = float(-z_m if prob["sense"] == "max" else z_m)

    # degeneracy: more active constraints than variables
    active = int(np.sum(np.abs(A_ub @ x - b_ub) < 1e-7)) if len(A_ub) else 0
    active += len(b_eq) + int(np.sum(x < 1e-7))
    out["degenerate"] = active > n

    # alternative optima: can any variable move while staying on the optimal objective?
    A2 = np.vstack([A_eq, cm.reshape(1, -1)]) if len(A_eq) else cm.reshape(1, -1)
    b2 = np.append(b_eq, z_m)
    pts, spread = [x], 0.0
    for j in range(n):
        for sgn in (1.0, -1.0):
            obj = np.zeros(n); obj[j] = sgn
            r = _lp(obj, A_ub, b_ub, A2, b2)
            if r.status == 0:
                pts.append(r.x)
    for p in pts:
        spread = max(spread, float(np.max(np.abs(p - x))))
    if spread > 1e-5:
        out["status"] = "multiple_optima"
        face = []
        for p in pts:
            if not any(np.allclose(p, q, atol=1e-6) for q in face):
                face.append(p)
        out["face"] = [{v: float(p[i]) for i, v in enumerate(vs)} for p in face]
    else:
        out["status"] = "optimal"
        out["face"] = [out["solution"]]
    return out




def cross_check(prob, primary):
    """Compare the primary (exact) result with SciPy.  Returns a small JSON-safe report."""
    try:
        s = scipy_solve(prob)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "agrees": None, "note": f"SciPy cross-check failed: {e}"}
    p_status = primary["status"]
    agrees = p_status == s["status"] or {p_status, s["status"]} <= {"optimal", "multiple_optima"}
    if agrees and primary.get("value") is not None and s.get("value") is not None:
        agrees = abs(float(primary["value"]) - s["value"]) <= 1e-6 * max(1.0, abs(s["value"]))
    return {
        "available": True, "agrees": bool(agrees), "scipy_status": s["status"], "scipy_value": s["value"],
        "note": ("SciPy (HiGHS) reaches the same optimum." if agrees else
                 "SciPy (HiGHS) disagrees with the textbook solver; please report this problem.") +
                " This is a final-answer cross-check only; it does not validate intermediate steps.",
    }
