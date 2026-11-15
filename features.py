"""Turn (problem, solver truth, AI claim) into numeric features, plus the rule-based reference judge."""
from . import solver

LABELS = [
    "correct",
    "wrong_status",
    "wrong_value",
    "suboptimal_solution",
    "infeasible_solution",
    "missed_multiple_optima",
]

FEATURES = [
    "t_optimal", "t_multiple", "t_infeasible", "t_unbounded", "t_degenerate",
    "c_optimal", "c_multiple", "c_infeasible", "c_unbounded", "c_none", "c_unique",
    "status_ok", "unique_vs_multiple",
    "sol_state", "sol_feasible", "sol_viol", "sol_optimal", "sol_gap", "sol_match", "multi_points_ok",
    "val_provided", "val_match", "val_err_abs", "val_err_rel", "val_consistent",
    "n_vars", "n_cons", "is_max",
]


def _cap(x, hi):
    return float(min(abs(x), hi))


def build(prob, truth, claim):
    vs = prob["variables"]
    n = len(vs)
    st = truth["status"]
    cs = claim["status"]
    f = {k: 0.0 for k in FEATURES}
    f["t_optimal"] = float(st == "optimal")
    f["t_multiple"] = float(st == "multiple_optima")
    f["t_infeasible"] = float(st == "infeasible")
    f["t_unbounded"] = float(st == "unbounded")
    f["t_degenerate"] = float(truth.get("degenerate", False))
    f["c_optimal"] = float(cs == "optimal")
    f["c_multiple"] = float(cs == "multiple")
    f["c_infeasible"] = float(cs == "infeasible")
    f["c_unbounded"] = float(cs == "unbounded")
    f["c_none"] = float(cs is None)
    f["c_unique"] = float(bool(claim["unique"]))
    f["n_vars"], f["n_cons"] = float(n), float(len(prob["constraints"]))
    f["is_max"] = float(prob["sense"] == "max")

    ok = (
        (st == "optimal" and cs == "optimal")
        or (st == "multiple_optima" and cs in ("optimal", "multiple") and not claim["unique"])
        or (st == "infeasible" and cs == "infeasible")
        or (st == "unbounded" and cs == "unbounded")
    )
    f["status_ok"] = float(ok)
    f["unique_vs_multiple"] = float(st == "multiple_optima" and cs == "optimal" and claim["unique"])

    details = {"check": None, "multi_checks": [], "value_obj_at_point": None}
    point = claim["point"]
    prov = [v for v in vs if v in point]
    complete = len(prov) == n and n > 0
    ptol = max([p.tol for p in point.values()], default=0.0) if point else 0.0
    zstar = truth.get("value")
    csum = sum(abs(k) for k in prob["objective"].values())

    f["sol_state"] = 0.0 if not prov else (2.0 if complete else 1.0)
    f["sol_feasible"] = 1.0
    f["sol_optimal"] = 1.0
    f["sol_match"] = 1.0
    f["multi_points_ok"] = 1.0
    if complete:
        pt = {v: point[v].v for v in vs}
        chk = solver.check_point(prob, pt, ptol)
        details["check"] = chk
        f["sol_feasible"] = float(chk["feasible"])
        f["sol_viol"] = _cap(chk["max_violation"], 10)
        if zstar is not None:
            gap = abs(chk["objective"] - zstar)
            f["sol_gap"] = _cap(gap / (1 + abs(zstar)), 10)
            f["sol_optimal"] = float(chk["feasible"] and gap <= 1e-6 + ptol * csum)
        elif st in ("infeasible", "unbounded"):
            f["sol_optimal"] = 0.0
        # does the point coincide with a point of the true optimal face?
        face = truth.get("face") or []
        if face:
            f["sol_match"] = float(
                f["sol_optimal"] == 1.0 and (st == "multiple_optima" or any(
                    all(abs(pt[v] - fp[v]) <= ptol + 1e-6 for v in vs) for fp in face))
            )
        else:
            f["sol_match"] = 0.0
    elif prov:
        # partial solution: compare given coordinates with the unique optimum if there is one
        if st == "optimal" and truth.get("solution"):
            good = all(abs(point[v].v - truth["solution"][v]) <= point[v].tol + 1e-6 for v in prov)
            f["sol_optimal"] = float(good)
            f["sol_match"] = float(good)
        elif st in ("infeasible", "unbounded"):
            f["sol_optimal"] = 0.0
            f["sol_match"] = 0.0

    if claim["status"] == "multiple" and claim["multi_points"] and st == "multiple_optima":
        allok = True
        for mp in claim["multi_points"]:
            if all(v in mp for v in vs):
                c2 = solver.check_point(prob, {v: mp[v].v for v in vs}, max(p.tol for p in mp.values()))
                details["multi_checks"].append(c2)
                tol = 1e-6 + max(p.tol for p in mp.values()) * csum
                allok &= c2["feasible"] and abs(c2["objective"] - zstar) <= tol
        f["multi_points_ok"] = float(allok)
        if not allok:
            f["sol_optimal"] = 0.0

    val = claim["value"]
    if val is not None:
        f["val_provided"] = 1.0
        if zstar is not None:
            diff = abs(val.v - zstar)
            f["val_match"] = float(diff <= val.tol + 1e-9 * abs(zstar))
            f["val_err_abs"] = _cap(diff, 1000)
            f["val_err_rel"] = _cap(diff / (1 + abs(zstar)), 10)
        if complete and details["check"]:
            objp = details["check"]["objective"]
            details["value_obj_at_point"] = objp
            f["val_consistent"] = float(abs(val.v - objp) <= val.tol + ptol * csum + 1e-6)
        else:
            f["val_consistent"] = 1.0
    else:
        f["val_consistent"] = 1.0
    return {k: float(f[k]) for k in FEATURES}, details


def judge(f):
    """Deterministic reference verdict from features (used to label training data and to cross-check the ML model)."""
    if f["c_none"]:
        return "unclear"
    if not f["status_ok"]:
        return "missed_multiple_optima" if f["unique_vs_multiple"] else "wrong_status"
    if f["t_infeasible"] or f["t_unbounded"]:
        return "correct"
    if f["sol_state"] > 0 and not f["sol_feasible"]:
        return "infeasible_solution"
    if f["sol_state"] > 0 and not f["sol_optimal"]:
        return "suboptimal_solution"
    if f["val_provided"] and not f["val_match"]:
        return "wrong_value"
    if f["sol_state"] == 0 and not f["val_provided"]:
        return "correct" if (f["c_multiple"] and f["t_multiple"]) else "unclear"
    return "correct"
