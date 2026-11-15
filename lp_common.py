"""Shared result building for the tableau-based LP methods (Simplex, Big M, Two-Phase)."""
from __future__ import annotations

from fractions import Fraction

from .exact import fmt_q, json_num
from .tableau import alternative_optima, snap_json


def describe_iteration(s, decision):
    """One paragraph of textbook-style explanation for an iteration snapshot."""
    cols = s["columns"]
    d = s["delta"]
    k = s["iteration"]
    ph = f"{s['phase']} - " if s.get("phase") else ""
    head = f"{ph}Tableau {k}" + (" (initial)" if k == 0 else "")
    bfs = ", ".join(f"{bn} = {fmt_q(s['b'][i])}" for i, bn in enumerate(s["basis"]))
    parts = [f"{head}: basic solution {bfs}; Z = {fmt_q(s['z'])}."]
    if s["conclusion"] == "optimal":
        parts.append(s["notes"][-1])
        return " ".join(parts)
    e = s["entering"]
    ej = cols.index(e)
    rule = "largest positive" if s["sense"] == "max" else "most negative"
    parts.append(f"Cj - Zj values: " + ", ".join(f"{cols[j]}: {fmt_q(d[j])}" for j in range(len(cols))
                                                  if s["basis_idx"].count(j) == 0 and j not in s.get("excluded", [])) + ".")
    parts.append(f"{e} has the {rule} Cj - Zj ({fmt_q(d[ej])}), so {e} enters the basis.")
    if s["conclusion"] == "unbounded":
        parts.append(s["notes"][-1])
        return " ".join(parts)
    rtxt = []
    for i, r in enumerate(s["ratios"]):
        a = s["A"][i][ej]
        rtxt.append(f"{s['basis'][i]}: {fmt_q(s['b'][i])}/{fmt_q(a)} = {fmt_q(r)}" if r is not None
                    else f"{s['basis'][i]}: - (entry {fmt_q(a)} is not positive)")
    parts.append("Minimum-ratio test (RHS / pivot-column entry, positive entries only): " + "; ".join(rtxt) + ".")
    parts.append(f"The minimum ratio is in the row of {s['leaving']}, so {s['leaving']} leaves. "
                 f"Pivot element = {fmt_q(s['pivot']['value'])} (row {s['pivot']['row'] + 1}, column {e}).")
    parts.append("Row operations: " + "; ".join(o["text"] for o in s["row_ops"]) + ".")
    for n in s["notes"]:
        if "Degenerate" in n or "Tie" in n or "Bland" in n:
            parts.append(n)
    return " ".join(parts)


def finish(method, sf, phases, final_tab, status, decision, extra_explanation=()):
    """Build the result dict common to all tableau methods."""
    sense = sf["sense"]
    res = {
        "method": method, "variables": decision, "sense": sense,
        "objective": f"{'Maximize' if sense == 'max' else 'Minimize'} Z = " + _lin(sf["c"], decision),
        "standard_form": {"steps": sf["steps"], "equations": sf["equations"], "columns": sf["columns"], "kinds": sf["kinds"]},
        "phases": phases, "status": status, "solution": None, "value": None, "all_values": None,
        "degenerate": any(s.get("degenerate") for p in phases for s in p["iterations"]),
        "alternative": None, "optimal_points": [], "optimal_rays": [], "explanation": list(extra_explanation),
    }
    for p in phases:
        for s in p["iterations"]:
            res["explanation"].append(describe_iteration(s, decision))
    if status in ("optimal", "multiple_optima") and final_tab is not None:
        sol = final_tab.solution(decision)
        res["solution"] = sol
        allv = {c: Fraction(0) for c in final_tab.cols}
        for i, j in enumerate(final_tab.basis):
            allv[final_tab.cols[j]] = final_tab.b[i]
        res["all_values"] = allv
        z = sum(sf["c"][v] * sol[v] for v in decision)
        res["value"] = z
        pts, rays, alt = alternative_optima(final_tab, decision)
        res["optimal_points"] = pts
        res["optimal_rays"] = rays
        if len(pts) > 1 or rays:
            res["status"] = "multiple_optima"
            res["alternative"] = alt
            zero = [final_tab.cols[j] for j in range(len(final_tab.cols))
                    if j not in final_tab.basis and j not in final_tab.excluded and final_tab.kinds[j] != "artificial"
                    and final_tab.delta()[j] == 0]
            msg = (f"In the optimal tableau the non-basic variable(s) {', '.join(zero)} have Cj - Zj = 0, so bringing one of them "
                   "into the basis gives another optimal solution with the same Z: the problem has multiple (alternative) optima.")
            if alt is not None:
                msg += " Alternative optimal solution: " + ", ".join(
                    f"{v} = {fmt_q(alt_val)}" for v, alt_val in zip(decision, pts[1])) + "."
            if rays:
                msg += " Z also stays optimal along an unbounded edge."
            res["explanation"].append(msg)
        res["explanation"].append(
            "Optimal solution: " + ", ".join(f"{v} = {fmt_q(sol[v])}" for v in decision) +
            f"; {'maximum' if sense == 'max' else 'minimum'} Z = {fmt_q(z)}.")
        if res["degenerate"]:
            res["explanation"].append("Degeneracy occurred: at least one tableau had a basic variable equal to 0.")
    elif status == "unbounded":
        res["explanation"].append("Conclusion: the problem has an unbounded solution (no finite optimum).")
    elif status == "infeasible":
        res["explanation"].append("Conclusion: the problem has no feasible solution (infeasible).")
    return res


def _lin(c, vs):
    from .lp_parser import fmt_linear
    return fmt_linear(c, vs)


def to_json(res):
    vs = res["variables"]
    out = {
        "method": res["method"], "variables": vs, "sense": res["sense"], "objective": res["objective"],
        "status": res["status"], "degenerate": res["degenerate"], "standard_form": res["standard_form"],
        "phases": [{"name": p["name"], "status": p["status"], "note": p.get("note", ""),
                    "iterations": [snap_json(s) for s in p["iterations"]]} for p in res["phases"]],
        "solution": None if res["solution"] is None else {v: json_num(res["solution"][v]) for v in vs},
        "all_values": None if res["all_values"] is None else {k: json_num(v) for k, v in res["all_values"].items()},
        "value": json_num(res["value"]),
        "alternative": snap_json(res["alternative"]) if res.get("alternative") else None,
        "optimal_points": [[json_num(x) for x in p] for p in res["optimal_points"]],
        "optimal_rays": [[json_num(x) for x in r] for r in res["optimal_rays"]],
        "explanation": res["explanation"],
    }
    return out
