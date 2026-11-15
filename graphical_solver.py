"""Graphical (corner-point) method for two-variable linear programs.

Implements the textbook procedure step by step with exact Fractions:

1. Replace every inequality by its boundary equation and find its intercepts.
2. Intersect every pair of boundary lines (including the axes x = 0 and y = 0).
3. Test every intersection point against all constraints and non-negativity;
   the points that pass are the corner points of the feasible region.
4. Evaluate Z at every corner point.
5. Decide the outcome:
   * no corner point             -> infeasible (with x, y >= 0 a non-empty region always has a corner)
   * an improving recession ray  -> unbounded objective
   * otherwise the best corner(s) -> optimal; several best corners, or a best
     corner with a zero-change ray, -> multiple optimal solutions.

Every intermediate result is returned so it can be shown to the user and
compared with an AI's working.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from .exact import fmt_q, json_num
from .lp_parser import exact_view, fmt_linear


class GraphicalError(ValueError):
    pass


def _pt(p):
    return f"({fmt_q(p[0])}, {fmt_q(p[1])})"


def _eq_text(a1, a2, b, vs):
    return f"{fmt_linear({vs[0]: a1, vs[1]: a2}, vs)} = {fmt_q(b)}"


def _satisfies(row, p, vs):
    lhs = row["coefs"][vs[0]] * p[0] + row["coefs"][vs[1]] * p[1]
    if row["op"] == "<=":
        return lhs <= row["rhs"], lhs
    if row["op"] == ">=":
        return lhs >= row["rhs"], lhs
    return lhs == row["rhs"], lhs


def solve(prob):
    ev = exact_view(prob)
    vs = ev["variables"]
    if len(vs) != 2:
        raise GraphicalError(
            f"The graphical method needs exactly two decision variables; this problem has {len(vs)} ({', '.join(vs)}). "
            "Use the Simplex, Big M or Two-Phase method instead.")
    c = ev["c"]
    sense = ev["sense"]
    rows = ev["rows"]
    explanation = []

    # ---------------------------------------------------------------- 1. boundary lines + intercepts
    lines = []  # every line: a1*x + a2*y = b
    for i, r in enumerate(rows):
        a1, a2, b = r["coefs"][vs[0]], r["coefs"][vs[1]], r["rhs"]
        ix = (b / a1, Fraction(0)) if a1 != 0 else None
        iy = (Fraction(0), b / a2) if a2 != 0 else None
        if a1 == 0:
            note = f"horizontal line {vs[1]} = {fmt_q(b / a2)} (parallel to the {vs[0]}-axis)"
        elif a2 == 0:
            note = f"vertical line {vs[0]} = {fmt_q(b / a1)} (parallel to the {vs[1]}-axis)"
        elif b == 0:
            note = "passes through the origin"
        else:
            note = f"meets the axes at {_pt(ix)} and {_pt(iy)}"
        lines.append({
            "id": f"C{i + 1}", "kind": "constraint", "index": i, "a": (a1, a2), "b": b, "op": r["op"],
            "constraint": r["text"], "equation": _eq_text(a1, a2, b, vs),
            "x_intercept": ix, "y_intercept": iy, "note": note,
        })
    axes = [
        {"id": f"{vs[0]}=0", "kind": "axis", "index": None, "a": (Fraction(1), Fraction(0)), "b": Fraction(0),
         "op": ">=", "constraint": f"{vs[0]} >= 0", "equation": f"{vs[0]} = 0", "note": f"the {vs[1]}-axis"},
        {"id": f"{vs[1]}=0", "kind": "axis", "index": None, "a": (Fraction(0), Fraction(1)), "b": Fraction(0),
         "op": ">=", "constraint": f"{vs[1]} >= 0", "equation": f"{vs[1]} = 0", "note": f"the {vs[0]}-axis"},
    ]
    all_lines = lines + axes
    explanation.append("Step 1 - Boundary lines: replace each inequality by an equation. " + "; ".join(
        f"{l['id']}: {l['equation']} ({l['note']})" for l in lines) + ".")

    nonneg_rows = [{"coefs": {vs[0]: Fraction(1), vs[1]: Fraction(0)}, "op": ">=", "rhs": Fraction(0), "text": f"{vs[0]} >= 0"},
                   {"coefs": {vs[0]: Fraction(0), vs[1]: Fraction(1)}, "op": ">=", "rhs": Fraction(0), "text": f"{vs[1]} >= 0"}]
    all_rows = rows + nonneg_rows

    # ---------------------------------------------------------------- 2./3. intersections + feasibility
    intersections = []
    for L1, L2 in combinations(all_lines, 2):
        (a1, a2), b1 = L1["a"], L1["b"]
        (c1, c2), b2 = L2["a"], L2["b"]
        D = a1 * c2 - a2 * c1
        item = {"lines": [L1["id"], L2["id"]], "equations": [L1["equation"], L2["equation"]]}
        if D == 0:
            same = (a1 * b2 == c1 * b1) and (a2 * b2 == c2 * b1)
            item.update(point=None, parallel=True, coincident=same, feasible=False, violated=[],
                        calc=f"{L1['equation']} and {L2['equation']} are {'the same line' if same else 'parallel'}: no single intersection point.")
            intersections.append(item)
            continue
        Dx = b1 * c2 - a2 * b2
        Dy = a1 * b2 - b1 * c1
        p = (Dx / D, Dy / D)
        violated, checks = [], []
        for r in all_rows:
            ok, lhs = _satisfies(r, p, vs)
            checks.append({"constraint": r["text"], "lhs": lhs, "ok": ok})
            if not ok:
                violated.append(r["text"])
        calc = (f"Solve {L1['equation']} and {L2['equation']}: D = ({fmt_q(a1)})({fmt_q(c2)}) - ({fmt_q(a2)})({fmt_q(c1)}) = {fmt_q(D)}, "
                f"{vs[0]} = {fmt_q(Dx)}/{fmt_q(D)} = {fmt_q(p[0])}, {vs[1]} = {fmt_q(Dy)}/{fmt_q(D)} = {fmt_q(p[1])}")
        item.update(point=p, parallel=False, coincident=False, feasible=not violated, violated=violated, calc=calc, checks=checks)
        intersections.append(item)

    # corner points: distinct feasible intersection points
    corners = []
    for it in intersections:
        if it["point"] is None or not it["feasible"]:
            continue
        p = it["point"]
        found = next((cn for cn in corners if cn["point"] == p), None)
        if found:
            for lid in it["lines"]:
                if lid not in found["on_lines"]:
                    found["on_lines"].append(lid)
            continue
        corners.append({"point": p, "on_lines": list(it["lines"])})
    for cn in corners:
        p = cn["point"]
        # every line that passes through the point (not only the pair that produced it)
        cn["on_lines"] = [l["id"] for l in all_lines if l["a"][0] * p[0] + l["a"][1] * p[1] == l["b"]]
        cn["z"] = c[vs[0]] * p[0] + c[vs[1]] * p[1]
        cn["z_calc"] = f"Z = {fmt_q(c[vs[0]])}({fmt_q(p[0])}) + {fmt_q(c[vs[1]])}({fmt_q(p[1])}) = {fmt_q(cn['z'])}"
        cn["degenerate"] = len(cn["on_lines"]) > 2
    _order_corners(corners)
    for k, cn in enumerate(corners):
        cn["label"] = chr(ord("A") + k) if k < 26 else f"P{k + 1}"

    n_inf = sum(1 for it in intersections if it["point"] is not None and not it["feasible"])
    explanation.append(
        f"Step 2 - Intersections: {len([i for i in intersections if i['point'] is not None])} pairwise intersection points were found "
        f"(including the axes); {n_inf} of them violate at least one constraint and are discarded.")

    # ---------------------------------------------------------------- recession directions (unbounded region?)
    rays = _recession_rays(all_rows, vs)

    out = {
        "method": "graphical", "variables": vs, "sense": sense,
        "objective": f"{'Maximize' if sense == 'max' else 'Minimize'} Z = {fmt_linear(c, vs)}",
        "boundary_lines": lines, "intersections": intersections, "corners": corners,
        "recession_rays": rays, "region": None, "status": None, "optimum": None,
        "optimal_corners": [], "optimal_ray": None, "degenerate": any(cn["degenerate"] for cn in corners),
        "explanation": explanation,
    }

    if prob.get("trivially_infeasible") or not corners:
        out["region"] = "empty"
        out["status"] = "infeasible"
        explanation.append("Step 3 - Feasible region: no intersection point satisfies every constraint, so the feasible region is empty. "
                           "The problem is infeasible.")
        return out

    out["region"] = "unbounded" if rays else "bounded"
    explanation.append("Step 3 - Corner points of the feasible region: " + ", ".join(
        f"{cn['label']}{_pt(cn['point'])}" for cn in corners) + "." +
        (" The region is unbounded (it extends without limit)." if rays else " The region is bounded."))
    explanation.append("Step 4 - Objective at each corner: " + "; ".join(
        f"{cn['label']}: {cn['z_calc']}" for cn in corners) + ".")

    better = (lambda d: d > 0) if sense == "max" else (lambda d: d < 0)
    improving = [r for r in rays if better(c[vs[0]] * r[0] + c[vs[1]] * r[1])]
    if improving:
        r = improving[0]
        out["status"] = "unbounded"
        out["improving_ray"] = r
        explanation.append(
            f"Step 5 - The feasible region is unbounded in the direction {_pt(r)}, and Z changes by "
            f"{fmt_q(c[vs[0]] * r[0] + c[vs[1]] * r[1])} per unit along it, so Z can be made "
            f"{'arbitrarily large' if sense == 'max' else 'arbitrarily small'}. The objective is unbounded: no finite optimum.")
        return out

    zs = [cn["z"] for cn in corners]
    zbest = max(zs) if sense == "max" else min(zs)
    best = [cn for cn in corners if cn["z"] == zbest]
    flat_rays = [r for r in rays if c[vs[0]] * r[0] + c[vs[1]] * r[1] == 0]
    out["optimum"] = {"point": best[0]["point"], "value": zbest}
    out["optimal_corners"] = [cn["point"] for cn in best]
    if len(best) > 1:
        out["status"] = "multiple_optima"
        explanation.append(
            f"Step 5 - The {'largest' if sense == 'max' else 'smallest'} value Z = {fmt_q(zbest)} occurs at "
            f"{len(best)} corners ({', '.join(cn['label'] + _pt(cn['point']) for cn in best)}). The objective line is parallel "
            "to a binding constraint, so every point on the edge joining them is optimal: multiple optimal solutions.")
    elif flat_rays:
        out["status"] = "multiple_optima"
        out["optimal_ray"] = flat_rays[0]
        explanation.append(
            f"Step 5 - The best corner is {best[0]['label']}{_pt(best[0]['point'])} with Z = {fmt_q(zbest)}, and Z does not change "
            f"along the unbounded edge in direction {_pt(flat_rays[0])}, so infinitely many points (a ray) are optimal.")
    else:
        out["status"] = "optimal"
        extra = " This corner is degenerate (more than two boundary lines pass through it)." if best[0]["degenerate"] else ""
        explanation.append(
            f"Step 5 - The {'maximum' if sense == 'max' else 'minimum'} is Z = {fmt_q(zbest)} at "
            f"{best[0]['label']}{_pt(best[0]['point'])}, i.e. {vs[0]} = {fmt_q(best[0]['point'][0])}, "
            f"{vs[1]} = {fmt_q(best[0]['point'][1])}.{extra}")
    if rays and out["status"] != "unbounded":
        explanation.append("Note: the feasible region is unbounded, but the objective still has a finite optimum because it "
                           "does not improve in any unbounded direction.")
    return out


def _order_corners(corners):
    """Order corners around the region (counter-clockwise), starting nearest the origin, like a textbook figure."""
    import math
    if len(corners) < 3:
        corners.sort(key=lambda cn: (cn["point"][0] + cn["point"][1], cn["point"][0]))
        return
    cx = sum(float(cn["point"][0]) for cn in corners) / len(corners)
    cy = sum(float(cn["point"][1]) for cn in corners) / len(corners)
    corners.sort(key=lambda cn: math.atan2(float(cn["point"][1]) - cy, float(cn["point"][0]) - cx))
    start = min(range(len(corners)), key=lambda k: (corners[k]["point"][0] + corners[k]["point"][1], corners[k]["point"][1]))
    corners[:] = corners[start:] + corners[:start]


def _recession_rays(all_rows, vs):
    """Extreme rays of the recession cone {d : constraint directions hold, d >= 0}.

    In two dimensions the extreme rays lie along the axes or along a constraint's
    boundary direction, so testing those candidates is exact."""
    cands = [(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
    for r in all_rows:
        a1, a2 = r["coefs"][vs[0]], r["coefs"][vs[1]]
        if a1 == 0 and a2 == 0:
            continue
        cands += [(a2, -a1), (-a2, a1)]
    out = []
    for d in cands:
        if d == (0, 0) or d[0] < 0 or d[1] < 0:
            continue
        ok = True
        for r in all_rows:
            ad = r["coefs"][vs[0]] * d[0] + r["coefs"][vs[1]] * d[1]
            if (r["op"] == "<=" and ad > 0) or (r["op"] == ">=" and ad < 0) or (r["op"] == "=" and ad != 0):
                ok = False
                break
        if ok:
            g = max(d)
            dn = (d[0] / g, d[1] / g)
            if dn not in out:
                out.append(dn)
    return out


# ----------------------------------------------------------------------------- serialisation
def _jp(p):
    return None if p is None else [json_num(p[0]), json_num(p[1])]


def to_json(res):
    """JSON-safe copy (exact numbers become {"v": float, "s": "3/2"})."""
    vs = res["variables"]
    return {
        "method": "graphical", "variables": vs, "sense": res["sense"], "objective": res["objective"],
        "status": res["status"], "region": res["region"], "degenerate": res["degenerate"],
        "boundary_lines": [{
            "id": l["id"], "constraint": l["constraint"], "equation": l["equation"], "note": l["note"],
            "x_intercept": _jp(l["x_intercept"]), "y_intercept": _jp(l["y_intercept"])} for l in res["boundary_lines"]],
        "intersections": [{
            "lines": it["lines"], "equations": it["equations"], "point": _jp(it["point"]), "parallel": it["parallel"],
            "feasible": it["feasible"], "violated": it["violated"], "calc": it["calc"]} for it in res["intersections"]],
        "corners": [{
            "label": cn["label"], "point": _jp(cn["point"]), "z": json_num(cn["z"]), "z_calc": cn["z_calc"],
            "on_lines": cn["on_lines"], "degenerate": cn["degenerate"]} for cn in res["corners"]],
        "recession_rays": [_jp(r) for r in res["recession_rays"]],
        "optimum": None if not res["optimum"] else {"point": _jp(res["optimum"]["point"]), "value": json_num(res["optimum"]["value"])},
        "optimal_corners": [_jp(p) for p in res["optimal_corners"]],
        "optimal_ray": _jp(res.get("optimal_ray")),
        "explanation": res["explanation"],
    }
