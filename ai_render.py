"""Render simplex working as an AI would write it, optionally with one injected mistake.

Used to build test cases and the step-level dataset.  The mistake is injected
*during* the simulation and the simplex method then continues from the wrong
tableau, so errors propagate to later tableaus exactly as they would in a real
answer (this is what lets us test downstream-consequence reporting).

Mistake types
-------------
entering_not_best   valid improving variable that is not the largest Cj - Zj (a deviation, not an error)
entering_bad        a variable whose Cj - Zj does not improve Z
leaving_bad         a row that does not have the minimum ratio
ratio_text          one ratio written wrongly in the Ratio column
pivot_claim         the stated pivot element is wrong
normalise_rhs       pivot-row RHS not divided by the pivot element
rowop_cell          one entry of a non-pivot row computed wrongly (then carried forward)
zj_cell             one Zj / Cj - Zj entry written wrongly (display only)
premature_stop      stop and declare optimality while an improving variable exists
wrong_value         final Z stated wrongly
unique_claim        alternative optima exist but the answer calls the optimum unique
"""
from __future__ import annotations

import random
from copy import deepcopy
from fractions import Fraction

from .exact import fmt_q, sign
from .tableau import Tableau, standard_form


def _fmt(x, rng, decimals=False):
    s = fmt_q(x)
    if decimals and "/" in s and rng.random() < 0.6:
        return f"{float(x):.2f}"
    return s


def simulate(prob, mistake=None, at_iter=0, seed=0, max_iters=12):
    """Run the simplex method (slack basis) with an optional mistake.  Returns (records, final_status, notes)."""
    rng = random.Random(seed)
    sf = standard_form(prob, "simplex")
    vs = sf["variables"]
    cost = [sf["c"][v] for v in vs] + [Fraction(0)] * (len(sf["columns"]) - len(vs))
    t = Tableau(sf["columns"], sf["kinds"], cost, sf["A"], sf["b"], sf["basis"], sf["sense"])
    recs = []
    applied = False
    status = None
    for k in range(max_iters):
        snap = t.snapshot()
        rec = {"snap": snap, "entering": None, "leaving": None, "pivot": None, "ratios": None, "display": {}, "notes": []}
        recs.append(rec)
        cands = t.candidates()
        if mistake == "premature_stop" and k == at_iter and cands and not applied:
            applied = True
            status = "claimed_optimal"
            break
        if not cands:
            status = "optimal"
            break
        e, _ = t.choose_entering()
        if k == at_iter and not applied and mistake in ("entering_not_best", "entering_bad"):
            d = t.delta()
            if mistake == "entering_not_best":
                others = [j for j in cands if d[j] != d[e]]
                if others:
                    e = others[0]
                    applied = True
            else:
                bad = [j for j in range(len(t.cols)) if j not in t.basis and not t.improving(d[j])
                       and any(t.A[i][j] > 0 for i in range(len(t.A)))]
                if bad:
                    e = bad[0]
                    applied = True
        leave, rs, _ = t.choose_leaving(e)
        rec["entering"] = t.cols[e]
        rec["ratios"] = rs
        if leave is None:
            status = "unbounded"
            break
        if k == at_iter and not applied and mistake == "leaving_bad":
            others = [i for i in range(len(t.A)) if i != leave and t.A[i][e] != 0 and (rs[i] is None or rs[i] != rs[leave])]
            if others:
                leave = others[0]
                applied = True
        if k == at_iter and not applied and mistake == "ratio_text":
            valid = [i for i, r in enumerate(rs) if r is not None]
            if valid:
                i = valid[-1]
                rec["display"]["ratio"] = (i, rs[i] + rng.choice([1, 2, Fraction(1, 2)]))
                applied = True
        rec["leaving"] = t.cols[t.basis[leave]]
        piv = t.A[leave][e]
        rec["pivot"] = piv
        if k == at_iter and not applied and mistake == "pivot_claim":
            rec["display"]["pivot_claim"] = piv + rng.choice([1, 2, -1]) if piv != 1 else 2
            applied = True
        if k == at_iter and not applied and mistake == "zj_cell":
            j = rng.randrange(len(t.cols))
            rec["display"]["delta_cell"] = (j, snap["delta"][j] + rng.choice([1, -1, 2]))
            applied = True
        t.pivot(leave, e)
        if k == at_iter and not applied and mistake == "normalise_rhs" and piv != 1:
            t.b[leave] = t.b[leave] * piv
            applied = True
        if k == at_iter and not applied and mistake == "rowop_cell":
            rows = [i for i in range(len(t.A)) if i != leave]
            if rows:
                i = rng.choice(rows)
                if rng.random() < 0.5:
                    t.b[i] = t.b[i] + rng.choice([1, 2, -1])
                else:
                    j = rng.choice([jj for jj in range(len(t.cols)) if jj != e])
                    t.A[i][j] = t.A[i][j] + rng.choice([1, -1])
                applied = True
        if any(x < 0 for x in t.b):
            # the wrong pivot produced a negative RHS; a careless answer would still carry on, we stop here
            recs.append({"snap": t.snapshot(), "entering": None, "leaving": None, "pivot": None, "ratios": None,
                         "display": {}, "notes": []})
            status = "claimed_optimal"
            break
    else:
        status = "claimed_optimal"
    return recs, status, applied


def render(prob, recs, status, style="md", slack_naming="s", seed=0, decimals=False, final_value_delta=None,
           unique_claim=False, show_zj=True, show_ratio=True):
    rng = random.Random(seed)
    sf = standard_form(prob, "simplex")
    vs = sf["variables"]
    cols = sf["columns"]
    names = list(cols)
    if slack_naming == "x" and vs and vs[0][-1:].isdigit():
        base = vs[0].rstrip("0123456789")
        names = list(vs) + [f"{base}{len(vs) + k + 1}" for k in range(len(cols) - len(vs))]
    nm = dict(zip(cols, names))
    out = []
    out.append("Convert to standard form by adding slack variables:")
    for eq in sf["equations"]:
        e2 = eq
        for c in cols[len(vs):]:
            e2 = e2.replace(c, nm[c])
        out.append(e2)
    out.append("")
    for k, rec in enumerate(recs):
        s = rec["snap"]
        title = "Initial simplex tableau" if k == 0 else f"Iteration {k}"
        out.append(f"**{title}**")
        out.append("")
        out += _table(s, rec, names, style, rng, decimals, show_zj, show_ratio)
        out.append("")
        if rec["entering"]:
            e = nm[rec["entering"]]
            if rec["leaving"]:
                pc = rec["display"].get("pivot_claim", rec["pivot"])
                out.append(rng.choice([
                    f"Entering variable: {e}. Leaving variable: {nm[rec['leaving']]}. Pivot element = {_fmt(pc, rng)}.",
                    f"{e} enters the basis and {nm[rec['leaving']]} leaves (minimum ratio). The pivot element is {_fmt(pc, rng)}.",
                ]))
                out.append("")
            else:
                out.append(f"{e} can enter, but no entry in its column is positive, so the solution is unbounded.")
    if status == "unbounded":
        out.append("Therefore the problem has an unbounded solution.")
        return "\n".join(out)
    last = recs[-1]["snap"]
    sol = {v: Fraction(0) for v in vs}
    for i, bn in enumerate(last["basis"]):
        if bn in sol:
            sol[bn] = last["b"][i]
    z = sum(sf["c"][v] * sol[v] for v in vs)
    if final_value_delta:
        z = z + final_value_delta
    out.append("All Cj - Zj values satisfy the optimality condition, so the current solution is optimal.")
    pt = ", ".join(f"{v} = {_fmt(sol[v], rng, decimals)}" for v in vs)
    tail = " The optimal solution is unique." if unique_claim else ""
    out.append(f"**Final answer:** {pt}, {'maximum' if sf['sense'] == 'max' else 'minimum'} Z = {_fmt(z, rng, decimals)}.{tail}")
    return "\n".join(out)


def _table(s, rec, names, style, rng, decimals, show_zj, show_ratio):
    ratios = rec.get("ratios") if show_ratio and rec.get("entering") else None
    head = ["Basis", "CB"] + names + ["RHS"] + (["Ratio"] if ratios is not None else [])
    rows = []
    for i, bn in enumerate(s["basis"]):
        r = [names[s["columns"].index(bn)], fmt_q(s["cb"][i])] + [_fmt(x, rng, decimals) for x in s["A"][i]] + [
            _fmt(s["b"][i], rng, decimals)]
        if ratios is not None:
            rv = ratios[i]
            if rec["display"].get("ratio") and rec["display"]["ratio"][0] == i:
                rv = rec["display"]["ratio"][1]
            r.append("-" if rv is None else _fmt(rv, rng, decimals))
        rows.append(r)
    delta = list(s["delta"])
    if rec["display"].get("delta_cell"):
        j, v = rec["display"]["delta_cell"]
        delta[j] = v
    pad = [""] if ratios is not None else []
    if show_zj:
        rows.append(["Zj", ""] + [fmt_q(x) for x in s["zj"]] + [fmt_q(s["z"])] + pad)
    rows.append(["Cj - Zj", ""] + [fmt_q(x) for x in delta] + [""] + pad)
    if style == "latex":
        lines = ["$$", "\\begin{array}{" + "c" * len(head) + "}", " & ".join(f"\\text{{{h}}}" if h.isalpha() and len(h) > 2 else h
                                                                      for h in head) + " \\\\ \\hline"]
        lines += [" & ".join(r) + " \\\\" for r in rows]
        lines += ["\\end{array}", "$$"]
        return lines
    if style == "plain":
        w = max(len(x) for r in [head] + rows for x in r) + 2
        return ["".join(x.ljust(w) for x in head).rstrip()] + ["".join(x.ljust(w) for x in r).rstrip() for r in rows]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def make_answer(prob, mistake=None, at_iter=0, seed=0, style="md", slack_naming="s", decimals=False):
    recs, status, applied = simulate(prob, mistake, at_iter, seed)
    fvd = None
    if mistake == "wrong_value" and status != "unbounded":
        fvd = random.Random(seed).choice([1, 2, -1, -2, 5])
        applied = True
    text = render(prob, recs, status, style=style, slack_naming=slack_naming, seed=seed, decimals=decimals,
                  final_value_delta=fvd, unique_claim=(mistake == "unique_claim"))
    if mistake == "unique_claim":
        applied = True
    return text, applied


# ----------------------------------------------------------------------------- graphical answers
GRAPHICAL_MISTAKES = ["intercept", "intersection", "infeasible_corner", "evaluation", "wrong_pick", "missed_corner"]


def make_graphical_answer(prob, mistake=None, seed=0):
    """Corner-point working for a 2-variable LP, optionally with one mistake.  Returns (text, applied)."""
    from .graphical_solver import solve as gsolve
    rng = random.Random(seed)
    g = gsolve(prob)
    vs = g["variables"]
    c = {v: prob["objective_q"][v] if "objective_q" in prob else Fraction(prob["objective"][v]) for v in vs}
    lines = []
    applied = False
    for l in g["boundary_lines"]:
        ix, iy = l["x_intercept"], l["y_intercept"]
        if ix and iy and ix != iy:
            a, b = ix, iy
            if mistake == "intercept" and not applied:
                b = (b[0], b[1] + 1)
                applied = True
            lines.append(f"{l['equation']} meets the axes at ({fmt_q(a[0])}, {fmt_q(a[1])}) and ({fmt_q(b[0])}, {fmt_q(b[1])}).")
        else:
            lines.append(f"Boundary line: {l['equation']}.")
    corners = [list(cn["point"]) for cn in g["corners"]]
    inter_lines = []
    for it in g["intersections"]:
        if it["point"] is None or not it["feasible"] or "=0" in it["lines"][0] or "=0" in it["lines"][1]:
            continue
        p = list(it["point"])
        if mistake == "intersection" and not applied:
            wrong = [p[0], p[1] + 1]
            corners = [wrong if q == p else q for q in corners]
            p = wrong
            applied = True
        inter_lines.append(f"Intersection of {it['equations'][0]} and {it['equations'][1]}: solving gives ({fmt_q(p[0])}, {fmt_q(p[1])}).")
    if mistake == "infeasible_corner" and not applied:
        bad = [it["point"] for it in g["intersections"] if it["point"] is not None and not it["feasible"]
               and it["point"][0] >= 0 and it["point"][1] >= 0]
        if bad:
            corners.append(list(bad[0]))
            applied = True
    if mistake == "missed_corner" and not applied and len(corners) > 3:
        opt = list(g["optimum"]["point"]) if g["optimum"] else None
        drop = [q for q in corners if q != opt]
        if drop:
            corners.remove(drop[-1])
            applied = True
    lines += inter_lines
    if not corners:
        lines.append("No point satisfies all the constraints, so the feasible region is empty: the problem is infeasible.")
        return "\n".join(lines), applied
    lines.append("Corner points of the feasible region: " + ", ".join(f"({fmt_q(p[0])}, {fmt_q(p[1])})" for p in corners) + ".")
    evals = []
    for p in corners:
        z = c[vs[0]] * p[0] + c[vs[1]] * p[1]
        if mistake == "evaluation" and not applied and z != 0:
            z = z + rng.choice([1, 2, -1])
            applied = True
        evals.append((p, z))
        lines.append(f"At ({fmt_q(p[0])}, {fmt_q(p[1])}): Z = {fmt_q(c[vs[0]])}({fmt_q(p[0])}) + {fmt_q(c[vs[1]])}({fmt_q(p[1])}) = {fmt_q(z)}.")
    if g["status"] == "unbounded":
        lines.append("However, the region is unbounded and Z keeps increasing along it, so the problem is unbounded.")
        return "\n".join(lines), applied
    best = max(evals, key=lambda e: e[1]) if g["sense"] == "max" else min(evals, key=lambda e: e[1])
    if mistake == "wrong_pick" and not applied and len(evals) > 1:
        best = min(evals, key=lambda e: e[1]) if g["sense"] == "max" else max(evals, key=lambda e: e[1])
        applied = True
    p, z = best
    lines.append(f"Therefore the {'maximum' if g['sense'] == 'max' else 'minimum'} is Z = {fmt_q(z)} at {vs[0]} = {fmt_q(p[0])}, "
                 f"{vs[1]} = {fmt_q(p[1])}.")
    return "\n".join(lines), applied
