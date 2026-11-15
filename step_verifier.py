"""Step-by-step verification of an AI-written solution against MathGuard's own working.

Design principles
-----------------
* The independent solvers (graphical_solver, simplex/bigm/two_phase) are the
  authority for every number.  Nothing here is learned or guessed.
* Every intermediate tableau written by the AI is checked against what *should*
  follow from the AI's own previous tableau and its own pivot choice.  This is
  how downstream consequences are reported: an arithmetic slip in iteration 1 is
  flagged once, and later tableaus that are computed correctly from that wrong
  input are marked "consequential" (amber), not as new independent errors.
* A different but valid path (e.g. choosing an improving entering variable that
  is not the one with the largest Cj - Zj, or converting Min Z to Max -Z) is a
  "deviation" (amber), not an error.
* Anything that cannot be read or checked is "unverifiable" (amber) - never
  silently counted as correct or wrong.

Step statuses: correct | incorrect | consequential | deviation | unverifiable | incomplete
"""
from __future__ import annotations

import re
from copy import deepcopy
from fractions import Fraction

from .answer_parser import parse_answer
from .error_catalog import CATALOG, FINAL_TO_CODE, info
from .exact import MNum, fmt_q, json_num, matches, sign, snap_decimal
from .lp_parser import exact_view, fmt_linear
from .solution_parser import Cell, norm_var, parse_solution

GREEN = {"correct"}
RED = {"incorrect"}
AMBER = {"consequential", "deviation", "unverifiable", "incomplete"}


# ----------------------------------------------------------------------------- report container
class Report:
    def __init__(self):
        self.steps = []
        self.tableaus = []
        self.notes = []

    def add(self, group, title, status, code=None, message="", ai=None, correct=None, iteration=None, row=None,
            column=None, consequence=None):
        st = {"group": group, "title": title, "status": status, "message": message,
              "ai_value": _s(ai), "correct_value": _s(correct), "iteration": iteration, "row": row, "column": column,
              "consequence": consequence}
        if code:
            st.update(code=code, **{k: v for k, v in info(code).items() if k in ("kind", "kind_text")},
                      category=CATALOG.get(code, {}).get("title", code), note=CATALOG.get(code, {}).get("note", ""))
        elif status == "unverifiable":
            st.update(kind="unverifiable", kind_text="Unverifiable step")
        elif status == "incomplete":
            st.update(kind="incomplete", kind_text="Incomplete working")
        self.steps.append(st)
        return st


def _s(x):
    if x is None:
        return None
    if isinstance(x, Cell):
        return x.raw
    if isinstance(x, (Fraction, MNum, int)):
        return fmt_q(x)
    if isinstance(x, tuple):
        return "(" + ", ".join(_s(v) for v in x) + ")"
    return str(x)


def _same(cell, truth):
    if cell is None or not isinstance(cell, Cell) or not cell.ok or truth is None:
        return None
    return matches(cell.value, cell.tol, truth)


def _snap(cell):
    """Exact value for an AI number: rounded decimals snap to the nearest simple fraction (0.333 -> 1/3)."""
    if cell is None or not cell.ok:
        return None
    return snap_decimal(cell.value, cell.tol)


# ----------------------------------------------------------------------------- shared: restated problem
def check_setup(rep, prob, parsed):
    ev = exact_view(prob)
    vs = ev["variables"]
    o = parsed.objective
    if o is not None:
        want = ev["c"]
        diffs = [v for v in vs if Fraction(o["coefs"].get(v, 0)) != want[v]]
        if o["sense"] != ev["sense"]:
            rep.add("Problem setup", "Objective function", "incorrect", "E01",
                    f"The answer {'maximises' if o['sense'] == 'max' else 'minimises'} although the problem asks to "
                    f"{'maximise' if ev['sense'] == 'max' else 'minimise'}.", ai=o["text"],
                    correct=f"{'Maximize' if ev['sense'] == 'max' else 'Minimize'} Z = {fmt_linear(want, vs)}")
        elif diffs:
            rep.add("Problem setup", "Objective function", "incorrect", "E01",
                    "Objective coefficient(s) changed: " + ", ".join(
                        f"{v}: answer uses {fmt_q(o['coefs'].get(v, 0))}, problem has {fmt_q(want[v])}" for v in diffs) + ".",
                    ai=o["text"], correct=f"Z = {fmt_linear(want, vs)}")
        else:
            rep.add("Problem setup", "Objective function", "correct", message=f"Restated correctly: {o['text']}")
    # restated constraints
    rows = ev["rows"]
    seen = set()
    for c in parsed.constraints:
        key = (tuple(sorted((k, str(v)) for k, v in c["coefs"].items())), c["op"], str(c["rhs"].value))
        if key in seen:
            continue
        seen.add(key)
        hit = _match_constraint(c, rows, vs)
        if hit == "exact":
            continue
        if hit is None:
            rep.add("Problem setup", "Restated constraint", "unverifiable",
                    message=f"'{c['text']}' does not correspond to any constraint of the problem (it may be part of other "
                            "working).", ai=c["text"])
        else:
            r, why = hit
            rep.add("Problem setup", "Restated constraint", "incorrect", "E09", f"{why}", ai=c["text"], correct=r["text"])
    if parsed.constraints and not any(s["code"] == "E09" for s in rep.steps if s.get("code")):
        rep.add("Problem setup", "Constraints", "correct",
                message="The constraints restated in the answer match the problem.")


def _match_constraint(c, rows, vs):
    co = {v: Fraction(c["coefs"].get(v, 0)) for v in vs}
    rhs = c["rhs"].value
    best = None
    for r in rows:
        for flip in (1, -1):
            a = {v: flip * r["coefs"][v] for v in vs}
            b = flip * r["rhs"]
            op = r["op"] if flip == 1 else {"<=": ">=", ">=": "<=", "=": "="}[r["op"]]
            if co == a and rhs == b and op == c["op"]:
                return "exact"
            same_coef = co == a
            same_vars = {v for v in vs if co[v]} == {v for v in vs if a[v]}
            if same_coef and op == c["op"] and rhs != b:
                best = best or (r, f"Right-hand side should be {fmt_q(b)}, not {fmt_q(rhs)}.")
            elif same_coef and rhs == b and op != c["op"]:
                best = best or (r, f"The inequality direction is wrong: the problem has '{op}'.")
            elif same_vars and rhs == b and op == c["op"] and sum(1 for v in vs if co[v] != a[v]) == 1:
                v = next(v for v in vs if co[v] != a[v])
                best = best or (r, f"Coefficient of {v} should be {fmt_q(a[v])}, not {fmt_q(co[v])}.")
    return best


def check_standard_form(rep, prob, parsed, std):
    """Compare standard-form equations written by the AI with MathGuard's (slack sign errors = E02)."""
    if not parsed.std_equations or std is None:
        return
    vs = std["variables"]
    cols, kinds, owner, A, b = std["columns"], std["kinds"], std["owner"], std["A"], std["b"]
    bad = 0
    for eq in parsed.std_equations:
        co = eq["coefs"]
        dec = {v: Fraction(co.get(v, 0)) for v in vs}
        # which constraint row has the same decision coefficients and RHS?
        row = next((i for i in range(len(A)) if [A[i][j] for j in range(len(vs))] == [dec[v] for v in vs]
                    and eq["rhs"].value == b[i]), None)
        if row is None:
            continue
        extra = {k: Fraction(v) for k, v in co.items() if k not in vs}
        want = {cols[j]: A[row][j] for j in range(len(vs), len(cols)) if owner[j] == row}
        # compare by kind/sign (names may differ, e.g. x3 instead of s1)
        want_signs = sorted((kinds[cols.index(n)], sign(v)) for n, v in want.items())
        got_signs = sorted(sign(v) for v in extra.values())
        if sorted(s for _, s in want_signs) == got_signs:
            continue
        bad += 1
        exp = fmt_linear({v: dec[v] for v in vs}, vs) + "".join(
            (" + " if v > 0 else " - ") + n for n, v in want.items()) + f" = {fmt_q(b[row])}"
        rep.add("Problem setup", f"Standard form of constraint {row + 1}", "incorrect", "E02",
                "Slack / surplus / artificial variables have the wrong sign or are missing: " +
                ", ".join(f"{kinds[cols.index(n)]} {n} needs coefficient {fmt_q(v)}" for n, v in want.items()) + ".",
                ai=eq["text"], correct=exp)
    if not bad:
        rep.add("Problem setup", "Standard form", "correct",
                message="The standard-form equations (slack / surplus / artificial variables) are correct.")


# ----------------------------------------------------------------------------- column mapping
def map_columns(ai_cols, ref_cols, ref_kinds, decision, ai_first=None, ref_first=None):
    """Map AI column names onto MathGuard's columns.

    1. decision variables and identical names (case-insensitive; R1 is accepted for A1);
    2. index aliases: x3, x4, ... used for slack variables (x(n+k) -> k-th non-decision column);
    3. content: identical initial-tableau column vectors;
    4. order of the remaining columns."""
    m, notes = {}, []
    low = {c.lower(): j for j, c in enumerate(ref_cols)}
    n = len(decision)
    non_dec = [j for j, k in enumerate(ref_kinds) if k != "decision"]
    for c in ai_cols:
        cl = c.lower()
        if cl in low:
            m[c] = low[cl]
        elif re.fullmatch(r"r\d+", cl) and ("a" + cl[1:]) in low:
            m[c] = low["a" + cl[1:]]
    used = set(m.values())
    dec_base = re.match(r"([A-Za-z]+)(\d+)$", decision[0]) if decision else None
    for c in ai_cols:
        if c in m:
            continue
        mm = re.match(r"([A-Za-z]+)(\d+)$", c)
        if dec_base and mm and mm.group(1).lower() == dec_base.group(1).lower():
            k = int(mm.group(2)) - n - 1
            if 0 <= k < len(non_dec) and non_dec[k] not in used:
                m[c] = non_dec[k]
                used.add(non_dec[k])
                notes.append(f"Column {c} is read as {ref_cols[non_dec[k]]} (the answer numbers slack variables after the "
                             "decision variables).")
    if ai_first is not None and ref_first is not None:
        for c in ai_cols:
            if c in m:
                continue
            vec = [r["cells"].get(c) for r in ai_first.rows]
            if any(x is None or not x.ok for x in vec) or len(vec) != len(ref_first["A"]):
                continue
            hits = [j for j in range(len(ref_cols)) if j not in used and all(
                vec[i].value == ref_first["A"][i][j] for i in range(len(vec)))]
            if len(hits) == 1:
                m[c] = hits[0]
                used.add(hits[0])
                notes.append(f"Column {c} is read as {ref_cols[hits[0]]} (same column in the initial tableau).")
    rest_ai = [c for c in ai_cols if c not in m]
    rest_ref = [j for j in range(len(ref_cols)) if j not in used]
    for c, j in zip(rest_ai, rest_ref):
        m[c] = j
        notes.append(f"Column {c} is assumed to be {ref_cols[j]} (by position).")
    for c in rest_ai[len(rest_ref):]:
        notes.append(f"Column {c} has no counterpart in MathGuard's tableau and is ignored.")
    return m, notes


# ----------------------------------------------------------------------------- AI tableau in MathGuard coordinates
class AT:
    """An AI tableau re-expressed in MathGuard's column order."""

    def __init__(self, tab, cmap, ncols):
        self.src = tab
        self.ncols = ncols
        inv = {j: c for c, j in cmap.items()}
        self.inv = inv
        self.basis_names = tab.basis()
        self.basis = [cmap.get(b) if b in cmap else _alias(b, cmap) for b in self.basis_names]
        self.A = [[r["cells"].get(inv[j]) if j in inv else None for j in range(ncols)] for r in tab.rows]
        self.b = [r["rhs"] for r in tab.rows]
        self.cb = [r["cb"] for r in tab.rows]
        self.ratio = [r["ratio"] for r in tab.rows]
        self.ratio_raw = [r.get("ratio_raw", "") for r in tab.rows]
        self.cj = {cmap[c]: v for c, v in tab.cj.items() if c in cmap}
        self.zj = {cmap[c]: v for c, v in tab.zj.items() if c in cmap}
        self.delta = {cmap[c]: v for c, v in tab.delta.items() if c in cmap}
        self.z = tab.z

    def exact(self):
        """Exact matrix / rhs (None where unreadable)."""
        A = [[_snap(c) for c in row] for row in self.A]
        b = [_snap(c) for c in self.b]
        return A, b


def _alias(b, cmap):
    for c, j in cmap.items():
        if c.lower() == b.lower():
            return j
    return None


# ----------------------------------------------------------------------------- tableau-method verification
def verify_tableaus(rep, prob, method, ref, parsed, sense_info):
    """Check every AI tableau and every transition between consecutive tableaus."""
    decision = ref["variables"]
    ai_tabs = parsed.tableaus
    phases = ref["phases"]
    # ---- split AI tableaus into phases
    if len(phases) == 2 and any(t.phase for t in ai_tabs):
        groups = [[t for t in ai_tabs if t.phase != "II"], [t for t in ai_tabs if t.phase == "II"]]
    elif len(phases) == 2:
        groups = _guess_two_phase_split(ai_tabs, phases)
    else:
        groups = [ai_tabs]
    first_err_global = None
    last_info = None
    for ph_idx, (ph, tabs) in enumerate(zip(phases, groups)):
        if not tabs:
            if len(phases) > 1:
                rep.add(f"{ph['name']}", f"{ph['name']} tableaus", "incomplete",
                        message=f"No tableau for {ph['name']} was found in the answer, so this phase cannot be checked.")
            continue
        s0 = ph["iterations"][0]
        cols, kinds = s0["columns"], s0["kinds"]
        exp_cost = list(s0["cost"])
        cmap, notes = map_columns(tabs[0].columns, cols, kinds, decision, tabs[0], s0)
        for n_ in notes:
            rep.notes.append(n_)
        missing = [cols[j] for j in range(len(cols)) if j not in cmap.values()]
        if missing:
            rep.notes.append(f"{ph['name']}: the answer's tableaus have no column for {', '.join(missing)}; "
                             "those entries are treated as unknown.")
        ats = [AT(t, cmap, len(cols)) for t in tabs]
        ref_cols = cols
        inv_map = {j: c for c, j in cmap.items()}
        cols = [inv_map.get(j, c) for j, c in enumerate(ref_cols)]  # show the answer's own column names
        prefix = f"{ph['name']}: " if len(phases) > 1 else ""
        # ---- convention: Min Z handled as Max (-Z)?  or Zj - Cj written as Cj - Zj?
        flip = _detect_flip(ats[0], s0, exp_cost)
        if flip:
            rep.add(f"{prefix}Initial tableau", "Sign convention", "deviation",
                    message="The answer works with the opposite sign convention (Max -Z instead of Min Z, or Zj - Cj instead "
                            "of Cj - Zj). This is a valid alternative; all signs are compared accordingly.")
        costs = [-c for c in exp_cost] if flip == "objective" else exp_cost
        dsign = -1 if flip == "delta" else 1
        tsense = s0["sense"]
        if flip == "objective":
            tsense = "max" if tsense == "min" else "min"
        # ---- the first tableau of the phase
        prev_err = first_err_global
        if ph_idx == 0:
            expected0 = {"A": s0["A"], "b": s0["b"], "basis": s0["basis_idx"]}
            err0 = _compare_tableau(rep, f"{prefix}Initial tableau", ats[0], expected0, cols, kinds, costs, dsign, tsense,
                                    kind="initial", upstream=None)
        else:
            # Phase II starts from the last Phase I tableau (artificial columns removed, original costs restored)
            if last_info is not None:
                expA, expb, expbasis = _phase2_start(last_info, ref_cols)
                expected0 = {"A": expA, "b": expb, "basis": expbasis}
                err0 = _compare_tableau(rep, f"{prefix}Initial tableau", ats[0], expected0, cols, kinds, costs, dsign, tsense,
                                        kind="phase2", upstream=prev_err)
            else:
                expected0 = {"A": s0["A"], "b": s0["b"], "basis": s0["basis_idx"]}
                err0 = _compare_tableau(rep, f"{prefix}Initial tableau", ats[0], expected0, cols, kinds, costs, dsign, tsense,
                                        kind="phase2", upstream=None)
        rep.tableaus.append(_pair_json(f"{prefix}Initial tableau", ats[0], expected0, cols, kinds, costs, ph["name"]))
        err_at = None if not err0 else "Initial tableau"
        if err0 and first_err_global is None:
            first_err_global = f"{prefix}Initial tableau"
        # ---- transitions
        for k in range(len(ats)):
            cur = ats[k]
            nxt = ats[k + 1] if k + 1 < len(ats) else None
            lo = cur.src.end
            hi = nxt.src.pos if nxt else (tabs[k + 1].pos if k + 1 < len(tabs) else len(parsed.text))
            if nxt is None:
                # claims after the last tableau of this phase, up to the next phase's first tableau
                nxt_phase_first = groups[ph_idx + 1][0].pos if ph_idx + 1 < len(groups) and groups[ph_idx + 1] else len(parsed.text)
                hi = nxt_phase_first
            claims = [c for c in parsed.claims if lo <= c.pos < hi]
            if k == 0:
                claims = [c for c in parsed.claims if (cur.src.pos - 400 < c.pos < cur.src.pos and c.kind in ("entering", "leaving")
                          and not _claims_before_belong_prev(parsed, cur, ats))] + claims
            upstream = (f"{prefix}{err_at}" if err_at is not None else None) or prev_err
            res = _check_transition(rep, f"{prefix}Iteration {k + 1}" if nxt else f"{prefix}Final tableau", cur, nxt, claims, cols, kinds,
                                    costs, dsign, tsense, upstream, is_last=nxt is None,
                                    final_phase=ph_idx == len(phases) - 1, method=method)
            if nxt is not None:
                exp_next = res.get("expected_next")
                rep.tableaus.append(_pair_json(f"{prefix}Tableau after iteration {k + 1}", nxt, exp_next, cols, kinds, costs,
                                               ph["name"], upstream=upstream))
                if res.get("next_error"):
                    err_at = f"Iteration {k + 1}"
                    if first_err_global is None:
                        first_err_global = f"{prefix}Iteration {k + 1}"
        last_info = {"at": ats[-1], "cols": ref_cols, "kinds": kinds}
    return first_err_global


def _claims_before_belong_prev(parsed, cur, ats):
    return False


def _guess_two_phase_split(ai_tabs, phases):
    p1_cols = set(phases[0]["iterations"][0]["columns"])
    p2_cols = set(phases[1]["iterations"][0]["columns"])
    arts = {c.lower() for c in p1_cols - p2_cols}
    g1, g2 = [], []
    for t in ai_tabs:
        has_art = any(c.lower() in arts or re.fullmatch(r"[ar]\d+", c.lower()) for c in t.columns)
        (g1 if has_art and not g2 else g2).append(t)
    if not g1:
        n1 = len(phases[0]["iterations"])
        return [ai_tabs[:n1], ai_tabs[n1:]]
    return [g1, g2]


def _phase2_start(last, cols2):
    at = last["at"]
    cols1 = last["cols"]
    A, b = at.exact()
    idx = [cols1.index(c) if c in cols1 else None for c in cols2]
    A2 = [[row[j] if j is not None else None for j in idx] for row in A]
    basis2 = [cols2.index(cols1[j]) if j is not None and cols1[j] in cols2 else None for j in at.basis]
    return A2, b, basis2


def _detect_flip(at, s0, exp_cost):
    """'objective' if the answer uses Max(-Z) for a Min problem (or vice versa); 'delta' if it writes Zj - Cj
    under a Cj - Zj label; None otherwise."""
    dec = [j for j, k in enumerate(s0["kinds"]) if k == "decision"]
    costed = [j for j in range(len(exp_cost)) if sign(exp_cost[j]) != 0]
    if at.cj:
        vals = [(at.cj.get(j), exp_cost[j]) for j in costed if at.cj.get(j) is not None and at.cj.get(j).ok]
        if vals and any(sign(e) != 0 for _, e in vals) and all(_same(c, -e) for c, e in vals) and not all(
                _same(c, e) for c, e in vals):
            return "objective"
    if not at.cj and at.cb and all(c is not None and c.ok for c in at.cb):
        # no Cj row: the CB column of the starting basis reveals the convention (e.g. +1 for A1 when Phase I minimises W)
        vals = [(c, exp_cost[j]) for c, j in zip(at.cb, s0["basis_idx"]) if sign(exp_cost[j]) != 0]
        if vals and all(_same(c, -e) for c, e in vals):
            return "objective"
    if at.delta:
        nb = [j for j in range(len(exp_cost)) if j not in s0["basis_idx"]]
        vals = [(at.delta.get(j), s0["delta"][j]) for j in nb if at.delta.get(j) is not None and at.delta.get(j).ok]
        if vals and any(sign(e) != 0 for _, e in vals) and all(_same(c, -e) for c, e in vals):
            return "delta"
    return None


def _zj_delta(A, basis_cost, costs):
    n = len(costs)
    zj, delta = [], []
    for j in range(n):
        tot = Fraction(0)
        ok = True
        for i, row in enumerate(A):
            if row[j] is None or basis_cost[i] is None:
                ok = False
                break
            tot = tot + basis_cost[i] * row[j]
        zj.append(tot if ok else None)
        delta.append(costs[j] - tot if ok else None)
    return zj, delta


def _compare_tableau(rep, group, at, exp, cols, kinds, costs, dsign, tsense, kind, upstream):
    """Compare an AI tableau with an expected tableau (body, RHS, basis, CB, Cj, Zj, Cj - Zj, Z).

    Returns True if the tableau contains an error that is the AI's own (not inherited)."""
    own_error = False
    m = len(exp["A"])
    # rows are matched by basic variable (the AI may list rows in a different order)
    order = _row_order(at, exp)
    if order is None:
        rep.add(group, "Basic variables", "incorrect", "E07" if kind != "initial" else "E08",
                f"The basis column lists {', '.join(at.basis_names)}, but it should be "
                f"{', '.join(cols[j] if j is not None else '?' for j in exp['basis'])}.",
                ai=", ".join(at.basis_names),
                correct=", ".join(cols[j] if j is not None else "?" for j in exp["basis"]), consequence=upstream)
        order = list(range(min(m, len(at.A))))
        own_error = True
    body_bad, rhs_bad, art_sign_bad = [], [], []
    for i_exp, i_ai in enumerate(order):
        if i_ai is None or i_ai >= len(at.A):
            continue
        for j in range(len(cols)):
            e = exp["A"][i_exp][j]
            if e is None:
                continue
            r = _same(at.A[i_ai][j], e)
            if r is False:
                body_bad.append((i_exp, j, at.A[i_ai][j], e))
                if kinds[j] in ("slack", "surplus", "artificial") and at.A[i_ai][j].value == -e and kind == "initial":
                    art_sign_bad.append(j)
        if exp["b"][i_exp] is not None and _same(at.b[i_ai], exp["b"][i_exp]) is False:
            rhs_bad.append((i_exp, at.b[i_ai], exp["b"][i_exp]))
    label = {"initial": "Initial tableau", "phase2": "Phase II starting tableau"}.get(kind, "Tableau entries")
    if body_bad or rhs_bad:
        own_error = True
        for (i, j, a, e) in body_bad[:6]:
            code = "E02" if j in art_sign_bad else ("E08" if kind == "initial" else "E07")
            rep.add(group, f"{label}: row {cols[exp['basis'][i]] if exp['basis'][i] is not None else i + 1}, column {cols[j]}",
                    "incorrect", code, f"Entry should be {fmt_q(e)}.", ai=a, correct=e, row=i + 1, column=cols[j],
                    consequence=upstream)
        for (i, a, e) in rhs_bad[:4]:
            rep.add(group, f"{label}: RHS of row {i + 1}", "incorrect", "E08" if kind == "initial" else "E07",
                    f"Right-hand side should be {fmt_q(e)}.", ai=a, correct=e, row=i + 1, column="RHS", consequence=upstream)
        extra = len(body_bad) - 6
        if extra > 0:
            rep.add(group, f"{label}: further entries", "incorrect", "E08" if kind == "initial" else "E07",
                    f"{extra} more entries differ from the correct tableau.")
    else:
        rep.add(group, label, "correct" if upstream is None else "consequential",
                message="Constraint rows and RHS match MathGuard's tableau." if upstream is None else
                f"Rows match what follows from the earlier tableau ({upstream}).", consequence=upstream)
    # CB and Cj
    _check_costs(rep, group, at, exp, order, cols, costs, upstream)
    # Zj / Cj - Zj / Z, computed from the answer's own rows (internal consistency) and from the correct rows
    _check_eval_rows(rep, group, at, order, exp, cols, costs, dsign, upstream)
    return own_error


def _row_order(at, exp):
    """For each expected row, which AI row has the same basic variable?  None if the bases differ."""
    want = exp["basis"]
    got = at.basis
    if len(got) != len(want):
        return None
    if None in got:
        return list(range(len(want))) if got == want else None
    if set(got) != set(j for j in want if j is not None) and None not in want:
        return None
    order = []
    for j in want:
        order.append(got.index(j) if j in got else None)
    return order


def _check_costs(rep, group, at, exp, order, cols, costs, upstream):
    bad = []
    for i_exp, i_ai in enumerate(order):
        if i_ai is None or i_ai >= len(at.cb) or at.cb[i_ai] is None:
            continue
        bj = exp["basis"][i_exp]
        if bj is None:
            continue
        r = _same(at.cb[i_ai], costs[bj])
        if r is False:
            bad.append((i_exp, at.cb[i_ai], costs[bj], cols[bj]))
    for j, c in at.cj.items():
        if c is not None and c.ok and _same(c, costs[j]) is False:
            bad.append((None, c, costs[j], cols[j]))
    for (i, a, e, name) in bad[:4]:
        rep.add(group, f"{'CB of ' + name if i is not None else 'Cj of ' + name}", "incorrect", "E01",
                f"The objective coefficient of {name} is {fmt_q(e)}.", ai=a, correct=e, column=name, consequence=upstream)


def _check_eval_rows(rep, group, at, order, exp, cols, costs, dsign, upstream):
    if not at.zj and not at.delta and at.z is None:
        rep.add(group, "Zj and Cj - Zj rows", "incomplete",
                message="The tableau has no Zj / Cj - Zj row, so the optimality test cannot be checked from the table.")
        return
    A_ai, b_ai = at.exact()
    cb_ai = [(_snap(c) if c is not None else None) for c in at.cb]
    # basis costs: as written by the AI when given, otherwise the correct costs of the AI's basic variables
    bc = []
    for i in range(len(A_ai)):
        if cb_ai[i] is not None:
            bc.append(cb_ai[i])
        elif at.basis[i] is not None:
            bc.append(costs[at.basis[i]])
        else:
            bc.append(None)
    zj, delta = _zj_delta(A_ai, bc, costs)
    zbad, dbad = [], []
    for j, c in at.zj.items():
        if zj[j] is not None and _same(c, zj[j]) is False:
            zbad.append((j, c, zj[j]))
    for j, c in at.delta.items():
        if delta[j] is not None and _same(c, dsign * delta[j]) is False:
            dbad.append((j, c, dsign * delta[j]))
    zval = None
    if all(x is not None for x in bc) and all(x is not None for x in b_ai):
        zval = sum((bc[i] * b_ai[i] for i in range(len(b_ai))), Fraction(0))
    zwrong = at.z is not None and zval is not None and _same(at.z, zval) is False and _same(at.z, -zval) is False
    if zbad:
        j, a, e = zbad[0]
        rep.add(group, f"Zj row (column {cols[j]})", "incorrect", "E07",
                f"Zj = sum of CB x column entries = {fmt_q(e)} for {cols[j]}" + (f" ({len(zbad) - 1} more Zj entries are wrong)"
                                                                                if len(zbad) > 1 else "") + ".",
                ai=a, correct=e, column=cols[j], consequence=upstream)
    if dbad:
        j, a, e = dbad[0]
        rep.add(group, f"Cj - Zj row (column {cols[j]})", "incorrect", "E07",
                f"Cj - Zj for {cols[j]} should be {fmt_q(e)}" + (f" ({len(dbad) - 1} more entries are wrong)" if len(dbad) > 1 else "") + ".",
                ai=a, correct=e, column=cols[j], consequence=upstream)
    if zwrong:
        rep.add(group, "Value of Z in the tableau", "incorrect", "E07", f"Z = sum of CB x RHS = {fmt_q(zval)}.",
                ai=at.z, correct=zval, consequence=upstream)
    if not zbad and not dbad and not zwrong:
        rep.add(group, "Zj / Cj - Zj rows", "correct" if upstream is None else "consequential",
                message="Zj and Cj - Zj are computed correctly from the tableau." +
                ("" if upstream is None else f" (The tableau itself inherits the earlier error in {upstream}.)"),
                consequence=upstream)


def _improving(d, tsense):
    return sign(d) > 0 if tsense == "max" else sign(d) < 0


def _check_transition(rep, group, cur, nxt, claims, cols, kinds, costs, dsign, tsense, upstream, is_last, final_phase, method):
    """Entering / leaving / ratios / pivot / row operations / next tableau, judged from the AI's own tableau."""
    out = {}
    A, b = cur.exact()
    m = len(A)
    bc = [(_snap(c) if c is not None and c.ok else (costs[cur.basis[i]] if cur.basis[i] is not None else None))
          for i, c in enumerate(cur.cb)]
    zj, delta = _zj_delta(A, bc, costs)
    if dsign == -1 and tsense != _base_sense(costs, cur):
        pass
    excluded = set(j for j, k in enumerate(kinds) if k == "artificial" and method == "two_phase" and "Phase II" in group)
    cands = [j for j in range(len(cols)) if j not in cur.basis and delta[j] is not None and _improving(delta[j], tsense)
             and j not in excluded]
    written = {j: c for j, c in cur.delta.items() if c is not None and c.ok}
    # what the answer itself would choose from the numbers it wrote
    written_cands = [j for j, c in written.items() if j not in cur.basis and _improving(dsign * c.value, tsense)]

    enter_claims = [c for c in claims if c.kind == "entering"]
    leave_claims = [c for c in claims if c.kind == "leaving"]
    concl = [c.value for c in claims if c.kind == "conclusion"]

    e_ai = l_ai = None
    l_row = None
    if nxt is not None:
        newb = [j for j in nxt.basis if j not in cur.basis]
        oldb = [i for i, j in enumerate(cur.basis) if j not in nxt.basis]
        if len(newb) == 1 and len(oldb) == 1:
            e_ai, l_row = newb[0], oldb[0]
            l_ai = cur.basis[l_row]
        elif len(newb) > 1 or len(oldb) > 1:
            rep.add(group, "Basis change", "incorrect", "E07",
                    "More than one basic variable changed between consecutive tableaus; a simplex iteration replaces exactly "
                    "one basic variable.", ai=", ".join(nxt.basis_names), consequence=upstream)
            out["next_error"] = True
            return out
    name_to_idx = {c.lower(): j for j, c in enumerate(cols)}
    inv = {v: k for k, v in cur.inv.items()}

    def idx_of(v):
        if v.lower() in name_to_idx:
            return name_to_idx[v.lower()]
        for k, j in ((k.lower(), j) for k, j in ((n, jj) for jj, n in cur.inv.items())):
            if k == v.lower():
                return j
        return None

    ce = idx_of(enter_claims[-1].value) if enter_claims else None
    cl = idx_of(leave_claims[-1].value) if leave_claims else None
    if e_ai is not None and ce is not None and ce != e_ai:
        rep.add(group, "Entering variable statement", "unverifiable",
                message=f"The text says {enter_claims[-1].value} enters, but the next tableau brings in {cols[e_ai]}. "
                        "The answer contradicts itself; the tableau is used for the remaining checks.")
    if e_ai is None:
        e_ai = ce
    if l_ai is None and cl is not None:
        l_ai = cl
        l_row = cur.basis.index(cl) if cl in cur.basis else None

    # ---------------- is the current tableau already optimal?
    if not cands:
        if e_ai is not None and nxt is not None:
            rep.add(group, "Optimality test", "incorrect", "E13",
                    "This tableau is already optimal (no Cj - Zj can improve Z), but the answer carries on pivoting.",
                    ai=f"{cols[e_ai]} enters", correct="stop: optimal", consequence=upstream)
            out["next_error"] = True
        elif is_last:
            if "unbounded" in concl or "infeasible" in concl:
                pass  # judged in the final-answer section
            else:
                rep.add(group, "Optimality test", "correct" if upstream is None else "consequential",
                        message=("All Cj - Zj values satisfy the optimality condition, so stopping here is correct."
                                 if upstream is None else f"Correctly stops at an optimal tableau, but this tableau inherits the earlier error in {upstream}."),
                        consequence=upstream)
        return out
    if is_last and e_ai is None:
        # answer stopped while an improving variable exists (unless it declared unboundedness correctly)
        best = _best(cands, delta, tsense)
        if "unbounded" in concl:
            proof = [j for j in cands if all(A[i][j] is not None and A[i][j] <= 0 for i in range(m))]
            if proof:
                rep.add(group, "Unboundedness test", "correct" if upstream is None else "consequential",
                        message=f"{cols[proof[0]]} improves Z and its column has no positive entry, so the solution is unbounded.",
                        consequence=upstream)
            else:
                rep.add(group, "Unboundedness test", "incorrect", "E14",
                        f"{cols[best]} has positive entries in its column, so the ratio test is possible: the problem is not "
                        "shown to be unbounded here.", ai="unbounded", correct="continue: ratio test", consequence=upstream)
            return out
        colb = [A[i][best] for i in range(m)]
        if all(x is not None and x <= 0 for x in colb):
            rep.add(group, "Unboundedness test", "incorrect", "E14",
                    f"{cols[best]} still improves Z (Cj - Zj = {fmt_q(dsign * delta[best])}) and no entry in its column is "
                    "positive, so Z can grow without limit: the problem is unbounded, not optimal.",
                    ai="stop: optimal", correct="unbounded", consequence=upstream)
            return out
        rep.add(group, "Optimality test", "incorrect", "E13",
                f"The answer stops here, but {cols[best]} still has Cj - Zj = {fmt_q(dsign * delta[best])}, which improves Z. "
                "The simplex method must continue (premature termination).",
                ai="stop: optimal", correct=f"{cols[best]} enters", consequence=upstream)
        return out
    if e_ai is None:
        rep.add(group, "Entering variable", "unverifiable", message="The answer does not show which variable enters.")
        return out

    # ---------------- entering variable
    best = _best(cands, delta, tsense)
    bestval = delta[best]
    if e_ai in cands:
        if delta[e_ai] == bestval:
            rep.add(group, "Entering variable", "correct" if upstream is None else "consequential",
                    message=f"{cols[e_ai]} has the {'largest positive' if tsense == 'max' else 'most negative'} Cj - Zj "
                            f"({fmt_q(dsign * delta[e_ai])}).", ai=cols[e_ai], correct=cols[e_ai], consequence=upstream)
        else:
            rep.add(group, "Entering variable", "deviation", "E03",
                    f"{cols[e_ai]} improves Z (Cj - Zj = {fmt_q(dsign * delta[e_ai])}), but the textbook rule chooses "
                    f"{cols[best]} (Cj - Zj = {fmt_q(dsign * bestval)}). This is a different but valid simplex path, so the later "
                    "tableaus are checked on their own terms.", ai=cols[e_ai], correct=cols[best], consequence=upstream)
    else:
        follows = e_ai in written_cands
        st = "consequential" if follows and upstream else "incorrect"
        rep.add(group, "Entering variable", st, "E03",
                f"{cols[e_ai]} cannot enter: its Cj - Zj is {fmt_q(dsign * delta[e_ai]) if delta[e_ai] is not None else '?'}, "
                f"which does not improve Z. {cols[best]} should enter." + (" (This follows from the wrong Cj - Zj row above.)" if follows else ""),
                ai=cols[e_ai], correct=cols[best], consequence=upstream)
        if st == "incorrect":
            out["next_error"] = True

    e = e_ai
    col = [A[i][e] for i in range(m)]
    ratios = [(b[i] / col[i] if col[i] is not None and b[i] is not None and col[i] > 0 else None) for i in range(m)]
    # ---------------- ratio column / ratio statements
    rbad = []
    for i in range(m):
        rc = cur.ratio[i]
        raw = (cur.ratio_raw[i] or "").strip()
        if rc is None and not raw:
            continue
        if ratios[i] is None:
            if rc is not None and rc.ok and not re.search(r"(?i)∞|inf|--|—|n/?a|undefined|-$|^-$", raw):
                rbad.append((i, rc, "no ratio (entry is not positive)"))
        elif rc is not None and rc.ok and _same(rc, ratios[i]) is False:
            rbad.append((i, rc, fmt_q(ratios[i])))
    for c in [c for c in claims if c.kind == "ratio"]:
        a, bb, r = c.value
        if bb.value != 0 and _same(r, a.value / bb.value) is False:
            rbad.append((None, r, fmt_q(a.value / bb.value), c.text))
    if rbad:
        for item in rbad[:3]:
            i, a, e_ = item[0], item[1], item[2]
            where = f"row {cur.basis_names[i]}" if i is not None else item[3]
            rep.add(group, f"Ratio test ({where})", "incorrect", "E04", f"The ratio should be {e_}.", ai=a, correct=e_,
                    row=None if i is None else i + 1, consequence=upstream)
    elif any(cur.ratio_raw) or any(c.kind == "ratio" for c in claims):
        rep.add(group, "Ratio calculations", "correct" if upstream is None else "consequential",
                message="Each ratio RHS / pivot-column entry is computed correctly.", consequence=upstream)

    # ---------------- leaving variable
    valid = [r for r in ratios if r is not None]
    if not valid:
        if nxt is not None:
            rep.add(group, "Leaving variable", "incorrect", "E14",
                    f"No entry in the {cols[e]} column is positive, so no variable can leave: the problem is unbounded and "
                    "the answer should have stopped here.", ai=cur.basis_names[l_row] if l_row is not None else None,
                    correct="unbounded", consequence=upstream)
            out["next_error"] = True
        elif "unbounded" in concl:
            rep.add(group, "Unboundedness test", "correct" if upstream is None else "consequential",
                    message=f"{cols[e]} has no positive entry in its column, so Z can increase without limit: unbounded.",
                    consequence=upstream)
        return out
    mn = min(valid)
    if l_row is None:
        rep.add(group, "Leaving variable", "unverifiable", message="The answer does not show which variable leaves.")
        return out
    if ratios[l_row] is not None and ratios[l_row] == mn:
        ties = [cur.basis_names[i] for i, r in enumerate(ratios) if r == mn and i != l_row]
        rep.add(group, "Leaving variable", "correct" if upstream is None else "consequential",
                message=f"{cur.basis_names[l_row]} has the minimum ratio {fmt_q(mn)}." +
                (f" (Tie with {', '.join(ties)}: either choice is valid; the next solution is degenerate.)" if ties else ""),
                ai=cur.basis_names[l_row], correct=cur.basis_names[l_row], consequence=upstream)
    else:
        good = [cur.basis_names[i] for i, r in enumerate(ratios) if r == mn]
        why = ("its pivot-column entry is not positive, so it has no ratio" if ratios[l_row] is None
               else f"its ratio is {fmt_q(ratios[l_row])}, larger than the minimum {fmt_q(mn)}")
        rep.add(group, "Leaving variable", "incorrect", "E04",
                f"{cur.basis_names[l_row]} cannot leave: {why}. {', '.join(good)} should leave. Pivoting on the wrong row "
                "makes some RHS values negative (an infeasible basic solution).",
                ai=cur.basis_names[l_row], correct=", ".join(good), consequence=upstream)
        out["next_error"] = True

    # ---------------- pivot element
    piv = A[l_row][e]
    for c in [c for c in claims if c.kind == "pivot"]:
        if _same(c.value, piv) is False:
            rep.add(group, "Pivot element", "incorrect", "E11",
                    f"The pivot element is the entry in row {cur.basis_names[l_row]}, column {cols[e]}: {fmt_q(piv)}.",
                    ai=c.value, correct=piv, consequence=upstream)
        else:
            rep.add(group, "Pivot element", "correct" if upstream is None else "consequential",
                    message=f"Pivot element {fmt_q(piv)} (row {cur.basis_names[l_row]}, column {cols[e]}).", consequence=upstream)
        break

    # ---------------- row-operation statements
    for c in [c for c in claims if c.kind == "row_op"]:
        _check_row_op(rep, group, c, A, e, l_row, upstream)

    # ---------------- next tableau
    if nxt is None or piv is None or piv == 0:
        return out
    expA = [list(r) for r in A]
    expb = list(b)
    if any(x is None for x in expA[l_row]) or expb[l_row] is None:
        rep.add(group, "Next tableau", "unverifiable", message="The pivot row has unreadable entries, so the next tableau "
                                                                "cannot be recomputed.")
        return out
    expA[l_row] = [x / piv for x in expA[l_row]]
    expb[l_row] = expb[l_row] / piv
    for i in range(m):
        if i == l_row:
            continue
        f = A[i][e]
        if f is None:
            expA[i] = [None] * len(cols)
            expb[i] = None
            continue
        expA[i] = [(x - f * y) if x is not None and y is not None else None for x, y in zip(A[i], expA[l_row])]
        expb[i] = (b[i] - f * expb[l_row]) if b[i] is not None else None
    basis = list(cur.basis)
    basis[l_row] = e
    exp = {"A": expA, "b": expb, "basis": basis}
    out["expected_next"] = exp
    # compare
    order = _row_order(nxt, exp)
    gnext = group
    if order is None:
        rep.add(gnext, "Basic variables after the pivot", "incorrect", "E07",
                f"After {cols[e]} replaces {cur.basis_names[l_row]}, the basis should be "
                f"{', '.join(cols[j] for j in basis)}.", ai=", ".join(nxt.basis_names),
                correct=", ".join(cols[j] for j in basis), consequence=upstream)
        out["next_error"] = True
        return out
    piv_bad, other_bad = [], []
    for i_exp, i_ai in enumerate(order):
        for j in range(len(cols)):
            ev = expA[i_exp][j]
            if ev is None:
                continue
            r = _same(nxt.A[i_ai][j], ev)
            if r is False:
                (piv_bad if i_exp == l_row else other_bad).append((i_exp, j, nxt.A[i_ai][j], ev))
        if expb[i_exp] is not None and _same(nxt.b[i_ai], expb[i_exp]) is False:
            (piv_bad if i_exp == l_row else other_bad).append((i_exp, "RHS", nxt.b[i_ai], expb[i_exp]))
    if piv_bad:
        i, j, a, ev = piv_bad[0]
        rep.add(gnext, f"Pivot row ({cols[e]}) normalisation", "incorrect", "E05",
                f"New pivot row = old row {cur.basis_names[l_row]} / {fmt_q(piv)}. Column {j if j == 'RHS' else cols[j]} should be "
                f"{fmt_q(ev)}" + (f"; {len(piv_bad) - 1} more entries of this row are wrong" if len(piv_bad) > 1 else "") + ".",
                ai=a, correct=ev, row=i + 1, column=j if j == "RHS" else cols[j], consequence=upstream)
        out["next_error"] = True
    else:
        rep.add(gnext, f"Pivot row ({cols[e]}) normalisation", "correct" if upstream is None else "consequential",
                message=f"Row {cur.basis_names[l_row]} divided by the pivot {fmt_q(piv)} correctly.", consequence=upstream)
    if other_bad:
        by_row = {}
        for item in other_bad:
            by_row.setdefault(item[0], []).append(item)
        for i, items in list(by_row.items())[:3]:
            _, j, a, ev = items[0]
            f = A[i][e]
            rep.add(gnext, f"Row operation on row {cols[basis[i]]}", "incorrect", "E06",
                    f"New row = old row - ({fmt_q(f)}) x new pivot row. Column {j if j == 'RHS' else cols[j]} should be "
                    f"{fmt_q(ev)}" + (f"; {len(items) - 1} more entries of this row are wrong" if len(items) > 1 else "") + ".",
                    ai=a, correct=ev, row=i + 1, column=j if j == "RHS" else cols[j], consequence=upstream)
        out["next_error"] = True
    else:
        rep.add(gnext, "Other row operations", "correct" if upstream is None else "consequential",
                message="Every other row was updated correctly (old row - factor x new pivot row).", consequence=upstream)
    nup = upstream if not out.get("next_error") else (upstream or gnext)
    _check_costs(rep, gnext, nxt, exp, order, cols, costs, nup)
    _check_eval_rows(rep, gnext, nxt, order, exp, cols, costs, dsign, nup if out.get("next_error") else upstream)
    return out


def _base_sense(costs, cur):
    return None


def _best(cands, delta, tsense):
    if tsense == "max":
        return max(cands, key=lambda j: (delta[j], -j))
    return min(cands, key=lambda j: (delta[j], j))


_RTERM = re.compile(r"([+-]?)\s*(\(?-?\d+(?:\.\d+)?(?:/\d+)?\)?)?\s*\*?\s*x?\s*R\s*(\d)")


def _check_row_op(rep, group, claim, A, e, l_row, upstream):
    target, expr = claim.value
    ti = target - 1
    if ti < 0 or ti >= len(A):
        return
    expr = expr.split(",")[0]
    expr = re.sub(r"\(new\)|new|'|_new", "", expr, flags=re.I)
    terms = list(_RTERM.finditer(expr))
    if not terms:
        return
    coef = {}
    for t in terms:
        sgn = -1 if t.group(1) == "-" else 1
        k = t.group(2)
        kv = Fraction(1)
        if k:
            k = k.strip("()")
            try:
                kv = Fraction(k)
            except (ValueError, ZeroDivisionError):
                return
        coef[int(t.group(3)) - 1] = coef.get(int(t.group(3)) - 1, 0) + sgn * kv
    div = re.search(r"R\s*\d\s*/\s*(\(?-?\d+(?:\.\d+)?(?:/\d+)?\)?)", expr)
    if ti == l_row:
        piv = A[l_row][e]
        if div and piv is not None:
            try:
                d = Fraction(div.group(1).strip("()"))
            except (ValueError, ZeroDivisionError):
                return
            if d != piv:
                rep.add(group, f"Row operation R{target}", "incorrect", "E05",
                        f"The pivot row must be divided by the pivot element {fmt_q(piv)}.", ai=claim.text,
                        correct=f"R{target} / {fmt_q(piv)}", consequence=upstream)
        return
    others = {k: v for k, v in coef.items() if k != ti}
    if set(others) != {l_row}:
        return
    f = A[ti][e]
    if f is None:
        return
    if others[l_row] != -f:
        want = f"R{target} - {fmt_q(f)}R{l_row + 1}" if f > 0 else f"R{target} + {fmt_q(-f)}R{l_row + 1}"
        rep.add(group, f"Row operation R{target}", "incorrect", "E06",
                f"Row {target} has {fmt_q(f)} in the pivot column, so it must be updated with R{target} - ({fmt_q(f)}) x pivot row.",
                ai=claim.text.strip(), correct=want, consequence=upstream)


# ----------------------------------------------------------------------------- claims-only answers (no tableaus)
def verify_claims_only(rep, ref, parsed):
    rep.add("Working", "Simplex tableaus", "incomplete",
            message="The answer does not show its tableaus, so tableau entries and row operations cannot be verified. "
                    "Stated entering / leaving choices are checked against MathGuard's tableaus where possible.")
    its = [s for p in ref["phases"] for s in p["iterations"]]
    ent = [c for c in parsed.claims if c.kind == "entering"]
    lea = [c for c in parsed.claims if c.kind == "leaving"]
    on_path = True
    for k, ec in enumerate(ent):
        g = f"Iteration {k + 1}"
        if k >= len(its) or not on_path:
            rep.add(g, "Entering variable", "unverifiable",
                    message=f"'{ec.value} enters' cannot be checked: the earlier steps differ from MathGuard's path or are not shown.")
            continue
        s = its[k]
        cols = s["columns"]
        if s.get("entering") is None:
            rep.add(g, "Entering variable", "incorrect", "E13",
                    "MathGuard's tableau at this stage is already optimal, so no variable should enter.", ai=ec.value,
                    correct="stop: optimal")
            on_path = False
            continue
        if ec.value.lower() == s["entering"].lower():
            rep.add(g, "Entering variable", "correct", message=f"{ec.value} enters, as in MathGuard's tableau.",
                    ai=ec.value, correct=s["entering"])
        else:
            j = next((jj for jj, c in enumerate(cols) if c.lower() == ec.value.lower()), None)
            improving = j is not None and _improving(s["delta"][j], s["sense"])
            rep.add(g, "Entering variable", "deviation" if improving else "incorrect", "E03",
                    (f"{ec.value} improves Z but the textbook rule chooses {s['entering']}." if improving else
                     f"{ec.value} does not improve Z; {s['entering']} should enter."), ai=ec.value, correct=s["entering"])
            on_path = False
            continue
        lc = lea[k] if k < len(lea) else None
        if lc is not None:
            ties = [s["leaving"]] + s.get("leaving_ties", [])
            if lc.value.lower() in [t.lower() for t in ties]:
                rep.add(g, "Leaving variable", "correct", message=f"{lc.value} has the minimum ratio.", ai=lc.value,
                        correct=s["leaving"])
            else:
                rep.add(g, "Leaving variable", "incorrect", "E04", f"{s['leaving']} has the minimum ratio, not {lc.value}.",
                        ai=lc.value, correct=s["leaving"])
                on_path = False


# ----------------------------------------------------------------------------- graphical method
def verify_graphical(rep, prob, g, parsed):
    vs = g["variables"]
    lines = g["boundary_lines"]
    corners = {cn["point"]: cn for cn in g["corners"]}
    ev = exact_view(prob)
    c = ev["c"]
    # 1. boundary equations
    checked_eq = False
    for eq in parsed.equations:
        a = (Fraction(eq["coefs"].get(vs[0], 0)), Fraction(eq["coefs"].get(vs[1], 0)))
        bq = eq["rhs"].value
        if a == (0, 0):
            continue
        if _parallel(a, (c[vs[0]], c[vs[1]])) and not any(_same_line(a, bq, l) for l in lines):
            continue  # an iso-profit / objective line, not a constraint boundary
        hit = next((l for l in lines if _same_line(a, bq, l)), None)
        if hit:
            checked_eq = True
            continue
        cand = next((l for l in lines if (l["a"][0] != 0) == (a[0] != 0) and (l["a"][1] != 0) == (a[1] != 0)
                     and (l["a"] == a or l["b"] == bq)), None)
        if cand:
            rep.add("Boundary lines", f"Boundary of {cand['constraint']}", "incorrect", "G01",
                    f"The boundary line of {cand['constraint']} is {cand['equation']}.", ai=eq["text"], correct=cand["equation"])
            checked_eq = True
    if checked_eq and not any(s.get("code") == "G01" for s in rep.steps):
        rep.add("Boundary lines", "Boundary equations", "correct",
                message="Every boundary equation in the answer matches a constraint of the problem.")
    # 2. intercepts: points written on the same line as a boundary equation, with a zero coordinate
    eq_by_line = {}
    for eq in parsed.equations:
        eq_by_line.setdefault(eq["line_no"], []).append(eq)
    icount = 0
    for p in parsed.points:
        if p["line_no"] not in eq_by_line:
            continue
        if not re.search(r"(?i)axes|axis|intercept|meets|cuts|put|when", p["line"]):
            continue
        pa, pb = p["point"]
        if not (pa.ok and pb.ok) or (pa.value != 0 and pb.value != 0):
            continue
        for eq in eq_by_line[p["line_no"]]:
            a = (Fraction(eq["coefs"].get(vs[0], 0)), Fraction(eq["coefs"].get(vs[1], 0)))
            line = next((l for l in lines if _same_line(a, eq["rhs"].value, l)), None)
            if line is None:
                continue
            icount += 1
            truth = line["x_intercept"] if pb.value == 0 and pa.value != 0 else line["y_intercept"]
            if truth is None:
                continue
            if not (_same(pa, truth[0]) and _same(pb, truth[1])):
                rep.add("Boundary lines", f"Intercept of {line['equation']}", "incorrect", "G01",
                        f"{line['equation']} meets the {'x' if pb.value == 0 else 'y'}-axis at "
                        f"({fmt_q(truth[0])}, {fmt_q(truth[1])}).", ai=(pa, pb), correct=truth)
            break
    if icount and not any(s.get("code") == "G01" and "Intercept" in s["title"] for s in rep.steps):
        rep.add("Boundary lines", "Intercepts", "correct", message="The axis intercepts are correct.")
    # 3. intersection calculations
    wrong_pts = []  # (answer's point, true point) for miscalculated intersections -> downstream consequences
    for it in parsed.intersections:
        eqs = eq_by_line.get(_line_no(parsed, it), [])
        for pa, pb in it["points"]:
            if not (pa.ok and pb.ok):
                continue
            p = (pa.value, pb.value)
            on = [l for l in lines + _axes(vs) if l["a"][0] * p[0] + l["a"][1] * p[1] == l["b"]]
            named = [l for l in (next((L for L in lines if _same_line((Fraction(e["coefs"].get(vs[0], 0)),
                                                                        Fraction(e["coefs"].get(vs[1], 0))), e["rhs"].value, L)), None)
                                 for e in eqs) if l]
            if len(named) >= 2:
                L1, L2 = named[0], named[1]
                truth = _intersect(L1, L2)
                if truth is None:
                    continue
                if _same(pa, truth[0]) and _same(pb, truth[1]):
                    rep.add("Intersections", f"{L1['equation']} and {L2['equation']}", "correct",
                            message=f"Intersection ({fmt_q(truth[0])}, {fmt_q(truth[1])}) is correct.")
                else:
                    rep.add("Intersections", f"{L1['equation']} and {L2['equation']}", "incorrect", "G02",
                            f"Solving the two equations gives ({fmt_q(truth[0])}, {fmt_q(truth[1])}).",
                            ai=(pa, pb), correct=truth)
                    wrong_pts.append(((pa.value, pb.value), truth, f"{L1['equation']} and {L2['equation']}"))
            elif len(on) >= 2:
                rep.add("Intersections", f"Point ({pa.raw}, {pb.raw})", "correct",
                        message="The point lies on two boundary lines, so it is a genuine intersection point.")
            else:
                rep.add("Intersections", f"Point ({pa.raw}, {pb.raw})", "incorrect", "G02",
                        "This point is not the intersection of any two boundary lines.", ai=(pa, pb))
            break
    # 4. corner points
    lists = parsed.corner_list
    if lists:
        cl = max(lists, key=lambda x: len(x["points"]))
        given = []
        for pa, pb in cl["points"]:
            if pa.ok and pb.ok:
                given.append((pa, pb))
        extra, matched = [], set()
        for pa, pb in given:
            hit = next((p for p in corners if _same(pa, p[0]) and _same(pb, p[1])), None)
            if hit is not None:
                matched.add(hit)
                continue
            extra.append((pa, pb))
        missing = [p for p in corners if p not in matched]
        for pa, pb in extra:
            p = (pa.value, pb.value)
            viol = _violations(ev, vs, p)
            src = next((w for w in wrong_pts if w[0] == p), None)
            if src:
                rep.add("Feasible region", f"Corner point ({pa.raw}, {pb.raw})", "consequential", "G03",
                        f"Listed as a corner because of the wrong intersection of {src[2]}" +
                        (f"; it actually violates {', '.join(viol)}." if viol else "."), ai=(pa, pb), correct=src[1],
                        consequence=f"Intersections: {src[2]}")
            elif viol:
                rep.add("Feasible region", f"Corner point ({pa.raw}, {pb.raw})", "incorrect", "G03",
                        f"This point violates {', '.join(viol)}, so it is not in the feasible region.", ai=(pa, pb))
            else:
                rep.add("Feasible region", f"Corner point ({pa.raw}, {pb.raw})", "incorrect", "G03",
                        "This point is feasible but is not a corner point of the feasible region.", ai=(pa, pb))
        caused = [p for p in missing if any(w[1] == p for w in wrong_pts)]
        for p in caused:
            src = next(w for w in wrong_pts if w[1] == p)
            rep.add("Feasible region", f"Corner point ({fmt_q(p[0])}, {fmt_q(p[1])})", "consequential", "G03",
                    f"Missing because the intersection of {src[2]} was miscalculated.", correct=p,
                    consequence=f"Intersections: {src[2]}")
        missing = [p for p in missing if p not in caused]
        if missing:
            opt = set(g["optimal_corners"])
            hurts = any(p in opt for p in missing)
            rep.add("Feasible region", "Missing corner points", "incorrect", "G03",
                    "The corner-point list is incomplete; missing: " + ", ".join(f"({fmt_q(p[0])}, {fmt_q(p[1])})" for p in missing) +
                    (". One of them is optimal, so leaving it out changes the answer." if hurts else
                     ". (It is not optimal, so the final answer is unaffected.)"),
                    correct=", ".join(f"({fmt_q(p[0])}, {fmt_q(p[1])})" for p in missing))
        if not extra and not missing and not caused:
            rep.add("Feasible region", "Corner points", "correct",
                    message="The answer lists exactly the corner points of the feasible region.")
    elif g["status"] not in ("infeasible",):
        rep.add("Feasible region", "Corner points", "incomplete",
                message="No list of corner points was found, so the feasible region cannot be checked step by step.")
    # 5. objective at each corner
    seen = set()
    ok_evals = 0
    for e in parsed.evaluations:
        pa, pb = e["point"]
        key = (pa.raw, pb.raw, e["value"].raw)
        if key in seen or not (pa.ok and pb.ok):
            continue
        seen.add(key)
        z = c[vs[0]] * pa.value + c[vs[1]] * pb.value
        if _same(e["value"], z) is False:
            rep.add("Objective evaluation", f"Z at ({pa.raw}, {pb.raw})", "incorrect", "G04",
                    f"Z = {fmt_q(c[vs[0]])}({pa.raw}) + {fmt_q(c[vs[1]])}({pb.raw}) = {fmt_q(z)}.", ai=e["value"], correct=z)
        else:
            ok_evals += 1
    if ok_evals and not any(s.get("code") == "G04" for s in rep.steps):
        rep.add("Objective evaluation", "Z at the corner points", "correct",
                message=f"All {ok_evals} objective evaluations are arithmetically correct.")
    # 6. choice of the optimum among the answer's own evaluations
    if parsed.evaluations:
        vals = [(e["value"].value, e) for e in parsed.evaluations if e["value"].ok]
        if vals:
            best = max(v for v, _ in vals) if g["sense"] == "max" else min(v for v, _ in vals)
            final = parse_answer(parsed.text, vs)
            fv = final.get("value")
            if fv is not None and fv.v is not None and abs(Fraction(str(fv.v)) - best) > Fraction(1, 1000) * max(1, abs(best)):
                rep.add("Optimal corner", "Choice of the optimal corner", "incorrect", "G05",
                        f"Among its own evaluations the {'largest' if g['sense'] == 'max' else 'smallest'} Z is {fmt_q(best)}, "
                        "but the answer reports a different value.", ai=fv.v, correct=best)
            elif fv is not None:
                rep.add("Optimal corner", "Choice of the optimal corner", "correct",
                        message=f"The answer picks the {'largest' if g['sense'] == 'max' else 'smallest'} of its own Z values.")


def _line_no(parsed, it):
    return it.get("line_no")


def _axes(vs):
    return [{"a": (Fraction(1), Fraction(0)), "b": Fraction(0), "equation": f"{vs[0]} = 0"},
            {"a": (Fraction(0), Fraction(1)), "b": Fraction(0), "equation": f"{vs[1]} = 0"}]


def _parallel(a, c):
    return a[0] * c[1] == a[1] * c[0]


def _same_line(a, b, l):
    (p, q), r = l["a"], l["b"]
    if a[0] * q != a[1] * p:
        return False
    k = (a[0] / p) if p != 0 else (a[1] / q)
    return b == k * r


def _intersect(L1, L2):
    (a1, a2), b1 = L1["a"], L1["b"]
    (c1, c2), b2 = L2["a"], L2["b"]
    D = a1 * c2 - a2 * c1
    if D == 0:
        return None
    return ((b1 * c2 - a2 * b2) / D, (a1 * b2 - b1 * c1) / D)


def _violations(ev, vs, p):
    out = []
    for r in ev["rows"]:
        lhs = r["coefs"][vs[0]] * p[0] + r["coefs"][vs[1]] * p[1]
        if (r["op"] == "<=" and lhs > r["rhs"]) or (r["op"] == ">=" and lhs < r["rhs"]) or (r["op"] == "=" and lhs != r["rhs"]):
            out.append(r["text"])
    if p[0] < 0:
        out.append(f"{vs[0]} >= 0")
    if p[1] < 0:
        out.append(f"{vs[1]} >= 0")
    return out


# ----------------------------------------------------------------------------- tableau JSON for side-by-side display
def _pair_json(label, at, exp, cols, kinds, costs, phase, upstream=None):
    rows = []
    order = _row_order(at, exp) if exp else None
    inv_order = {}
    if order:
        for i_exp, i_ai in enumerate(order):
            if i_ai is not None:
                inv_order[i_ai] = i_exp
    for i_ai, bn in enumerate(at.basis_names):
        i_exp = inv_order.get(i_ai)
        cells = []
        for j in range(len(cols)):
            c = at.A[i_ai][j]
            e = exp["A"][i_exp][j] if exp and i_exp is not None else None
            st = None if c is None or e is None else ("ok" if _same(c, e) else "bad")
            cells.append({"ai": c.raw if c is not None else "", "exp": fmt_q(e) if e is not None else None, "st": st})
        bc = at.b[i_ai]
        be = exp["b"][i_exp] if exp and i_exp is not None else None
        rows.append({"basis": bn, "cb": at.cb[i_ai].raw if at.cb[i_ai] is not None else "",
                     "cells": cells, "rhs": {"ai": bc.raw if bc is not None else "", "exp": fmt_q(be) if be is not None else None,
                                            "st": None if bc is None or be is None else ("ok" if _same(bc, be) else "bad")}})
    expected = None
    if exp:
        bas = exp["basis"]
        expected = {"basis": [cols[j] if j is not None else "?" for j in bas],
                    "cb": [fmt_q(costs[j]) if j is not None else "" for j in bas],
                    "rows": [[fmt_q(x) if x is not None else "?" for x in r] for r in exp["A"]],
                    "rhs": [fmt_q(x) if x is not None else "?" for x in exp["b"]]}
        if all(j is not None for j in bas) and all(x is not None for r in exp["A"] for x in r):
            zj, d = _zj_delta(exp["A"], [costs[j] for j in bas], costs)
            expected["zj"] = [fmt_q(x) for x in zj]
            expected["delta"] = [fmt_q(x) for x in d]
    return {"label": label, "phase": phase, "columns": cols, "ai_rows": rows, "expected": expected,
            "ai_delta": {cols[j]: c.raw for j, c in at.delta.items()}, "ai_zj": {cols[j]: c.raw for j, c in at.zj.items()},
            "upstream_error": upstream}


# ----------------------------------------------------------------------------- final answer
def check_final(rep, prob, truth, answer_text, method_res=None):
    from . import features
    vs = prob["variables"]
    claim = parse_answer(answer_text, vs)
    f, det = features.build(prob, truth, claim)
    label = features.judge(f)
    group = "Final answer"
    if label == "unclear":
        rep.add(group, "Final answer", "unverifiable",
                message="No final solution or objective value could be found in the answer (state results like "
                        "'x = 2, y = 2, Z = 10').")
        return {"label": "unclear", "claim": claim, "features": f, "details": det}
    if label == "correct":
        msg = {"infeasible": "Correctly concludes that the problem is infeasible.",
               "unbounded": "Correctly concludes that the problem is unbounded."}.get(truth["status"])
        if not msg:
            bits = []
            if f["sol_state"] == 2:
                bits.append("the stated solution is feasible and optimal")
            elif f["sol_state"] == 1:
                bits.append("the variables given match the optimum")
            if f["val_provided"]:
                bits.append("the objective value is correct")
            msg = "Final answer verified: " + ", ".join(bits) + "." if bits else "Final answer verified."
            if truth["status"] == "multiple_optima" and claim["status"] != "multiple":
                msg += " Note: this problem has alternative optima; the answer gives one of them without saying so."
        rep.add(group, "Final answer", "correct", message=msg)
        if truth["status"] == "multiple_optima" and claim["status"] != "multiple" and not claim["unique"]:
            rep.add(group, "Alternative optima", "incomplete",
                    message="The problem has alternative optimal solutions, which the answer does not mention.")
        return {"label": label, "claim": claim, "features": f, "details": det}
    code = FINAL_TO_CODE.get(label, "E12")
    chk = det.get("check")
    if label == "wrong_status":
        names = {"optimal": "a unique finite optimum", "multiple_optima": "multiple optimal solutions",
                 "infeasible": "no feasible solution", "unbounded": "an unbounded objective"}
        said = {"optimal": "a finite optimum", "multiple": "multiple optima", "infeasible": "infeasible",
                "unbounded": "unbounded"}.get(claim["status"], claim["status"])
        rep.add(group, "Problem status", "incorrect", "E14" if truth["status"] in ("infeasible", "unbounded") or
                claim["status"] in ("infeasible", "unbounded") else "E13",
                f"The answer says the problem is {said}, but it has {names[truth['status']]}.", ai=said,
                correct=names[truth["status"]])
    elif label == "missed_multiple_optima":
        rep.add(group, "Uniqueness of the optimum", "incorrect", "E13",
                "The answer calls the optimum unique, but alternative optimal solutions exist (a non-basic variable has "
                "Cj - Zj = 0 / the objective line is parallel to a binding constraint).", ai="unique", correct="multiple optima")
    elif label == "infeasible_solution":
        bad = [r for r in chk["rows"] if not r["ok"]] if chk else []
        rep.add(group, "Final solution", "incorrect", "E10",
                "The reported solution violates " + (", ".join(r["text"] for r in bad[:3]) or "a constraint") + ".",
                ai=_claim_pt(claim, vs), correct=_truth_pt(truth, vs))
    elif label == "suboptimal_solution":
        rep.add(group, "Final solution", "incorrect", "E13",
                f"The reported solution is feasible but not optimal" + (f": it gives Z = {chk['objective']:g}" if chk else "") +
                f", while the optimum is Z = {truth['value']:g}.", ai=_claim_pt(claim, vs), correct=_truth_pt(truth, vs))
    elif label == "wrong_value":
        rep.add(group, "Objective value", "incorrect", "E12",
                f"The optimal value is Z = {truth['value']:g}.", ai=claim["value"].v if claim["value"] else None,
                correct=f"{truth['value']:g}")
    return {"label": label, "claim": claim, "features": f, "details": det}


def _claim_pt(claim, vs):
    p = claim.get("point") or {}
    return ", ".join(f"{v} = {p[v].v:g}" for v in vs if v in p) or None


def _truth_pt(truth, vs):
    s = truth.get("solution")
    return ", ".join(f"{v} = {s[v]:g}" for v in vs) if s else None


# ----------------------------------------------------------------------------- summary
def summarize(rep, final_label):
    counts = {k: 0 for k in ("correct", "incorrect", "consequential", "deviation", "unverifiable", "incomplete")}
    for s in rep.steps:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    final_ok = final_label == "correct"
    any_bad = counts["incorrect"] > 0
    amber = counts["unverifiable"] + counts["incomplete"]
    if final_label == "unclear":
        verdict, text = "unverifiable", "No final answer found"
    elif not final_ok:
        verdict, text = "incorrect", "Final answer is wrong"
    elif any_bad:
        verdict, text = "incorrect_working", "Correct final answer, but the working has errors"
    elif amber:
        verdict, text = "partially_verified", "Final answer correct; some steps could not be verified"
    elif counts["deviation"]:
        verdict, text = "verified", "Correct; the working uses a valid path that differs from the textbook rule"
    else:
        verdict, text = "verified", "Every checked step and the final answer are correct"
    errors = [s for s in rep.steps if s["status"] == "incorrect"]
    first = errors[0] if errors else None
    return {
        "verdict": verdict, "verdict_text": text, "counts": counts,
        "correct_steps": counts["correct"], "incorrect_steps": counts["incorrect"],
        "unverifiable_steps": amber, "dependent_steps": counts["consequential"], "deviations": counts["deviation"],
        "first_error": None if not first else {"group": first["group"], "title": first["title"], "code": first.get("code")},
    }


# ----------------------------------------------------------------------------- entry point
def verify(prob, method, answer_text, ref, truth):
    """Verify an AI answer.

    prob   : parsed problem (lp_parser)
    method : graphical | simplex | bigm | two_phase
    ref    : MathGuard's own solution for that method (graphical_solver / *_solver result)
    truth  : summary dict (solver.summarize) used for the final-answer check
    """
    rep = Report()
    vs = prob["variables"]
    extra = []
    if method in ("simplex", "bigm", "two_phase") and ref is not None:
        extra = ref["standard_form"]["columns"]
    parsed = parse_solution(answer_text, vs, extra)
    for iss in parsed.issues:
        rep.notes.append(iss)
    check_setup(rep, prob, parsed)
    if method in ("simplex", "bigm", "two_phase") and ref is not None:
        std = {"variables": vs, "columns": ref["standard_form"]["columns"], "kinds": ref["standard_form"]["kinds"]}
        from .tableau import standard_form
        try:
            sf = standard_form(prob, "simplex" if method == "simplex" else "bigm")
            check_standard_form(rep, prob, parsed, sf)
        except Exception:  # noqa: BLE001
            pass
        if parsed.tableaus:
            verify_tableaus(rep, prob, method, ref, parsed, None)
        else:
            verify_claims_only(rep, ref, parsed)
    elif method == "graphical" and ref is not None:
        verify_graphical(rep, prob, ref, parsed)
    fin = check_final(rep, prob, truth, answer_text)
    summary = summarize(rep, fin["label"])
    return {"steps": rep.steps, "tableaus": rep.tableaus, "notes": rep.notes, "summary": summary,
            "final": {"label": fin["label"]}, "claim": fin["claim"], "features": fin["features"], "details": fin["details"],
            "parsed": {"tableaus": len(parsed.tableaus), "claims": len(parsed.claims),
                       "corner_points": sum(len(c["points"]) for c in parsed.corner_list),
                       "evaluations": len(parsed.evaluations), "issues": parsed.issues}}
