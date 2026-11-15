"""Structured parsing of a complete AI-written solution.

``answer_parser.parse_answer`` extracts only the final claim.  This module reads
the *working*:

* tables written as Markdown pipes, LaTeX ``array``/``tabular`` blocks or plain
  whitespace-aligned text, interpreted as simplex tableaus (Basis / CB / variable
  columns / RHS / Ratio, with Cj, Zj and Cj - Zj rows) or as corner-point tables;
* step statements: entering / leaving variables, pivot element, ratio
  calculations, row operations, optimality / unboundedness / infeasibility
  conclusions, phase markers;
* the restated objective function, constraints and standard-form equations;
* graphical-method statements: boundary equations, intercepts, intersection
  points, corner-point lists and objective evaluations.

The parser never guesses: anything it cannot read unambiguously is recorded in
``issues`` and later reported as *unverifiable* rather than marked wrong.
Positions (character offsets) are kept so that statements can be attached to
the tableau they belong to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .exact import clean_math_text, parse_number_tol
from .lp_parser import norm, parse_linear, ParseError

VAR = r"[A-Za-z]\d*'?"
_VAR_FULL = re.compile(r"^[A-Za-z]\d{0,3}$")


def normalise(text: str) -> str:
    t = norm(text or "").replace("\u00a0", " ").replace("\r", "")
    t = _latex_tables(t)
    t = clean_math_text(t)
    t = re.sub(r"\\(?:le|leq|leqslant)\b", "<=", t)
    t = re.sub(r"\\(?:ge|geq|geqslant)\b", ">=", t)
    t = re.sub(r"\\(?:to|rightarrow|longrightarrow)\b", "->", t)
    t = re.sub(r"\\(?:leftarrow|gets)\b", "<-", t)
    t = re.sub(r"\\(?:Delta)\b", "Δ", t)
    t = re.sub(r"\\(?:theta)\b", "θ", t)
    t = t.replace("→", "->").replace("←", "<-").replace("⟶", "->")
    # subscripts: x_1, x_{1}, C_j, Z_{j}, C_B  ->  x1, Cj, CB
    t = re.sub(r"\b([A-Za-z])_\{?\s*([A-Za-z0-9]+)\s*\}?", r"\1\2", t)
    t = t.replace("\\\\", "\n").replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    t = re.sub(r"\\(?:quad|qquad|;|:|,)", " ", t)
    t = re.sub(r"\\(?:begin|end)\{(?:aligned|align\*?|cases|equation\*?|gather\*?)\}", "\n", t)
    t = t.replace("&", " ")
    return t


def _latex_tables(t: str) -> str:
    def conv(m):
        body = m.group(3)
        body = re.sub(r"\\(?:hline|midrule|toprule|bottomrule)|\\cline\{[^}]*\}", "", body)
        rows = [r for r in re.split(r"\\\\", body) if r.strip()]
        out = []
        for r in rows:
            cells = [c.strip() for c in r.split("&")]
            out.append("| " + " | ".join(cells) + " |")
        return "\n" + "\n".join(out) + "\n"
    return re.sub(r"\\begin\{(array|tabular|matrix|bmatrix|pmatrix|vmatrix)\}(\{[^}]*\})?(.*?)\\end\{\1\}", conv, t, flags=re.S)


def norm_var(s: str) -> str:
    s = re.sub(r"[\s_{}\\*`$]", "", s or "")
    s = s.replace("’", "'")
    return s


# ----------------------------------------------------------------------------- data classes
@dataclass
class Cell:
    raw: str
    value: object = None
    tol: object = 0

    @property
    def ok(self):
        return self.value is not None


def _cell(raw):
    raw = (raw or "").strip().strip("*`").strip()
    v, tol = parse_number_tol(raw)
    return Cell(raw, v, tol)


@dataclass
class AITableau:
    pos: int
    end: int
    columns: list            # AI column names (variable columns only, normalised)
    rows: list               # [{"basis": str, "cb": Cell|None, "cells": {col: Cell}, "rhs": Cell|None, "ratio": Cell|None}]
    cj: dict = field(default_factory=dict)       # col -> Cell
    zj: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)    # col -> Cell, already converted to the Cj - Zj convention
    delta_label: str = ""
    delta_sign: int = 1
    z: Cell | None = None
    has_ratio: bool = False
    phase: str = ""
    raw: list = field(default_factory=list)

    def basis(self):
        return [r["basis"] for r in self.rows]


@dataclass
class Claim:
    kind: str     # entering | leaving | pivot | ratio | row_op | conclusion | phase
    pos: int
    value: object
    text: str
    extra: dict = field(default_factory=dict)


@dataclass
class ParsedSolution:
    text: str
    tables: list
    tableaus: list
    corner_tables: list
    claims: list
    objective: dict | None
    constraints: list
    std_equations: list
    equations: list
    points: list
    corner_list: list
    evaluations: list
    intersections: list
    issues: list


# ----------------------------------------------------------------------------- table extraction
_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def extract_tables(t: str):
    lines = t.split("\n")
    offs, o = [], 0
    for ln in lines:
        offs.append(o)
        o += len(ln) + 1
    tables = []
    i = 0
    while i < len(lines):
        if lines[i].count("|") >= 2:
            j = i
            rows = []
            while j < len(lines) and (lines[j].count("|") >= 2 or _SEP.match(lines[j])):
                if not _SEP.match(lines[j]):
                    cells = lines[j].strip()
                    if cells.startswith("|"):
                        cells = cells[1:]
                    if cells.endswith("|"):
                        cells = cells[:-1]
                    rows.append([c.strip() for c in cells.split("|")])
                j += 1
            if len(rows) >= 2:
                tables.append({"pos": offs[i], "end": offs[j - 1] + len(lines[j - 1]), "rows": rows, "source": "pipe"})
            i = j
            continue
        # plain whitespace table
        toks = _plain_tokens(lines[i])
        if toks and _plain_like(toks):
            j = i
            block = []
            while j < len(lines):
                tk = _plain_tokens(lines[j])
                if not tk or not _plain_like(tk):
                    break
                block.append(lines[j])
                j += 1
            if len(block) >= 3:
                tables.append({"pos": offs[i], "end": offs[j - 1] + len(lines[j - 1]), "rows": _align_plain(block),
                               "source": "plain"})
                i = j
                continue
        i += 1
    return tables


def _spans(line):
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+(?: \S+)*?(?=\s{2,}|\t|$)", line.rstrip())] \
        if re.search(r"\S {2,}\S|\t", line) else [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", line)]


def _align_plain(block):
    """Put each token of a whitespace-aligned table under the header column it sits beneath,
    so empty cells (e.g. a blank CB entry in the Cj - Zj row) do not shift the row."""
    header_k = max(range(len(block)), key=lambda k: sum(1 for _, _, t in _spans(block[k]) if _VAR_FULL.match(norm_var(t))))
    hdr = _spans(block[header_k])
    out = []
    for k, line in enumerate(block):
        sp = _spans(line)
        if k == header_k or len(sp) == len(hdr):
            out.append([t for _, _, t in sp])
            continue
        row = [""] * len(hdr)
        for a, b, t in sp:
            mid = (a + b) / 2
            best = min(range(len(hdr)), key=lambda c: (0 if hdr[c][0] - 1 <= mid <= hdr[c][1] + 1 else 1,
                                                       abs((hdr[c][0] + hdr[c][1]) / 2 - mid)))
            if row[best]:
                row[best] += " " + t
            else:
                row[best] = t
        out.append(row)
    return out


def _plain_tokens(line):
    s = line.strip()
    if not s or len(s) > 300:
        return None
    if "\t" in s or re.search(r"\S {2,}\S", s):
        toks = [x for x in re.split(r"\t+| {2,}", s) if x.strip()]
    else:
        toks = s.split()
    return toks if len(toks) >= 4 else None


_LABELS = re.compile(r"(?i)^(basis|basic|bv|b\.v\.?|cb|c_?b|rhs|b|solution|ratio|θ|cj|zj|cj-zj|zj-cj|Δj?|z|xb|min|quantity|values?)$")


def _plain_like(toks):
    nums = sum(1 for x in toks if _cell(x).ok or x in ("-", "—", "∞", "--"))
    names = sum(1 for x in toks if _VAR_FULL.match(norm_var(x)) or _LABELS.match(norm_var(x)) or _rowlabel(x))
    if names >= max(3, len(toks) - 1):
        return True                       # header line
    return nums >= len(toks) - 2 and nums >= 3  # data line (label + numbers)


def _rowlabel(x):
    return bool(re.match(r"(?i)^(c_?j|z_?j|c_?j\s*-\s*z_?j|z_?j\s*-\s*c_?j|Δ_?j?|net)", norm_var(x)))


# ----------------------------------------------------------------------------- tableau interpretation
_H_BASIS = re.compile(r"(?i)^(basis|basic(var(iable)?s?)?|b\.?v\.?|bv|base|basicvariables?|variablesinbasis|var|cbasis|basisvariable)$")
_H_CB = re.compile(r"(?i)^(cb|cbi|c\(b\)|cost|costofbasis|cbj)$")
_H_RHS = re.compile(r"(?i)^(rhs|b|bi|solution|sol\.?|soln|values?|xb|quantity|qty|constants?|righthandside|b-values?|bvalue|xbi|sol)$")
_H_RATIO = re.compile(r"(?i)(ratio|θ|theta|^min|b/a|xb/|replacement|minratio)")
_R_CJ = re.compile(r"(?i)^(cj|c|cost|cj→|cj->|objectivecoefficients?)$")
_R_ZJ = re.compile(r"(?i)^(zj|z_j)$")
_R_DELTA = re.compile(r"(?i)^(cj-zj|cj−zj|Δj?|delta|neteval\w*|netevaluation|c̄j|cbarj|reducedcosts?|cj-zjrow|δj)$")
_R_DELTA_NEG = re.compile(r"(?i)^(zj-cj|zj−cj|Δj\(zj-cj\))$")
_R_ZROW = re.compile(r"(?i)^(z|obj|objective|zrow|z-row|row0)$")


def interpret_tableau(tb, decision):
    rows = [[c for c in r] for r in tb["rows"]]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    dec_low = {d.lower() for d in decision}

    def varlike(c):
        n = norm_var(c)
        if n.lower() in dec_low:
            return True
        return bool(_VAR_FULL.match(n)) and not _H_RHS.match(n) and not _H_CB.match(n) and n.lower() not in ("z", "c", "b")

    # header row = row with the most variable-like cells, also accepting a header that holds the RHS keyword
    best, bi = -1, None
    for k, r in enumerate(rows[:4]):
        score = sum(1 for c in r if varlike(c))
        if any(_H_RHS.match(norm_var(c)) or _H_BASIS.match(norm_var(c)) for c in r):
            score += 1
        if score > best and sum(1 for c in r if varlike(c)) >= 2:
            best, bi = score, k
    if bi is None:
        return None
    header = [norm_var(c) for c in rows[bi]]
    kinds = []
    for k, h in enumerate(header):
        if _H_BASIS.match(h):
            kinds.append("basis")
        elif _H_CB.match(h):
            kinds.append("cb")
        elif _H_RATIO.search(h):
            kinds.append("ratio")
        elif _H_RHS.match(h):
            kinds.append("rhs")
        elif h.lower() == "z" and "z" not in dec_low:
            kinds.append("zcol")
        elif varlike(rows[bi][k]):
            kinds.append("var")
        else:
            kinds.append("label" if k <= 1 else "other")
    if "basis" not in kinds:
        # first label-ish column holding variable names in the body becomes the basis column
        for k in range(min(2, width)):
            body = [norm_var(r[k]) for r in rows[bi + 1:] if r[k].strip()]
            if body and sum(1 for b in body if _VAR_FULL.match(b)) >= max(1, len(body) // 2) and kinds[k] in ("label", "var", "other"):
                kinds[k] = "basis"
                break
    if "basis" not in kinds:
        return None
    var_cols = [k for k, kd in enumerate(kinds) if kd == "var"]
    if "rhs" not in kinds:
        # last non-ratio numeric column after the variable columns is the RHS (e.g. unlabeled)
        cand = [k for k in range(var_cols[-1] + 1, width) if kinds[k] in ("other", "label")]
        if cand:
            kinds[cand[0]] = "rhs"
        elif header and kinds[-1] == "var" and _is_rhs_name(header[-1], decision):
            kinds[-1] = "rhs"
            var_cols = var_cols[:-1]
    bcol = kinds.index("basis")
    cbcol = kinds.index("cb") if "cb" in kinds else None
    rcol = kinds.index("rhs") if "rhs" in kinds else None
    tcol = kinds.index("ratio") if "ratio" in kinds else None
    columns = [header[k] for k in var_cols]
    out = AITableau(pos=tb["pos"], end=tb["end"], columns=columns, rows=[], raw=tb["rows"])
    out.has_ratio = tcol is not None
    for k, r in enumerate(rows):
        if k == bi:
            continue
        labels = [norm_var(r[x]) for x in range(min(2, width))]
        label = norm_var(r[bcol])
        lab_any = " ".join(l for l in labels if l)
        lab_txt = re.sub(r"\s+", "", lab_any)
        cells = {header[x]: _cell(r[x]) for x in var_cols}
        if _R_CJ.match(label) or _R_CJ.match(lab_txt) or any(_R_CJ.match(l) for l in labels if l):
            if sum(1 for c in cells.values() if c.ok) >= 2:
                out.cj = cells
            continue
        if _R_ZJ.match(label) or any(_R_ZJ.match(l) for l in labels if l):
            out.zj = cells
            if rcol is not None and _cell(r[rcol]).ok:
                out.z = _cell(r[rcol])
            continue
        if _R_DELTA_NEG.match(label) or any(_R_DELTA_NEG.match(l) for l in labels if l):
            out.delta = {c: Cell(v.raw, -v.value if v.ok else None, v.tol) for c, v in cells.items()}
            out.delta_label, out.delta_sign = "Zj - Cj", -1
            continue
        if _R_DELTA.match(label) or any(_R_DELTA.match(l) for l in labels if l):
            out.delta = cells
            out.delta_label, out.delta_sign = "Cj - Zj", 1
            continue
        if _R_ZROW.match(label) and label.lower() not in dec_low:
            out.delta = {c: Cell(v.raw, -v.value if v.ok else None, v.tol) for c, v in cells.items()}
            out.delta_label, out.delta_sign = "z-row (Zj - Cj)", -1
            if rcol is not None and _cell(r[rcol]).ok:
                out.z = _cell(r[rcol])
            continue
        if not _VAR_FULL.match(label):
            continue
        if not any(c.ok for c in cells.values()):
            continue
        out.rows.append({
            "basis": label, "cb": _cell(r[cbcol]) if cbcol is not None else None, "cells": cells,
            "rhs": _cell(r[rcol]) if rcol is not None else None, "ratio": _cell(r[tcol]) if tcol is not None else None,
            "ratio_raw": r[tcol].strip() if tcol is not None else "",
        })
    # a Cj row given above the header (common layout: "Cj | | 3 | 5 | 0 | 0 |")
    if not out.cj and bi > 0:
        for r in rows[:bi]:
            if any(_R_CJ.match(norm_var(x)) for x in r[:2]):
                cells = {header[x]: _cell(r[x]) for x in var_cols}
                if sum(1 for c in cells.values() if c.ok) >= 2:
                    out.cj = cells
    if not out.rows or rcol is None:
        return None
    return out


def _is_rhs_name(h, decision):
    return h.lower() in ("b", "rhs", "xb")


def interpret_corner_table(tb, decision):
    """Tables like | Corner | (x, y) | Z | or | Point | x | y | Z = 3x + 2y |."""
    rows = tb["rows"]
    if len(rows) < 2 or len(decision) != 2:
        return None
    header = [h.strip() for h in rows[0]]
    hl = [norm_var(h).lower() for h in header]
    zc = next((k for k, h in enumerate(hl) if h.startswith("z") or "objective" in h or "value" in h or "profit" in h or "cost" in h), None)
    if zc is None:
        return None
    vx = next((k for k, h in enumerate(hl) if h == decision[0].lower()), None)
    vy = next((k for k, h in enumerate(hl) if h == decision[1].lower()), None)
    out = []
    for r in rows[1:]:
        r = r + [""] * (len(header) - len(r))
        pt = None
        if vx is not None and vy is not None:
            a, b = _cell(r[vx]), _cell(r[vy])
            if a.ok and b.ok:
                pt = (a, b)
        else:
            for c in r:
                m = re.search(r"\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)", clean_math_text(c))
                if m:
                    a, b = _cell(m.group(1)), _cell(m.group(2))
                    if a.ok and b.ok:
                        pt = (a, b)
                        break
        zcell = r[zc]
        zv = _last_number(zcell)
        if pt is not None:
            out.append({"point": pt, "z": zv, "raw": " | ".join(r)})
    return out or None


def _last_number(s):
    s = clean_math_text(s or "")
    if "=" in s:
        s = s.split("=")[-1]
    ms = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", s.replace(",", ""))
    if not ms:
        return None
    c = _cell(ms[-1])
    return c if c.ok else None


# ----------------------------------------------------------------------------- statements (claims)
_VARX = r"([A-Za-z]\d{0,3})"
_ENTER = [
    re.compile(r"(?i)(?:entering|incoming)\s+variable\s*(?:is|=|:|-|->)?\s*\**\s*" + _VARX + r"\b"),
    re.compile(r"(?i)\b" + _VARX + r"\**\s+(?:enters|is\s+the\s+entering|will\s+enter|should\s+enter|can\s+enter|comes\s+in(?:to)?|is\s+entering|enters\s+the\s+basis)\b"),
    re.compile(r"(?i)(?:key|pivot)\s+column\s*(?:is|=|:)?\s*(?:the\s+)?(?:column\s+of\s+|under\s+)?\(?\s*" + _VARX + r"\b"),
    re.compile(r"(?i)\benter(?:ing)?\s*:\s*" + _VARX + r"\b"),
    re.compile(r"(?i)\bbring\s+" + _VARX + r"\s+into\s+the\s+basis"),
]
_LEAVE = [
    re.compile(r"(?i)(?:leaving|outgoing|departing|exiting)\s+variable\s*(?:is|=|:|-|->)?\s*\**\s*" + _VARX + r"\b"),
    re.compile(r"(?i)\b" + _VARX + r"\**\s+(?:leaves|is\s+the\s+leaving|will\s+leave|should\s+leave|departs|exits|goes\s+out|is\s+leaving|leaves\s+the\s+basis|drops\s+out)\b"),
    re.compile(r"(?i)(?:key|pivot)\s+row\s*(?:is|=|:)?\s*(?:the\s+)?(?:row\s+of\s+)?\(?\s*" + _VARX + r"\b(?!\s*\d)"),
    re.compile(r"(?i)\bleav(?:e|ing)\s*:\s*" + _VARX + r"\b"),
    re.compile(r"(?i)\breplac(?:e|ing)\s+" + _VARX + r"\s+(?:by|with)\s+[A-Za-z]\d{0,3}\b"),
]
_REPLACE_BY = re.compile(r"(?i)\breplac(?:e|ing)\s+" + _VARX + r"\s+(?:by|with)\s+" + _VARX + r"\b")
_PIVOT = re.compile(r"(?i)(?:pivot|key)\s+(?:element|entry|number|value)\s*(?:is|=|:|-)?\s*(?:the\s+)?(?:entry\s+)?(-?\d+(?:\.\d+)?(?:/\d+)?)")
_RATIO = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)")
_ROWOP = re.compile(r"(?i)\bR\s*(\d)\s*(?:\(new\)|'|\s*new|\s*\(n\))?\s*(?:->|<-|:=|=)\s*([^\n;,]{2,80})")
_PHASE = re.compile(r"(?i)\bphase\s*[-:]?\s*(ii|2|two|i|1|one)\b")


def extract_claims(t, known_vars):
    known = {v.lower() for v in known_vars}
    claims = []

    def ok(v):
        n = norm_var(v)
        return n.lower() in known or bool(re.fullmatch(r"[A-Za-z]\d{1,3}", n))

    for pats, kind in ((_ENTER, "entering"), (_LEAVE, "leaving")):
        for p in pats:
            for m in p.finditer(t):
                v = norm_var(m.group(1))
                if not ok(v):
                    continue
                claims.append(Claim(kind, m.start(), v, m.group(0)))
    for m in _REPLACE_BY.finditer(t):
        a, b = norm_var(m.group(1)), norm_var(m.group(2))
        if ok(a) and ok(b):
            claims.append(Claim("entering", m.start(), b, m.group(0)))
    for m in _PIVOT.finditer(t):
        c = _cell(m.group(1))
        if c.ok:
            claims.append(Claim("pivot", m.start(), c, m.group(0)))
    for m in _RATIO.finditer(t):
        a, b, r = _cell(m.group(1)), _cell(m.group(2)), _cell(m.group(3))
        if a.ok and b.ok and r.ok:
            pre = t[max(0, m.start() - 24): m.start()]
            mv = re.findall(r"([A-Za-z]\d{0,3})\s*[:=(]?\s*$", pre)
            row = norm_var(mv[-1]) if mv else None
            line = t[t.rfind("\n", 0, m.start()) + 1: m.start()].lower()
            if "ratio" in line or "θ" in line or row or "min" in line:
                claims.append(Claim("ratio", m.start(), (a, b, r), m.group(0), {"row": row}))
    for m in _ROWOP.finditer(t):
        claims.append(Claim("row_op", m.start(), (int(m.group(1)), m.group(2).strip()), m.group(0)))
    for m in _PHASE.finditer(t):
        ph = "II" if m.group(1).lower() in ("ii", "2", "two") else "I"
        claims.append(Claim("phase", m.start(), ph, m.group(0)))
    # conclusions about the tableau / problem
    for m in re.finditer(r"(?i)(optimal(?:ity)?\s+(?:condition|tableau|solution)?\s*(?:is\s+)?(?:reached|attained|achieved|satisfied|obtained)|"
                         r"(?:current|this)\s+(?:tableau|solution)\s+is\s+optimal|all\s+(?:the\s+)?(?:cj\s*-\s*zj|Δj?|net\s+evaluations?)"
                         r"[^.\n]{0,30}(?:<=|≤|non-?positive|>=|≥|non-?negative|negative|positive|zero)|"
                         r"no\s+(?:positive|negative)\s+(?:cj\s*-\s*zj|Δj?|net evaluation)|hence\s+optimal|is\s+the\s+optimal\s+tableau)", t):
        claims.append(Claim("conclusion", m.start(), "optimal", m.group(0)))
    for m in re.finditer(r"(?i)(unbounded|no\s+finite\s+(?:optimum|optimal|solution))", t):
        if not re.search(r"(?i)\b(not|isn't|no)\s+(?:be\s+)?unbounded", t[max(0, m.start() - 20): m.end()]):
            claims.append(Claim("conclusion", m.start(), "unbounded", m.group(0)))
    for m in re.finditer(r"(?i)(infeasible|no\s+feasible\s+solution|artificial\s+variable[^.\n]{0,40}(?:remains|still)[^.\n]{0,30}positive)", t):
        if not re.search(r"(?i)\b(not|isn't)\s+(?:be\s+)?infeasible", t[max(0, m.start() - 20): m.end()]):
            claims.append(Claim("conclusion", m.start(), "infeasible", m.group(0)))
    for m in re.finditer(r"(?i)(alternat(?:e|ive)\s+optim\w*|multiple\s+optim\w*|infinitely\s+many\s+(?:optimal\s+)?solutions)", t):
        if not re.search(r"(?i)\bno\s+(?:alternat|multiple)", t[max(0, m.start() - 6): m.end()]):
            claims.append(Claim("conclusion", m.start(), "multiple", m.group(0)))
    claims.sort(key=lambda c: c.pos)
    # de-duplicate the same claim found by two patterns at nearly the same place
    out = []
    for c in claims:
        if out and out[-1].kind == c.kind and abs(out[-1].pos - c.pos) < 25 and str(out[-1].value) == str(c.value):
            continue
        out.append(c)
    return out


# ----------------------------------------------------------------------------- restated problem / equations
_OBJ = re.compile(r"(?i)\b(max(?:imi[sz]e)?|min(?:imi[sz]e)?)[ \t]*:?[ \t]*(?:Z|z|P|C|W)?[ \t]*=[ \t]*([^\n;,]{1,120})")
_TERM = r"(?:\d+(?:\.\d+)?(?:/\d+)?[ \t]?\*?[ \t]?[A-Za-z]\d*|[A-Za-z]\d*)(?![A-Za-z(])"
_LHS = r"[+-]?[ \t]*" + _TERM + r"(?:[ \t]*[+-][ \t]*" + _TERM + r")*"
_INEQ = re.compile(r"(?<![\w.<>=/])(" + _LHS + r")[ \t]*(<=|>=|=)[ \t]*(-?\d+(?:\.\d+)?(?:/\d+)?)(?![\d\w(]|\.\d)")


def restated_objective(t, decision):
    for m in _OBJ.finditer(t):
        expr = m.group(2).strip()
        expr = re.split(r"\s+(?:subject|s\.t|where|with)\b", expr)[0]
        if re.search(r"[A-Za-z]{2,}", re.sub(r"[A-Za-z]\d+", "", expr)):
            continue
        try:
            coefs, const = parse_linear(expr.replace(" ", ""))
        except ParseError:
            continue
        extra = {k: v for k, v in coefs.items() if k not in decision}
        if any(v != 0 for v in extra.values()):
            continue
        coefs = {k: v for k, v in coefs.items() if k in decision}
        if not coefs:
            continue
        return {"sense": "max" if m.group(1).lower().startswith("max") else "min", "coefs": coefs, "pos": m.start(), "text": m.group(0).strip()}
    return None


def linear_relations(t, decision):
    """All 'linear expression  op  number' statements, split into
    (a) restated constraints in the decision variables, (b) standard-form equations with slack/artificial
    variables, and (c) boundary equations (decision variables only, '=')."""
    dec = set(decision)
    cons, std, eqs = [], [], []
    for m in _INEQ.finditer(t):
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        try:
            coefs, const = parse_linear(lhs.replace(" ", ""))
        except ParseError:
            continue
        if not coefs or const != 0:
            continue
        rv = _cell(rhs)
        if not rv.ok:
            continue
        names = set(coefs)
        if op == "=" and len(names) == 1:
            # skip assignments that are part of a stated point "x = 2, y = 3" / "x1 = 2 and x2 = 3"
            after = t[m.end(): m.end() + 30]
            before = t[max(0, m.start() - 30): m.start()]
            v0 = next(iter(names))
            ma = re.match(r"\s*(?:,|and)\s*([A-Za-z]\d*)\s*=\s*-?\d[\d./]*(?![\d./]*\s*[+\-*/]\s*\d*[A-Za-z]|[\d./]*[A-Za-z])", after)
            mb = re.search(r"(?:^|[^+\-*/.=<>\d\s])\s+([A-Za-z]\d*)\s*=\s*-?[\d./]+\s*(?:,|and)\s*$", before)
            if (ma and ma.group(1) != v0) or (mb and mb.group(1) != v0):
                continue
        line_no = t.count("\n", 0, m.start())
        item = {"coefs": coefs, "op": op, "rhs": rv, "pos": m.start(), "text": m.group(0).strip(), "line_no": line_no}
        if names <= dec:
            if op == "=":
                eqs.append(item)
            else:
                cons.append(item)
        elif op == "=" and names & dec and all(re.fullmatch(r"[A-Za-z]\d+", n) for n in names - dec):
            std.append(item)
    return cons, std, eqs


# ----------------------------------------------------------------------------- graphical statements
_PT = re.compile(r"(?<![A-Za-z0-9_])([A-Z]\d?)?\s*\(\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*\)")
_CORNER_WORDS = re.compile(r"(?i)corner|vertic|vertex|extreme\s+point|feasible\s+points|feasible\s+region\s+(?:is|has)|boundary\s+points")


def graphical_statements(t, decision):
    points, corner_list, evaluations, intersections = [], [], [], []
    if len(decision) != 2:
        return points, corner_list, evaluations, intersections
    x, y = decision
    lines = t.split("\n")
    off = 0
    for line_no, ln in enumerate(lines):
        pts = [(m, _cell(m.group(2)), _cell(m.group(3))) for m in _PT.finditer(ln)]
        # "x = 2, y = 3" style points
        for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(x) + r"\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*(?:,|and)\s*" + re.escape(y)
                             + r"\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)(?![\d.])", ln):
            pts.append((m, _cell(m.group(1)), _cell(m.group(2))))
        pts = [(m, a, b) for m, a, b in pts if a.ok and b.ok]
        for m, a, b in pts:
            points.append({"pos": off + m.start(), "point": (a, b), "line": ln.strip(), "line_no": line_no,
                           "label": (m.group(1) if m.re is _PT else None)})
        low = ln.lower()
        if _CORNER_WORDS.search(ln) and (len(pts) >= 2 or re.search(r"(?i)corner\s+points?|vertices", ln) and pts) \
                and not re.search(r"(?i)\bz\s*\(|evaluat", ln):
            corner_list.append({"pos": off, "points": [(a, b) for _, a, b in pts], "line": ln.strip()})
        # objective evaluations: "Z(2,2) = 10", "At (2,2): Z = 3(2) + 2(2) = 10", "A(2, 2) -> Z = 10"
        for k, (m, a, b) in enumerate(pts):
            seg_end = pts[k + 1][0].start() if k + 1 < len(pts) else len(ln)
            seg = ln[m.end(): seg_end]
            pre = ln[max(0, m.start() - 3): m.start()]
            if re.search(r"(?i)\bz\b|value|profit|cost|objective", seg) or pre.strip().lower().endswith("z"):
                v = _last_number(re.split(r"(?i)(?:;|\bat\b)", seg)[0]) if "=" in seg else None
                if v is not None:
                    evaluations.append({"pos": off + m.start(), "point": (a, b), "value": v, "line": ln.strip(), "line_no": line_no})
        if re.search(r"(?i)intersect|solv|meet|simultaneous|common point|point of intersection", low) and pts \
                and not re.search(r"(?i)\baxes\b|\baxis\b|intercept", low):
            intersections.append({"pos": off, "line": ln.strip(), "line_no": line_no, "points": [(a, b) for _, a, b in pts]})
        off += len(ln) + 1
    return points, corner_list, evaluations, intersections


# ----------------------------------------------------------------------------- entry point
def parse_solution(text, decision, extra_vars=()):
    t = normalise(text)
    issues = []
    tables = extract_tables(t)
    tableaus, corner_tables = [], []
    for tb in tables:
        tab = interpret_tableau(tb, decision)
        if tab is not None:
            tableaus.append(tab)
            continue
        ct = interpret_corner_table(tb, decision)
        if ct:
            corner_tables.append({"pos": tb["pos"], "rows": ct})
        elif len(tb["rows"]) >= 2:
            issues.append(f"A table starting with '{' | '.join(tb['rows'][0])[:60]}' could not be read as a simplex tableau or a corner-point table.")
    claims = extract_claims(t, list(decision) + list(extra_vars))
    # phases
    phase_marks = [c for c in claims if c.kind == "phase"]
    for tab in tableaus:
        before = [c for c in phase_marks if c.pos <= tab.pos]
        tab.phase = before[-1].value if before else ""
    obj = restated_objective(t, decision)
    cons, std, eqs = linear_relations(t, decision)
    pts, corner_list, evals, inters = graphical_statements(t, decision)
    for ct in corner_tables:
        for r in ct["rows"]:
            if r["z"] is not None:
                evals.append({"pos": ct["pos"], "point": r["point"], "value": r["z"], "line": r["raw"]})
        corner_list.append({"pos": ct["pos"], "points": [r["point"] for r in ct["rows"]], "line": "(corner-point table)"})
    return ParsedSolution(text=t, tables=tables, tableaus=tableaus, corner_tables=corner_tables, claims=claims,
                          objective=obj, constraints=cons, std_equations=std, equations=eqs, points=pts,
                          corner_list=corner_list, evaluations=evals, intersections=inters, issues=issues)
