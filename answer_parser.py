"""Extract the final claim (status, solution point, objective value) from a free-text AI answer."""
import re
from dataclasses import dataclass
from .lp_parser import norm


@dataclass
class Num:
    v: float
    d: int = 0  # decimals typed

    @property
    def tol(self):
        return 0.5 * 10 ** (-self.d) * 1.001 if self.d > 0 else 1e-6


NUM = r"-?\d+(?:\.\d+)?(?:/\d+)?"
VNUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"


def to_num(tok):
    tok = tok.replace(",", "")
    if "/" in tok:
        a, b = tok.split("/")
        return Num(float(a) / float(b), 0)
    return Num(float(tok), len(tok.split(".")[1]) if "." in tok else 0)


_CUE = re.compile(
    r"(?i)\b(?:optimal\s+(?:solution|point|value)|optimum|final\s+answer|answer|therefore|hence|thus|"
    r"in\s+conclusion|conclusion|result)\b"
)
_NEG = re.compile(
    r"(?i)\b(?:not|isn't|is\s+not|never|neither|nor|cannot\s+be|can't\s+be|doesn't|does\s+not|no)\s+"
    r"(?:be\s+|been\s+)?(?:infeasible|unbounded)\b"
)
_INFEAS = [
    r"\b(?:is|are|be|declared|deemed|therefore|thus|hence|so)\s+(?:an\s+)?infeasible\b",
    r"\binfeasible\s+(?:problem|lp|lpp|model|solution|region|system)\b",
    r"\bno\s+feasible\s+(?:solution|region|point|set|area)\b",
    r"\bfeasible\s+(?:region|set|area)\s+is\s+empty\b",
    r"\bempty\s+feasible\b",
    r"\b(?:inconsistent|contradictory|conflicting)\s+constraints\b",
    r"\bconstraints\s+(?:are\s+)?(?:inconsistent|contradict\w*|conflict\w*)\b",
    r"\bno\s+solution\s+exists\b",
    r"\bcannot\s+be\s+(?:simultaneously\s+)?satisfied\b",
    r"\b(?:status|result|answer|conclusion|verdict)\s*[:=\-]\s*infeasible\b",
]
_UNBND = [
    r"\b(?:is|are|be|therefore|thus|hence|so)\s+unbounded\b",
    r"\bunbounded\s+(?:problem|lp|lpp|solution|objective|model|region)\b",
    r"\bno\s+finite\s+(?:optimal|optimum|maximum|minimum|solution|value)\b",
    r"\bwithout\s+(?:any\s+)?bound\b",
    r"\b(?:increases?|grows?|decreases?)\s+(?:indefinitely|infinitely|without)\b",
    r"\b(?:tends?|goes|going)\s+to\s+(?:\+|-)?\s*(?:infinity|∞)\b",
    r"\b(?:status|result|answer|conclusion|verdict)\s*[:=\-]\s*unbounded\b",
]
_MULTI = [
    r"\bmultiple\s+optim\w*",
    r"\balternat(?:e|ive)\s+optim\w*",
    r"\binfinitely\s+many\s+(?:optimal\s+)?(?:solutions|optima|points)\b",
    r"\bmany\s+optimal\b",
    r"\bnot\s+unique\b",
    r"\bnon-?unique\b",
    r"\b(?:status|result|answer|conclusion|verdict)\s*[:=\-]\s*multiple\b",
]


def _first(patterns, low):
    best = None
    for p in patterns:
        m = re.search(p, low)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.end())
    return best


def detect_status(t):
    low = _NEG.sub(" ", t.lower())
    hits = {}
    for name, pats in (("infeasible", _INFEAS), ("unbounded", _UNBND), ("multiple", _MULTI)):
        h = _first(pats, low)
        if h:
            hits[name] = h
    # bare one-word answers ("Infeasible.", "Unbounded")
    words = re.findall(r"[a-z]+", low)
    if len(words) <= 6:
        for w in ("infeasible", "unbounded"):
            if w in words and w not in hits:
                hits[w] = (0, 0)
    unique = bool(re.search(r"(?<!not )(?<!non-)(?<!non )\bunique\b", low)) and "multiple" not in hits
    if not hits:
        return None, unique, None
    name = min(hits, key=lambda k: hits[k][0])
    return name, unique, hits[name][0]


def _assign_candidates(t, variables):
    ms = []
    for v in variables:
        pat = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(v) + r"(?![A-Za-z0-9_])\s*\*?\s*(?:=|:|≈|\bis\b|\bequals\b)\s*(" + NUM + r")"
        )
        for m in pat.finditer(t):
            before = t[: m.start()].rstrip()[-1:]
            if before and before in "+-*/":
                continue
            ms.append((m.start(), m.end(), v, m.group(1)))
    ms.sort()
    groups, cur, last_end, start = [], {}, None, 0
    for s, e, v, tok in ms:
        if cur and (v in cur or s - last_end > 80):
            groups.append({"pos": start, "pt": cur})
            cur = {}
        if not cur:
            start = s
        cur[v] = to_num(tok)
        last_end = e
    if cur:
        groups.append({"pos": start, "pt": cur})
    return groups


def _tuple_candidates(t, variables):
    n = len(variables)
    if n < 2:
        return []
    pat = re.compile(r"[\(\[]\s*" + r"\s*,\s*".join(["(" + NUM + ")"] * n) + r"\s*[\)\]]")
    out = []
    for m in pat.finditer(t):
        if m.start() > 0 and (t[m.start() - 1].isalnum() or t[m.start() - 1] == "_"):
            continue  # function-call style, e.g. "Z(2, 2) = 10"
        ls = max(t.rfind("\n", 0, m.start()), t.rfind(". ", 0, m.start()), t.rfind("; ", 0, m.start())) + 1
        ends = [x for x in (t.find("\n", m.end()), t.find(". ", m.end()), t.find("; ", m.end())) if x != -1]
        le = min(ends) if ends else len(t)
        line = t[ls: le]
        if len(pat.findall(line)) >= 3 or re.search(r"(?i)corner|vertic|vertex|extreme\s+point|candidate", line):
            continue  # a list of candidate vertices, not the stated answer
        out.append({"pos": m.start(), "pt": {v: to_num(m.group(i + 1)) for i, v in enumerate(variables)}})
    return out


def _last_cue(t):
    last = -1
    for m in _CUE.finditer(t):
        last = m.start()
    return last


def find_points(t, variables, anchor=None):
    n = len(variables)
    cands = sorted(_assign_candidates(t, variables) + _tuple_candidates(t, variables), key=lambda c: c["pos"])
    full = [c for c in cands if len(c["pt"]) == n]
    cue = _last_cue(t)
    main = None
    if full:
        after = [c for c in full if c["pos"] >= cue] if cue >= 0 else []
        main = after[0] if after else full[-1]
    elif cands:
        best = max(len(c["pt"]) for c in cands)
        main = [c for c in cands if len(c["pt"]) == best][-1]
    multi = []
    if anchor is not None:
        multi = [c for c in full if c["pos"] >= anchor]
        if not multi:
            multi = [c for c in full if 0 <= anchor - c["pos"] <= 160]
    return (main["pt"] if main else {}), [c["pt"] for c in multi], [c["pt"] for c in full]


def _is_expr_rest(rest):
    return rest.startswith("(") or bool(re.match(r"\s*[\*\+/×^]", rest)) or bool(re.match(r"\s*-\s*\d", rest))


def _seg_value(seg):
    """Value from text that follows a label: a plain number, or the result after the last '=' of a chain."""
    seg = re.split(r"[;\n]|,\s|\.\s|\.$", seg)[0]
    m = re.match(r"\s*(?:Rs\.?|₹|\$)?\s*(" + VNUM + r")", seg)
    if m and re.match(r"\s*\*?\s*[A-Za-z]\d*\s*(?:[+\-]|$|\s+(?:over|subject|s\.t|where|is))", seg[m.end():]):
        return None  # "2x + 3y": an expression for Z, not its value
    if m and not _is_expr_rest(seg[m.end():]):
        return to_num(m.group(1))  # "10", "10 at x=2", "10 (max)"
    if "=" in seg:
        m = re.match(r"\s*(?:Rs\.?|₹|\$)?\s*(" + VNUM + r")", seg.split("=")[-1])
        return to_num(m.group(1)) if m else None  # "3(2)+2(2) = 10"
    return None


def find_value(t):
    cands = []  # (priority, pos, Num)
    p1 = re.compile(
        r"(?i)(?:optimal|maximum|minimum|max|min|best)\s*(?:objective\s*)?(?:function\s*)?value"
        r"(?:\s+of\s+(?:the\s+)?(?:objective(?:\s+function)?|Z))?\s*(?:is|=|:|equals|of|-)?\s*(?:Z\s*=\s*)?"
        r"(?:Rs\.?|₹|\$)?\s*(" + VNUM + ")"
    )
    p1b = re.compile(
        r"(?i)(?:max(?:imum)?|min(?:imum)?|optimal|total)?\s*(?:profit|cost|revenue|objective(?:\s+value)?)"
        r"\s*(?:is|=|:|equals|of)\s*(?:Rs\.?|₹|\$)?\s*(" + VNUM + ")"
    )
    p1c = re.compile(
        r"(?i)\b(?:max(?:imum)?|min(?:imum)?|optimal|total|least|highest|lowest)\s+(?:profit|cost|revenue|value)\s+"
        r"(?:of\s+)?(?:Rs\.?|₹|\$|INR)?\s*(" + VNUM + r")(?!\d|\.\d|\s*[A-Za-z]\d*\s*[+\-])"
    )
    for p in (p1, p1b, p1c):
        for m in p.finditer(t):
            val = _seg_value(t[m.start(1):])
            if val is not None:
                cands.append((1, m.start(), val))
    p2 = re.compile(r"\bZ(?:\s*_?\s*\{?(?:max|min|opt)\}?|\s*\*)?\s*=\s*([^\n;]*)")
    for m in p2.finditer(t):
        val = _seg_value(m.group(1))
        if val is not None:
            cands.append((2, m.start(), val))
    if not cands:
        return None
    best = min(c[0] for c in cands)
    pool = [c for c in cands if c[0] == best]
    cue = _last_cue(t)
    after = [c for c in cands if c[1] >= cue] if cue >= 0 else []
    if after:
        best_after = min(c[0] for c in after)
        return [c for c in after if c[0] == best_after][-1][2]
    return pool[-1][2]


def parse_answer(text, variables):
    t = norm(text or "").replace(" ", " ")
    t = re.sub(r"\*\*|__", " ", t)  # Markdown bold markers are not multiplication signs
    status, unique, spos = detect_status(t)
    anchor = spos if status == "multiple" else None
    point, multi_pts, _ = find_points(t, variables, anchor)
    value = None if status in ("infeasible", "unbounded") else find_value(t)
    if status in ("infeasible", "unbounded"):
        point = {}
    if status is None and (point or value is not None):
        status = "optimal"
    return {"status": status, "unique": bool(unique), "point": point, "multi_points": multi_pts, "value": value}
