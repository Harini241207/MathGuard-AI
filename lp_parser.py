"""Parse a linear program written in plain text into a structured dict.

Accepts things like:
    Maximize Z = 3x + 2y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0
    Min Z = 2x1 + 3x2
    s.t.
      x1 + x2 >= 4
      x1 + 2x2 >= 6
      x1, x2 >= 0
All variables are assumed non-negative (standard LPP form).
"""
import re
from fractions import Fraction


class ParseError(ValueError):
    pass


_NUM = r"\d+(?:\.\d+)?(?:/\d+)?"
_TERM = re.compile(r"([+-]?)(" + _NUM + r"\*?)?([A-Za-z]\d*)?")
_SENSE = re.compile(r"\b(maximi[sz]e|maximum|max|minimi[sz]e|minimum|min)\b", re.I)
_MARK = re.compile(
    r"(?i)(?:\bsubject\s+to\b|\bsuch\s+that\b|\bs\.\s?t\.?(?=\s|:|$)|\bst\b(?=\s|:)"
    r"|\bconstraints?\b\s*:?|\bwhere\b|\bgiven\b|\bunder\b)"
)


def norm(s):
    s = s.replace("≤", "<=").replace("≥", ">=")
    for a, b in (("−", "-"), ("–", "-"), ("×", "*"), ("·", "*"), ("∗", "*"), ("=<", "<="), ("=>", ">=")):
        s = s.replace(a, b)
    return s


def fmt_num(x):
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6g}"


def fmt_linear(coefs, variables):
    out = []
    for v in variables:
        k = coefs.get(v, 0)
        if abs(k) < 1e-12:
            continue
        a = abs(k)
        body = (fmt_num(a) if abs(a - 1) > 1e-12 else "") + v
        if not out:
            out.append(("-" if k < 0 else "") + body)
        else:
            out.append((" - " if k < 0 else " + ") + body)
    return "".join(out) or "0"


def parse_linear(expr):
    """'3x+2y-5' -> ({'x':3,'y':2}, -5) using Fractions."""
    s = re.sub(r"\s+", "", expr)
    if re.search(r"[()\[\]{}^]", s):
        raise ParseError(f"Parentheses/powers are not supported in '{expr.strip()}'")
    coefs, const = {}, Fraction(0)
    i = 0
    while i < len(s):
        m = _TERM.match(s, i)
        if not m or m.end() == i:
            raise ParseError(f"Could not read '{s[i:i + 12]}' in '{expr.strip()}'")
        sign, num, var = m.groups()
        if num is None and var is None:
            raise ParseError(f"Dangling sign in '{expr.strip()}'")
        k = Fraction(num.rstrip("*")) if num else Fraction(1)
        if sign == "-":
            k = -k
        if var:
            coefs[var] = coefs.get(var, 0) + k
        else:
            const += k
        i = m.end()
    return coefs, const


def parse_problem(text):
    if not text or not text.strip():
        raise ParseError("The question is empty.")
    t = norm(text).strip()
    warnings = []
    m = _SENSE.search(t)
    if not m:
        raise ParseError("Could not find 'Maximize' or 'Minimize' in the question.")
    sense = "max" if m.group(1).lower().startswith("max") else "min"
    rest = t[m.end():]
    mk = _MARK.search(rest)
    if mk:
        obj_txt, cons_txt = rest[: mk.start()], rest[mk.end():]
    else:
        parts = re.split(r"[\n;]", rest, maxsplit=1)
        obj_txt = parts[0]
        cons_txt = parts[1] if len(parts) > 1 else ""

    # ---- objective ----
    obj_txt = obj_txt.strip(" :,;\n.\t")
    if "=" in obj_txt:
        left, right = obj_txt.split("=", 1)
        if not re.search(r"[\d+\-]", left):
            obj_txt = right
    obj_txt = re.sub(r"^(?:[A-Za-z]{2,}\s+)+", "", obj_txt.strip(" :"))
    obj_txt = obj_txt.strip(" :,.")
    try:
        obj_c, _ = parse_linear(obj_txt)
    except ParseError as e:
        raise ParseError(f"Could not read the objective function: {e}")
    if not obj_c:
        raise ParseError("The objective function has no variables.")

    # ---- constraints ----
    c = re.sub(r"(?i)^(?:\s*(?:the|following|constraints?|conditions?|restrictions?|:))+", "", cons_txt)
    c = re.sub(r"\([^)]*[A-Za-z]{3,}[^)]*\)", " ", c)  # drop "(non-negativity)", "(labour hours)"
    c = re.sub(r"(?i)non-?\s*negativity(?:\s+constraints?)?|all\s+variables|\bfor\s+all\b|\bwith\b", " ", c)
    c = re.sub(r"(?i)\band\b", ";", c)
    nonneg = set()

    def _nn(mm):
        for v in re.split(r"\s*,\s*", mm.group(1)):
            nonneg.add(v.strip())
        return ";"

    c = re.sub(
        r"(?<![\w+\-*/.])([A-Za-z]\d*(?:\s*,\s*[A-Za-z]\d*)*)\s*>=\s*0(?!\d|\.\d)", _nn, c
    )
    constraints = []
    infeasible_const = False
    for raw in re.split(r"[;\n,]+", c):
        item = raw.split(":")[-1].strip()
        item = re.sub(r"^\(?\d+[\).]\s+", "", item)  # "1. x+y<=4"
        item = re.sub(r"\.(?=\s|$)", "", item).strip()
        if not item or not re.search(r"<=|>=|=|<|>", item):
            continue
        words = [w for w in re.findall(r"[A-Za-z]+", item) if len(w) >= 2]
        if words:
            warnings.append(f"Skipped unreadable constraint: '{item}'")
            continue
        parts = re.split(r"(<=|>=|==|=|<|>)", item)
        if len(parts) != 3:
            warnings.append(f"Skipped chained/unsupported constraint: '{item}'")
            continue
        L, op, R = parts
        op = {"<": "<=", "<=": "<=", ">": ">=", ">=": ">=", "=": "=", "==": "="}[op]
        lc, lk = parse_linear(L)
        rc, rk = parse_linear(R)
        coefs = dict(lc)
        for v, k in rc.items():
            coefs[v] = coefs.get(v, 0) - k
        coefs_q = {v: Fraction(k) for v, k in coefs.items() if k != 0}
        if not coefs_q:
            if (op == "<=" and rk - lk < 0) or (op == ">=" and rk - lk > 0) or (op == "=" and rk != lk):
                warnings.append(f"Constraint '{item}' has no variables and can never hold: the problem is infeasible.")
                infeasible_const = True
            continue
        constraints.append({"coefs": {v: float(k) for v, k in coefs_q.items()}, "op": op, "rhs": float(rk - lk),
                            "coefs_q": coefs_q, "rhs_q": Fraction(rk - lk), "source": item})

    variables = []
    for v in list(obj_c) + [v for con in constraints for v in con["coefs"]]:
        if v not in variables:
            variables.append(v)
    # variables that only appear in the non-negativity line (e.g. "Minimize Z = x ... x, y >= 0")
    nn_order = re.findall(r"[A-Za-z]\d*", " ".join(sorted(nonneg, key=lambda v: (len(v), v))))
    for v in nn_order:
        if v not in variables and variables and re.fullmatch(r"[A-Za-z]\d*", v) and (
                v[0] == variables[0][0] or len(v) == 1 and len(variables[0]) == 1):
            variables.append(v)
    if not variables:
        raise ParseError("No decision variables found.")
    if not nonneg:
        warnings.append("No non-negativity line found; assuming all variables are >= 0.")
    for con in constraints:
        con["text"] = f"{fmt_linear(con['coefs'], variables)} {con['op']} {fmt_num(con['rhs'])}"
    return {
        "sense": sense,
        "objective": {v: float(k) for v, k in obj_c.items()},
        "objective_q": {v: Fraction(k) for v, k in obj_c.items()},
        "trivially_infeasible": infeasible_const,
        "constraints": constraints,
        "variables": variables,
        "objective_text": f"{'Maximize' if sense == 'max' else 'Minimize'} Z = {fmt_linear(obj_c, variables)}",
        "warnings": warnings,
    }


def exact_view(prob):
    """Exact (Fraction) view of a parsed problem, filling in values for dicts built by hand.

    Returns dict(variables, sense, c={var: Fraction}, rows=[dict(coefs, op, rhs, text)])."""
    from .exact import to_q
    vs = prob["variables"]
    c = prob.get("objective_q") or {v: to_q(k) for v, k in prob["objective"].items()}
    rows = []
    for con in prob["constraints"]:
        coefs = con.get("coefs_q") or {v: to_q(k) for v, k in con["coefs"].items()}
        rhs = con.get("rhs_q") if con.get("rhs_q") is not None else to_q(con["rhs"])
        rows.append({"coefs": {v: coefs.get(v, Fraction(0)) for v in vs}, "op": con["op"], "rhs": rhs,
                     "text": con.get("text") or f"{fmt_linear(coefs, vs)} {con['op']} {fmt_num(rhs)}"})
    return {"variables": list(vs), "sense": prob["sense"], "c": {v: c.get(v, Fraction(0)) for v in vs}, "rows": rows}
