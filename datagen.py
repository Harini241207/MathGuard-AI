"""Synthetic LP problems + realistic correct/wrong AI answers (rendered as free text, then parsed)."""
import random
from fractions import Fraction

from . import solver
from .lp_parser import parse_problem, fmt_num

WRONG = [
    "wrong_value", "other_vertex", "infeasible_point", "swap_sense", "sign_flip",
    "coord_swap", "wrong_status", "arith_slip",
]
BEHAVIORS = ["correct"] * 7 + WRONG * 1  # sampling pool; weights tuned in sample_behavior()


# ----------------------------------------------------------------------------- problems
def _terms(coefs, names):
    out = []
    for k, nm in zip(coefs, names):
        if k == 0:
            continue
        a = abs(k)
        body = (str(a) if a != 1 else "") + nm
        if not out:
            out.append(("-" if k < 0 else "") + body)
        else:
            out.append((" - " if k < 0 else " + ") + body)
    return "".join(out) or "0"


def problem_text(sp, rng):
    names, sense = sp["names"], sp["sense"]
    word = rng.choice(["Maximize", "Maximize", "Max", "maximize"] if sense == "max" else ["Minimize", "Minimize", "Min", "minimize"])
    obj = f"{word} Z = {_terms(sp['obj'], names)}"
    cons = [f"{_terms(a, names)} {op} {b}" for a, op, b in sp["cons"]]
    nn = f"{', '.join(names)} >= 0"
    style = rng.randrange(5)
    if style == 0:
        return obj + "\nsubject to\n" + "\n".join("  " + c for c in cons) + "\n  " + nn
    if style == 1:
        return obj + " subject to " + ", ".join(cons) + ", " + nn
    if style == 2:
        return (obj + "; s.t. " + "; ".join(cons) + "; " + nn).replace("<=", "≤").replace(">=", "≥")
    if style == 3:
        return obj + " such that " + " and ".join(cons) + " with " + nn
    return obj + "\nConstraints:\n" + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cons)) + "\n" + nn


def gen_struct(rng, kind):
    n = rng.choice([2, 2, 2, 3])
    names = rng.choice([["x", "y", "z"], ["x", "y", "z"], ["x1", "x2", "x3"]])[:n]
    sense = rng.choice(["max", "min"])
    m = rng.randint(2, 4)
    obj = [rng.randint(1, 9) for _ in range(n)]
    cons = []

    def row(allow_zero=True):
        while True:
            a = [rng.randint(0, 6) for _ in range(n)]
            if any(a):
                return a

    if kind == "multi":
        sense = "max"
        base = row()
        k = rng.randint(1, 3)
        obj = [k * a for a in base]
        if not any(obj):
            obj = [1] * n
        cons.append((base, "<=", rng.randint(6, 40)))
        for _ in range(m - 1):
            cons.append((row(), "<=", rng.randint(4, 40)))
    elif kind == "infeasible":
        a = row()
        b = rng.randint(2, 12)
        cons.append((a, "<=", b))
        cons.append((a, ">=", b + rng.randint(2, 10)))
        for _ in range(m - 2):
            cons.append((row(), rng.choice(["<=", ">="]), rng.randint(2, 30)))
    elif kind == "unbounded":
        sense = "max"
        a = [rng.randint(1, 4)] + [-rng.randint(1, 4)] + [0] * (n - 2)
        cons.append((a, "<=", rng.randint(1, 10)))
        for _ in range(m - 1):
            cons.append((row(), ">=", rng.randint(2, 20)))
        if rng.random() < 0.5:
            obj = [rng.randint(1, 9) for _ in range(n)]
    else:
        for _ in range(m):
            if sense == "max":
                op = "<=" if rng.random() < 0.85 else ">="
            else:
                op = ">=" if rng.random() < 0.85 else "<="
            cons.append((row(), op, rng.randint(3, 40)))
    return {"names": names, "sense": sense, "obj": obj, "cons": cons}


def make_problem(rng, kind):
    sp = gen_struct(rng, kind)
    text = problem_text(sp, rng)
    return text, parse_problem(text)


def canon(prob):
    return (prob["sense"], tuple(sorted(prob["objective"].items())),
            tuple(sorted((tuple(sorted(c["coefs"].items())), c["op"], c["rhs"]) for c in prob["constraints"])))


# ----------------------------------------------------------------------------- claims
def _obj(prob, pt):
    return sum(prob["objective"].get(v, 0.0) * pt[v] for v in prob["variables"])


def _feasible(prob, pt):
    return solver.check_point(prob, pt)["feasible"]


def make_claim(rng, prob, truth, behavior):
    """Return a claim spec: dict(status, point, points, value, unique)."""
    vs = prob["variables"]
    st = truth["status"]
    spec = {"status": "optimal", "point": None, "points": [], "value": None, "unique": False}
    verts = [dict(zip(vs, map(float, v))) for v in solver.vertices(prob)]
    z = truth.get("value")

    if st in ("optimal", "multiple_optima"):
        sol = dict(truth["solution"])
        face = truth["face"]
        if behavior == "correct":
            if st == "multiple_optima":
                if rng.random() < 0.5:
                    spec["status"] = "multiple"
                    spec["points"] = [dict(p) for p in face[: rng.choice([2, 2, 3])]]
                    spec["point"] = dict(spec["points"][0])
                else:
                    spec["point"] = dict(rng.choice(face))
            else:
                spec["point"] = sol
            spec["value"] = z
            r = rng.random()
            if r < 0.07:
                spec["value"] = None
            elif r < 0.12 and st != "multiple_optima":
                spec["point"] = None
        elif behavior == "wrong_value":
            spec["point"] = sol
            spec["value"] = z + rng.choice([-1, 1]) * rng.choice([1, 2, 3, 5, 10, 0.5 * max(abs(z), 2)])
        elif behavior == "arith_slip":
            spec["point"] = sol
            spec["value"] = z + rng.choice([-3, -2, -1, 1, 2, 3])
        elif behavior == "sign_flip":
            spec["point"] = sol
            spec["value"] = -z if abs(z) > 0.6 else z + 2
        elif behavior == "other_vertex":
            others = [v for v in verts if abs(_obj(prob, v) - z) > 1e-6]
            if not others:
                return make_claim(rng, prob, truth, "wrong_value")
            v = rng.choice(others)
            spec["point"] = v
            spec["value"] = _obj(prob, v) if rng.random() < 0.8 else z
        elif behavior == "infeasible_point":
            v = dict(sol)
            for _ in range(20):
                nv = dict(v)
                k = rng.choice(vs)
                nv[k] = nv[k] + rng.choice([1, 2, 3, -1])
                if not _feasible(prob, nv) and all(val >= 0 for val in nv.values()):
                    v = nv
                    break
            spec["point"] = v
            spec["value"] = _obj(prob, v) if rng.random() < 0.7 else z
        elif behavior == "swap_sense":
            flip = dict(prob)
            flip["sense"] = "min" if prob["sense"] == "max" else "max"
            t2 = solver.solve(flip)
            if t2["status"] not in ("optimal", "multiple_optima"):
                return make_claim(rng, prob, truth, "wrong_value")
            spec["point"] = t2["solution"]
            spec["value"] = t2["value"]
        elif behavior == "coord_swap":
            if len(vs) < 2:
                return make_claim(rng, prob, truth, "wrong_value")
            v = dict(sol)
            a, b = rng.sample(vs, 2)
            v[a], v[b] = v[b], v[a]
            spec["point"] = v
            spec["value"] = z if rng.random() < 0.5 else _obj(prob, v)
        elif behavior == "wrong_status":
            r = rng.random()
            if st == "multiple_optima":
                if r < 0.6:
                    spec["point"] = dict(face[0]); spec["value"] = z; spec["unique"] = True
                else:
                    spec["status"] = rng.choice(["infeasible", "unbounded"])
            else:
                if r < 0.4:
                    spec["status"] = rng.choice(["infeasible", "unbounded"])
                elif r < 0.7:
                    spec["status"] = "multiple"; spec["point"] = sol; spec["points"] = [sol]; spec["value"] = z
                else:
                    spec["status"] = "unbounded"
    elif st == "infeasible":
        if behavior == "correct":
            spec["status"] = "infeasible"
        elif behavior == "wrong_status" and rng.random() < 0.3:
            spec["status"] = "unbounded"
        else:
            pt = {v: float(rng.randint(0, 6)) for v in vs}
            spec["point"] = pt
            spec["value"] = _obj(prob, pt) if rng.random() < 0.85 else float(rng.randint(1, 30))
    else:  # unbounded
        if behavior == "correct":
            spec["status"] = "unbounded"
        elif behavior == "wrong_status" and rng.random() < 0.3:
            spec["status"] = "infeasible"
        else:
            vv = [v for v in verts]
            pt = rng.choice(vv) if vv else {v: 1.0 for v in vs}
            spec["point"] = pt
            spec["value"] = _obj(prob, pt)
    return spec


# ----------------------------------------------------------------------------- rendering
def _fmt(x, rng):
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    r = rng.random()
    if r < 0.35:
        fr = Fraction(x).limit_denominator(24)
        if abs(float(fr) - x) < 1e-9:
            return f"{fr.numerator}/{fr.denominator}"
    return f"{x:.{rng.choice([2, 2, 3, 4])}f}"


def _render_point(pt, vs, rng):
    s = rng.randrange(5)
    vals = [_fmt(pt[v], rng) for v in vs]
    if s == 0:
        return ", ".join(f"{v} = {a}" for v, a in zip(vs, vals))
    if s == 1:
        return f"({', '.join(vs)}) = ({', '.join(vals)})"
    if s == 2:
        return ", ".join(f"{v}={a}" for v, a in zip(vs, vals))
    if s == 3:
        return " and ".join(f"{v}* = {a}" for v, a in zip(vs, vals))
    return "at the point (" + ", ".join(vals) + ")"


def _render_value(val, sense, rng):
    a = _fmt(val, rng)
    s = rng.randrange(6)
    if s == 0:
        return f"Z = {a}"
    if s == 1:
        return f"the {'maximum' if sense == 'max' else 'minimum'} value of Z is {a}"
    if s == 2:
        return f"Optimal value: {a}"
    if s == 3:
        return f"Z{'max' if sense == 'max' else 'min'} = {a}"
    if s == 4:
        return f"objective value = {a}"
    return f"Z = {a}"


def render_answer(prob, spec, rng, truth=None):
    vs = prob["variables"]
    sense = prob["sense"]
    lines = []
    clutter = rng.random() < 0.45
    if clutter:
        lines.append(rng.choice([
            "Step 1: Write the constraints as equations to get the boundary lines.",
            "Convert to standard form and find the feasible region.",
            "Let me solve this with the graphical / corner point method.",
        ]))
        if len(vs) == 2:
            verts = solver.vertices(prob)
            if verts:
                lines.append("Corner points: " + ", ".join(f"({fmt_num(v[0])}, {fmt_num(v[1])})" for v in verts))
                lines.append("Evaluating Z: " + "; ".join(
                    f"Z({fmt_num(v[0])}, {fmt_num(v[1])}) = {fmt_num(_obj(prob, dict(zip(vs, v))))}" for v in verts))
    cue = rng.choice(["Therefore", "Hence", "Thus", "Final answer:", "Conclusion:", ""])
    st = spec["status"]
    if st == "infeasible":
        lines.append(rng.choice([
            "The constraints contradict each other, so the problem is infeasible.",
            "There is no feasible solution: the feasible region is empty.",
            "Hence the LP is infeasible.",
            "Infeasible.",
        ]))
    elif st == "unbounded":
        lines.append(rng.choice([
            "The objective can increase without bound along a feasible ray, so the LP is unbounded.",
            "Therefore the problem is unbounded and has no finite optimum.",
            "The LP is unbounded.",
        ]))
    elif st == "multiple":
        pts = spec["points"] or ([spec["point"]] if spec["point"] else [])
        ptxt = " and ".join("(" + ", ".join(_fmt(p[v], rng) for v in vs) + ")" for p in pts)
        tail = f", {_render_value(spec['value'], sense, rng)}" if spec["value"] is not None else ""
        lines.append(rng.choice([
            f"Multiple optimal solutions exist, for example {ptxt}{tail}.",
            f"Alternative optima: the objective is parallel to a binding constraint; optimal points include {ptxt}{tail}.",
        ]))
    else:
        parts = []
        if spec["point"]:
            parts.append(_render_point(spec["point"], vs, rng))
        if spec["value"] is not None:
            parts.append(_render_value(spec["value"], sense, rng))
        body = ", ".join(parts) if parts else "no solution given"
        if spec.get("unique"):
            body = "The optimal solution is unique: " + body
        else:
            body = rng.choice(["The optimal solution is ", "Optimal solution: ", ""]) + body
        lines.append((cue + " " + body).strip() + rng.choice([".", "", "."]))
    return "\n".join(lines)


def sample_behavior(rng, p_correct=0.38):
    return "correct" if rng.random() < p_correct else rng.choice(WRONG)
