"""Exact arithmetic helpers shared by every MathGuard solver.

All primary solvers work with ``fractions.Fraction`` so that textbook working
(pivots, ratios, Cj - Zj rows ...) is reproduced exactly, without floating-point
drift.  Floats only appear at the API boundary (JSON) and in the SciPy validator.

``MNum`` represents a Big-M quantity  a*M + b  symbolically.  M is treated as
"larger than any number we will ever compare against", which is exactly the
textbook convention, so no arbitrary huge numeric constant is needed.
"""
from __future__ import annotations

from fractions import Fraction
from functools import total_ordering
import re

Q = Fraction
ZERO = Fraction(0)
ONE = Fraction(1)


def to_q(x) -> Fraction:
    """Convert int / float / str / Fraction to a Fraction.

    Floats are converted through their shortest decimal representation, so 0.1
    becomes 1/10 (not 3602879701896397/36028797018963968)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, MNum):
        if x.a != 0:
            raise ValueError("Cannot convert a Big-M expression to a plain number")
        return x.b
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(repr(x)).limit_denominator(10**9)
    return Fraction(str(x).strip())


def fmt_q(x, max_den: int = 10000) -> str:
    """Textbook-style string for an exact number: 3, -1/2, 7/3 ...

    Very large denominators (rare) fall back to a 6-significant-digit decimal."""
    if isinstance(x, MNum):
        return str(x)
    x = to_q(x)
    if x.denominator == 1:
        return str(x.numerator)
    if x.denominator <= max_den:
        return f"{x.numerator}/{x.denominator}"
    return f"{float(x):.6g}"


def q_float(x) -> float:
    if isinstance(x, MNum):
        if x.a != 0:
            raise ValueError("Big-M value has no finite float value")
        x = x.b
    return float(x)


def json_num(x):
    """JSON-friendly representation of an exact number: {"v": float, "s": "3/2"}.

    Big-M numbers become {"v": null, "s": "2M + 3", "m": [2, 3]}."""
    if x is None:
        return None
    if isinstance(x, MNum):
        if x.a == 0:
            return {"v": float(x.b), "s": fmt_q(x.b)}
        return {"v": None, "s": str(x), "m": [float(x.a), float(x.b)]}
    x = to_q(x)
    return {"v": float(x), "s": fmt_q(x)}


@total_ordering
class MNum:
    """a*M + b with exact Fractions; M is an arbitrarily large positive constant."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = to_q(a)
        self.b = to_q(b)

    # ---- construction helpers
    @staticmethod
    def lift(x) -> "MNum":
        return x if isinstance(x, MNum) else MNum(0, to_q(x))

    @property
    def is_plain(self) -> bool:
        return self.a == 0

    # ---- arithmetic
    def __add__(self, o):
        o = MNum.lift(o)
        return MNum(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __neg__(self):
        return MNum(-self.a, -self.b)

    def __sub__(self, o):
        return self + (-MNum.lift(o))

    def __rsub__(self, o):
        return MNum.lift(o) - self

    def __mul__(self, o):
        o = MNum.lift(o)
        if self.a != 0 and o.a != 0:
            raise ArithmeticError("M*M term: not expected in a simplex tableau")
        return MNum(self.a * o.b + o.a * self.b, self.b * o.b)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = MNum.lift(o)
        if o.a != 0:
            raise ArithmeticError("Division by a Big-M expression is not supported")
        return MNum(self.a / o.b, self.b / o.b)

    # ---- comparison (M dominates)
    def _key(self):
        return (self.a, self.b)

    def __eq__(self, o):
        try:
            o = MNum.lift(o)
        except Exception:  # noqa: BLE001
            return NotImplemented
        return self._key() == o._key()

    def __lt__(self, o):
        o = MNum.lift(o)
        return self._key() < o._key()

    def __hash__(self):
        return hash(self._key()) if self.a else hash(self.b)

    def sign(self) -> int:
        if self.a != 0:
            return 1 if self.a > 0 else -1
        return (self.b > 0) - (self.b < 0)

    def __repr__(self):
        return f"MNum({self.a}, {self.b})"

    def __str__(self):
        a, b = self.a, self.b
        if a == 0:
            return fmt_q(b)
        if a == 1:
            ms = "M"
        elif a == -1:
            ms = "-M"
        else:
            ms = f"{fmt_q(a)}M"
        if b == 0:
            return ms
        return f"{ms} + {fmt_q(b)}" if b > 0 else f"{ms} - {fmt_q(-b)}"


def sign(x) -> int:
    if isinstance(x, MNum):
        return x.sign()
    x = to_q(x)
    return (x > 0) - (x < 0)


# ----------------------------------------------------------------------------- parsing numbers in AI text
_FRAC_TEX = re.compile(r"\\[dt]?frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}")


def clean_math_text(s: str) -> str:
    """Normalise unicode minus signs, LaTeX fractions and common TeX noise."""
    s = s.replace("−", "-").replace("–", "-").replace("—", "-").replace("\u2212", "-")
    s = s.replace("½", "1/2").replace("⅓", "1/3").replace("¼", "1/4").replace("¾", "3/4").replace("⅔", "2/3")
    for _ in range(3):
        s = _FRAC_TEX.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\!", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*").replace("·", "*")
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "")
    return s


_SIMPLE_NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+(?:\.\d+)?)?$")


def parse_number(tok: str):
    """Parse one table cell / token into a Fraction or MNum.

    Accepts: 3, -2.5, 3/2, (3)/(2), -1/3, 2M+3, 3-2M, -M, M, 0.333 ...
    Returns None for blanks, dashes, infinity, 'NA', '-', '--'."""
    if tok is None:
        return None
    s = clean_math_text(str(tok)).strip()
    s = s.strip("*` ").replace(" ", "")
    s = re.sub(r"^\((.*)\)$", r"\1", s) if s.count("(") == 1 and s.startswith("(") and s.endswith(")") else s
    if s in ("", "-", "--", "---", "NA", "N/A", "na", "n/a", "∞", "inf", "infinity", "+∞", "—", "x", "X") or s.lower() in (
            "undefined", "none", "nan", "infty", "\\infty", "notdefined"):
        return None
    s = s.replace("(", "").replace(")", "") if re.fullmatch(r"[\d()+\-/.]+", s) else s
    if _SIMPLE_NUM.match(s):
        if "/" in s:
            a, b = s.split("/")
            bq = to_q(b)
            if bq == 0:
                return None
            return to_q(a) / bq
        return to_q(s)
    # Big-M expressions: terms like 2M, -M, 3/2M, M/2, +5
    if "M" in s or "m" in s:
        s2 = s.replace("m", "M").replace("*", "")
        terms = re.findall(r"[+-]?[^+-]+", s2)
        if not terms:
            return None
        a = b = Fraction(0)
        for t in terms:
            sgn = -1 if t.startswith("-") else 1
            t = t.lstrip("+-")
            if not t:
                return None
            if "M" in t:
                coef = t.replace("M", "", 1)
                if "M" in coef:
                    return None
                if coef.startswith("/"):
                    coef = "1" + coef
                coef = coef or "1"
                try:
                    a += sgn * _qs(coef)
                except (ValueError, ZeroDivisionError):
                    return None
            else:
                try:
                    b += sgn * _qs(t)
                except (ValueError, ZeroDivisionError):
                    return None
        return MNum(a, b) if a != 0 else b
    return None


def decimal_tolerance(tok: str) -> Fraction:
    """How far a typed number may be from the exact value and still count as the same.

    Integers and fractions are exact (tolerance 0).  A decimal with d places may
    be off by one unit in its last place (covers both rounding and truncation:
    0.33 / 0.333 / 0.67 / 0.66 for 1/3 and 2/3)."""
    s = clean_math_text(str(tok))
    decs = [len(m) for m in re.findall(r"\d\.(\d+)", s)]
    if not decs:
        return Fraction(0)
    return Fraction(1, 10 ** min(decs)) * Fraction(101, 100)


def parse_number_tol(tok: str):
    """(value, tolerance) for a token; value None if not a number."""
    v = parse_number(tok)
    return v, (decimal_tolerance(tok) if v is not None else Fraction(0))


def matches(claimed, tol, truth) -> bool:
    """Is a claimed (possibly rounded) number the same as the exact truth?"""
    if claimed is None or truth is None:
        return False
    if isinstance(claimed, MNum) or isinstance(truth, MNum):
        A, B = MNum.lift(claimed), MNum.lift(truth)
        return abs(A.a - B.a) <= tol and abs(A.b - B.b) <= tol * max(1, abs(B.b))
    return abs(to_q(claimed) - to_q(truth)) <= tol * max(Fraction(1), abs(to_q(truth)) / 10) if tol else to_q(claimed) == to_q(truth)


def _qs(t: str) -> Fraction:
    if "/" in t:
        a, b = t.split("/", 1)
        return to_q(a or "1") / to_q(b)
    return to_q(t)


def approx_equal(a, b, tol: Fraction = Fraction(1, 200)) -> bool:
    """Equality that tolerates rounded decimals written by an AI (e.g. 0.333 for 1/3).

    Exact values compare exactly; a decimal with d places is accepted if it is
    within half a unit of its last place (min tolerance 0.5%) of the true value."""
    if a is None or b is None:
        return False
    if isinstance(a, MNum) or isinstance(b, MNum):
        A, B = MNum.lift(a), MNum.lift(b)
        return _close(A.a, B.a, tol) and _close(A.b, B.b, tol)
    return _close(to_q(a), to_q(b), tol)


def _close(x: Fraction, y: Fraction, tol: Fraction) -> bool:
    if x == y:
        return True
    scale = max(abs(x), abs(y), Fraction(1))
    return abs(x - y) <= tol * scale


def simplest_between(lo: Fraction, hi: Fraction) -> Fraction:
    """The fraction with the smallest denominator in the closed interval [lo, hi] (Stern-Brocot search)."""
    if lo > hi:
        lo, hi = hi, lo
    if lo <= 0 <= hi:
        return Fraction(0)
    neg = hi < 0
    if neg:
        lo, hi = -hi, -lo
    # continued-fraction based search
    def rec(a, b):
        fl = a.numerator // a.denominator
        if Fraction(fl) == a:
            return Fraction(fl)
        if fl + 1 <= b:
            return Fraction(fl + 1)
        return fl + 1 / rec(1 / (b - fl), 1 / (a - fl))
    r = rec(lo, hi)
    return -r if neg else r


def snap_decimal(value: Fraction, tol: Fraction, max_den: int = 12) -> Fraction:
    """A typed decimal (e.g. 0.67) that is within rounding distance of a simple fraction (2/3) becomes that fraction."""
    if not tol or isinstance(value, MNum):
        return value
    cand = simplest_between(value - tol, value + tol)
    return cand if cand.denominator <= max_den else value
