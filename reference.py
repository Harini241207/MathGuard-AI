"""Reference-book store: upload a PDF/TXT, chunk it, and retrieve supporting passages with TF-IDF.

The textbook is a *methodological* knowledge base: it supplies definitions, procedures and worked
examples that support MathGuard's explanations.  It is never the source of the numerical answer
(the solvers derive that independently).  Page numbers are only shown when they come from the
uploaded file: the PDF page index, plus the printed page label when the PDF defines one.  Nothing
is ever invented; if no book is loaded, explanations are labelled as MathGuard's general notes.
"""
import io
import pickle
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

STORE = Path(__file__).resolve().parent.parent / "models" / "reference_index.pkl"

ERROR_HINTS = {
    "wrong_status": "infeasible region no feasible solution unbounded objective alternative optima",
    "wrong_value": "objective function value substitute optimal solution evaluate Z",
    "suboptimal_solution": "optimality condition corner point extreme point largest objective value entering variable",
    "infeasible_solution": "feasible region constraints satisfied feasible solution non-negativity",
    "missed_multiple_optima": "alternative optimal solutions zero reduced cost objective parallel binding constraint",
    "correct": "optimal solution simplex graphical method",
}


METHOD_HINTS = {
    "graphical": "graphical method feasible region corner point extreme point iso-profit line boundary line intercept",
    "simplex": "simplex method simplex tableau slack variable pivot entering variable leaving variable Cj Zj net evaluation",
    "bigm": "Big M method penalty method artificial variable M large positive number simplex tableau",
    "two_phase": "two phase method phase I phase II artificial variable auxiliary objective feasibility",
    "transportation": "transportation problem north west corner least cost Vogel approximation MODI u v method supply demand",
    "assignment": "assignment problem Hungarian method row reduction column reduction covering zeros optimal assignment",
}
METHOD_TERMS = {
    "graphical": ["graphical", "corner", "feasible region", "extreme point"],
    "simplex": ["simplex", "tableau", "pivot", "slack"],
    "bigm": ["big m", "penalty", "artificial"],
    "two_phase": ["phase", "artificial", "auxiliary"],
    "transportation": ["transportation", "vogel", "modi", "north", "supply", "demand"],
    "assignment": ["assignment", "hungarian", "reduction", "zeros"],
}


def extract_text(filename, data):
    """Return list of (page_number, text).  For PDFs page_number is the 1-based PDF page index."""
    return [(p, t) for p, t, _ in extract_pages(filename, data)]


def extract_pages(filename, data):
    """List of (pdf_page_index, text, printed_label_or_None)."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        try:
            labels = list(reader.page_labels)
        except Exception:  # noqa: BLE001
            labels = []
        out = []
        for i, p in enumerate(reader.pages):
            lab = labels[i] if i < len(labels) else None
            if lab is not None and str(lab) == str(i + 1):
                lab = None  # identical to the index: nothing extra to show
            out.append((i + 1, p.extract_text() or "", lab))
        return out
    return [(1, data.decode("utf-8", errors="ignore"), None)]


def chunk(pages, words=110, overlap=30):
    out = []
    for item in pages:
        pno, text = item[0], item[1]
        label = item[2] if len(item) > 2 else None
        toks = re.sub(r"\s+", " ", text).strip().split(" ")
        if not toks or toks == [""]:
            continue
        step = words - overlap
        for i in range(0, max(len(toks) - overlap, 1), step):
            piece = " ".join(toks[i: i + words]).strip()
            if len(piece) > 60:
                out.append({"page": pno, "page_label": label, "text": piece})
    return out


class ReferenceStore:
    def __init__(self):
        self.docs = []  # [{name, pages, chunks}]
        self.chunks = []
        self.vec = None
        self.matrix = None
        self._load()

    # --- persistence
    def _load(self):
        if STORE.exists():
            try:
                self.docs, self.chunks = pickle.loads(STORE.read_bytes())
                self._fit()
            except Exception:
                self.docs, self.chunks = [], []

    def _save(self):
        STORE.write_bytes(pickle.dumps((self.docs, self.chunks)))

    def _fit(self):
        if not self.chunks:
            self.vec, self.matrix = None, None
            return
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    # --- API
    def add(self, filename, data):
        pages = extract_pages(filename, data)
        cs = chunk(pages)
        if not cs:
            raise ValueError("No readable text found in that file (scanned PDFs need OCR first).")
        for c in cs:
            c["doc"] = filename
        self.docs = [d for d in self.docs if d["name"] != filename] + [
            {"name": filename, "pages": len(pages), "chunks": len(cs)}]
        self.chunks = [c for c in self.chunks if c["doc"] != filename] + cs
        self._fit()
        self._save()
        return {"name": filename, "pages": len(pages), "chunks": len(cs)}

    def clear(self):
        self.docs, self.chunks, self.vec, self.matrix = [], [], None, None
        if STORE.exists():
            STORE.unlink()

    def status(self):
        return {"loaded": bool(self.chunks), "documents": self.docs, "chunks": len(self.chunks)}

    def search(self, question, error_type, k=3, method=None, extra=""):
        """Passages relevant to a method and an error category (or legacy final-answer label)."""
        if self.vec is None:
            return []
        from .error_catalog import CATALOG
        hint = ERROR_HINTS.get(error_type) or CATALOG.get(error_type, {}).get("query", "")
        query = f"{hint} {hint} {METHOD_HINTS.get(method, '')} {extra} {question}"
        sims = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        if method in METHOD_TERMS:
            terms = METHOD_TERMS[method]
            for i, ch in enumerate(self.chunks):
                low = ch["text"].lower()
                hits = sum(1 for t in terms if t in low)
                if hits:
                    sims[i] += 0.04 * hits  # prefer passages that are actually about the selected method
        order = sims.argsort()[::-1][:k]
        return [self._hit(i, sims[i]) for i in order if sims[i] > 0.05]

    def _hit(self, i, score):
        ch = self.chunks[i]
        lab = ch.get("page_label")
        where = f"PDF page {ch['page']}" + (f" (printed page {lab})" if lab else "") if ch["doc"].lower().endswith(".pdf") \
            else "text file"
        return {"doc": ch["doc"], "page": ch["page"], "page_label": lab, "where": where, "score": float(score),
                "text": ch["text"], "source": "textbook"}

    def method_passages(self, method, question="", k=2):
        """The textbook's description of the selected method itself (procedure / definition)."""
        if self.vec is None:
            return []
        return self.search(question, "procedure", k=k, method=method, extra="procedure steps algorithm definition example")

    def for_errors(self, method, codes, question="", k=2):
        """Supporting passages for each detected error category: {code: [passages]}."""
        out = {}
        for code in codes:
            hits = self.search(question, code, k=k, method=method)
            if hits:
                out[code] = hits
        return out
