import pathlib

import pytest

import backend.reference as R


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "STORE", tmp_path / "ref.pkl")
    return R.ReferenceStore()


BOOK = (b"Chapter 2. Graphical method. Draw each constraint as a boundary line from its intercepts. The corner points of the "
        b"feasible region are evaluated in the objective function and the best corner point is optimal. " * 3 +
        b"Chapter 3. The simplex method. The entering variable is the non-basic variable with the largest positive Cj - Zj. "
        b"The leaving variable is chosen by the minimum ratio test using positive entries of the key column. " * 3 +
        b"Chapter 6. Transportation problem. Vogel approximation method uses penalties; the MODI method tests optimality. " * 3)


def test_upload_and_method_aware_retrieval(store):
    info = store.add("notes.txt", BOOK)
    assert info["chunks"] >= 2
    hit = store.search("Maximize Z = 3x + 5y", "E04", method="simplex", k=1)[0]
    assert "ratio" in hit["text"].lower()
    assert hit["source"] == "textbook" and hit["where"] == "text file"
    g = store.search("", "G04", method="graphical", k=1)[0]
    assert "corner" in g["text"].lower()
    by = store.for_errors("simplex", ["E03", "E04"])
    assert set(by) == {"E03", "E04"}


def test_no_book_means_no_passages(store):
    assert store.search("x", "E03", method="simplex") == []
    assert store.method_passages("simplex") == []


def test_pdf_pages_are_real_page_numbers(store):
    from pypdf import PdfWriter
    from pypdf.generic import RectangleObject
    import io
    # build a 2-page PDF with text using reportlab if available, else skip
    rl = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Introduction to operations research and linear models in general terms.")
    c.showPage()
    c.drawString(72, 720, "Simplex method: minimum ratio test selects the leaving variable using positive key column entries.")
    c.showPage()
    c.save()
    store.add("book.pdf", buf.getvalue())
    hit = store.search("", "E04", method="simplex", k=1)[0]
    assert hit["page"] == 2 and hit["where"].startswith("PDF page 2")
