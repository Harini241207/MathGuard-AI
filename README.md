# MathGuard AI

MathGuard is an independent mathematical watchdog for optimisation solutions written by AI systems
(ChatGPT, Gemini, Claude, ...). It is **not** another AI solver. It solves the problem itself with
textbook algorithms implemented in exact arithmetic, then checks the AI's answer **step by step**:
standard form, every tableau, Cj / Zj / Cj - Zj rows, entering and leaving variables, ratios, pivots,
row operations, corner points, intersections, objective evaluations and the final conclusion. It reports
exactly where a mistake happens, which later steps merely inherit it, and how to correct it, with
supporting passages from your own textbook when you upload one.

## Quick start

```bash
cd mathguard
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                                           # open http://localhost:8000
```

Optional:

```bash
python -m pytest tests -q            # 891 tests, about 10 seconds
python -m scripts.build_datasets     # regenerate data/*.jsonl and the web samples
python -m backend.train              # retrain the experimental model + write models/metrics.json (about 15 s)
```

Requires Python 3.10 or newer. The full test suite was run on Python 3.13, and every module also compiles under Python 3.11. The server trains the experimental model automatically on first start if
`models/mathguard_model.joblib` is missing.

## Using the web app

1. Choose a sample, or paste the **optimisation problem**.
2. Choose the **method** (or *Detect automatically*) and, optionally, which **AI** wrote the answer
   (metadata only: MathGuard does not connect to those services).
3. Paste the AI's **complete solution**: Markdown tables, LaTeX `array` blocks and plain aligned text are all read.
4. Optionally upload your **reference textbook** (PDF or text).
5. Select **Verify the AI's solution**. *Show MathGuard's solution only* displays MathGuard's own working.

The report shows:

* An overall verdict: **Verified**, **Partially verified** (some steps could not be checked), **Working has
  errors** (correct final answer, wrong working), **Incorrect** (wrong final answer), or **Cannot verify**.
  There is no simple Correct / Incorrect when parts are unverified.
* Counts of correct, incorrect and unverifiable steps, plus steps that only follow an earlier error.
* The step-by-step comparison in green (correct), red (incorrect) and amber (unverifiable, incomplete,
  follows an earlier error, or a valid non-textbook choice).
* An error classification table with the code, category, location, AI value and correct value.
* The AI's tableaus next to the tableaus that should follow, with wrong cells highlighted.
* MathGuard's independently derived solution with full working, for every method.
* Textbook passages for the method and for each error, with page numbers from your file. MathGuard's general
  notes are labelled separately and are never presented as quotations.
* Cross-checks: SciPy agreement on the final optimum (secondary) and the experimental ML opinion (collapsed).

Problem formats:

```text
LP:              Maximize Z = 3x + 5y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0
                 (one constraint per line also works; <= / >= / = / ≤ / ≥; fractions like 3/2x)

Transportation:          D1  D2  D3  D4  Supply          (or "Supply: 7, 9, 18" on its own line)
                 S1      19  30  50  10   7
                 S2      70  30  40  60   9
                 S3      40   8  70  20  18
                 Demand   5   8   7  14                   "-" or "M" marks a prohibited route

Assignment:          M1  M2  M3                           "Maximize" anywhere -> profit problem
                 J1   9  11  14                           rectangular matrices get dummy rows / columns
                 J2   6  15  13
                 J3  12  13   6
```

## Architecture

```
question ──lp_parser / matrix_parser──► problem (exact Fractions)
   │
   ├─► chosen method solver ──────────► MathGuard's own full working
   │      graphical_solver   simplex_solver   bigm_solver   two_phase_solver
   │      transportation_solver   assignment_solver          (tableau.py = shared simplex engine)
   │
   └─► solver.solve (two-phase, exact) ► ground truth: status, optimum, alternative optima, degeneracy
                                            │
AI answer ──solution_parser (tables, claims, equations, points)──► step_verifier ◄──┘
           └─answer_parser (final claim) ──► features.build + judge (final-answer category)
                                            │
          report_generator ◄────────────────┘  + scipy_validator (secondary cross-check)
                 │                              + reference.py (textbook passages per method / error)
                 │                              + experimental ML opinion (never overrides)
                 ▼
            Flask API (app.py) ──► frontend/index.html
```

### Files

| File | Status | Purpose |
|---|---|---|
| `backend/app.py` | modified | Flask API; old endpoints kept; `/api/analyze` takes `method` and `ai_source`; new `/api/solve`, `/api/methods` |
| `backend/lp_parser.py` | modified | Also keeps exact Fraction coefficients; reads variables declared only in the non-negativity line; detects impossible constant constraints |
| `backend/answer_parser.py` | modified | Final-claim extraction; fixed "Z = 2x + 3y" misread as Z = 2, "maximum profit 2000", corner lists on the same line, Markdown bold |
| `backend/solver.py` | modified | Ground truth now from MathGuard's exact two-phase solver; SciPy removed from here. Geometry helpers kept |
| `backend/reference.py` | modified | Method-aware retrieval, per-error passages, real PDF page indices + printed page labels |
| `backend/features.py`, `backend/datagen.py` | retained | 28 final-answer features and the synthetic final-answer generator (used by the ML component) |
| `backend/train.py` | rewritten | Problem-level train / validation / test split, independent hand-labelled evaluation, ML vs rules comparison |
| `backend/exact.py` | new | Fraction helpers, symbolic Big-M numbers (`MNum`), number parsing with rounding tolerance |
| `backend/graphical_solver.py` | new | Graphical / corner-point method with full working |
| `backend/tableau.py` | new | Standard form, Cj / Zj / Cj - Zj, ratio test, pivots, row operations, anti-cycling, alternative optima |
| `backend/simplex_solver.py`, `bigm_solver.py`, `two_phase_solver.py`, `lp_common.py` | new | The three tableau methods and their shared result builder |
| `backend/transportation_solver.py` | new | NWC, Least Cost, Vogel, degeneracy (epsilon cells), MODI; answer verification |
| `backend/assignment_solver.py` | new | Hungarian method with every matrix; answer verification |
| `backend/matrix_parser.py` | new | Reads transportation / assignment matrices |
| `backend/solution_parser.py` | new | Reads the AI's working: tables (Markdown / LaTeX / plain), claims, equations, points |
| `backend/step_verifier.py` | new | The step-by-step verification engine |
| `backend/error_catalog.py` | new | Error categories and general method notes |
| `backend/report_generator.py` | new | Builds the full report; method auto-detection |
| `backend/scipy_validator.py` | new | SciPy / HiGHS cross-check (secondary only) |
| `backend/ai_render.py` | new | Renders AI-style working with injected mistakes (tests and dataset) |
| `scripts/build_datasets.py` | new | Builds the structured datasets and web samples |
| `frontend/index.html` | modified | Same visual design; new inputs and the full step-by-step report |
| `tests/` | new | 891 tests (see below) |

## Methods and algorithms

All primary solvers use exact `fractions.Fraction` arithmetic, so the working matches textbook hand calculation
(3/2 stays 3/2). SciPy is used **only** in `scipy_validator.py` and in tests, as an independent cross-check of
the final optimum. It never produces what MathGuard displays.

| Method | What is implemented |
|---|---|
| Graphical | Boundary equations, intercepts, all pairwise intersections (Cramer's rule shown), a feasibility test for every point, corner points ordered around the region, Z at each corner, infeasible / unbounded (exact recession-ray test) / multiple optima (edge or ray) / degenerate corners |
| Simplex | Slack-variable standard form (negative RHS normalised), every tableau with Cj, CB, Zj, Cj - Zj and Z, entering variable (largest positive / most negative), minimum-ratio test, pivot, the row operations as text, optimality, unboundedness, degeneracy and ties, Bland's rule after 50 iterations (Beale's cycling example tested), alternative optima with the alternative tableau. Max and Min |
| Big M | Surplus and artificial variables, penalty -M / +M kept **symbolic**: values are `aM + b` and compared on the M part first, so no arbitrary large constant is needed. Infeasibility when an artificial variable stays positive; unbounded penalised problems are checked against Phase I |
| Two-Phase | Phase I maximises W = -sum of artificial variables; W < 0 means infeasible; artificial variables at zero are driven out, or redundant rows removed; Phase II restores the objective without the artificial columns |
| Transportation | Balancing with dummy rows / columns, profit problems via opportunity loss, NWC / LCM / VAM with every allocation and VAM penalties, epsilon cells for degeneracy (spanning-tree completion), MODI (u, v, dij), stepping-stone loop, theta, alternative optima, prohibited routes |
| Assignment | Dummy rows / columns, maximisation via opportunity loss, row and column reduction, minimum line cover (via maximum matching, König's theorem), theta adjustment, final assignment, detection of alternative assignments, prohibited cells |

## The step verifier

* Every AI tableau is compared with what correctly **follows from the AI's own previous tableau and its own
  pivot choice**. An arithmetic slip is reported once, where it happens. Later steps computed correctly from the
  wrong numbers are marked *follows an earlier error*, not counted as new mistakes.
* A **different valid path** is not an error: an improving entering variable that is not the textbook choice,
  Max(-Z) for a Min problem, Phase I as Min W, Zj - Cj instead of Cj - Zj, x3 / x4 / x5 instead of s1 / s2 / s3,
  rows in a different order, and rounded decimals (0.67 for 2/3) are all accepted and noted.
* Statements are checked as well as tables: "x2 enters", "s3 leaves", "pivot element = 2", "4/1 = 4",
  "R2 → R2 - 3R1", "optimal", "unbounded", "infeasible", "alternative optima".
* Anything that cannot be read, or that contradicts itself (the text says x enters, the table brings in y),
  is marked **unverifiable**, never guessed.

Error codes:

| Code | Category | Code | Category |
|---|---|---|---|
| E01 | Incorrect objective coefficients (Cj / CB) | E08 | Incorrect standard-form conversion |
| E02 | Incorrect slack / surplus / artificial signs | E09 | Incorrect constraint interpretation |
| E03 | Incorrect entering variable | E10 | Incorrect feasibility assessment |
| E04 | Incorrect leaving variable or ratio test | E11 | Incorrect pivot element |
| E05 | Incorrect pivot-row normalisation | E12 | Incorrect final objective value |
| E06 | Incorrect row operation | E13 | Incorrect optimality conclusion (premature stop, missed alternative optima) |
| E07 | Incorrect updated tableau (Zj, Cj - Zj, basis, Z) | E14 | Incorrect unboundedness / infeasibility conclusion |
| G01–G05 | Graphical: boundary / intercept, intersection, feasible region / corners, objective evaluation, choice of optimum | T01–T03, H01 | Transportation (initial solution, cost / allocation, MODI optimality); Hungarian method |

The original final-answer categories (`wrong_status`, `wrong_value`, `suboptimal_solution`,
`infeasible_solution`, `missed_multiple_optima`) are still computed and mapped to E10–E14.

## Data and evaluation (honest summary)

| File | Contents |
|---|---|
| `data/mathguard_30q_dataset_step2.jsonl` | Your original 30 records, unchanged |
| `data/mathguard_extended_dataset.jsonl` | The 30 originals plus 427 generated records: correct and incorrect simplex tableaus (Markdown / LaTeX / plain, s or x slack naming, decimals), wrong entering / leaving / ratio / pivot / normalisation / row operation / Zj / premature stop / final value / unique-claim; graphical answers with wrong intercepts, intersections, infeasible or missing corners, evaluations and picks; infeasible, unbounded, multiple-optimum and degenerate problems. Fields: problem id, method, problem, correct solution, step-by-step solution, AI answer, error category, corrected calculation, verification status, reference source |
| `data/independent_eval.jsonl` | 50 hand-written, **hand-labelled** final answers in styles the generator never uses |
| `data/independent_holdout.jsonl` | 20 more hand-labelled answers on 7 new problems, written after development and evaluated **once** |
| `data/step_examples.json` | The step-level samples shown in the web app |

No textbook text is quoted anywhere. All problems are original test problems that follow standard methodology.

Results (`models/metrics.json`):

* **Step verifier** on 427 generated answers, with labels from the generator (the verifier never sees them):
  every injected error was detected with the expected code, and none of the 79 correct or valid-alternative
  answers was flagged (precision 1.0, recall 1.0).
* **Final-answer judgement**, rules versus the ML model:

  | Test set | Rules | ML |
  |---|---|---|
  | Held-out synthetic problems (problem-level split, n = 1177) | 1.000 | 1.000 |
  | Original-dataset problems, never trained on (n = 598) | 1.000 | 1.000 |
  | Independent hand-labelled answers (n = 50)\* | 1.000 | 1.000 |
  | Untouched hold-out (n = 20) | 1.000 | 1.000 |

  \*This set was used during development. It exposed 3 parser bugs (now fixed) and 2 of my own labelling slips
  (corrected); agreement before those fixes was 45 / 50.

**What these numbers do and do not show.** The 100 % figures on generated data are expected by construction,
because the labels and the features come from the same exact solver. They show internal consistency, not
real-world generalisation. The hand-labelled sets are small (70 answers in total) and contain final-answer
statements rather than long tableaus. Real AI output can be messier than any of these sets, and the parser will
still miss some phrasings; those parts are then reported as unverifiable.

**ML component verdict.** The Random Forest reproduces the deterministic rules exactly and gives **no measurable
improvement**. It is kept as an *experimental*, clearly labelled opinion in a collapsed panel and never changes a
verdict. The mathematical verifier is the sole authority.

## Tests

`python -m pytest tests -q` runs 891 tests:

* `test_graphical.py`: hand-checked working for bounded max and min, infeasible, unbounded, unbounded region
  with a finite optimum, multiple optima on an edge and on a ray, degenerate corner, equality constraints,
  parallel lines, 3 variables rejected, and 150 random problems against SciPy.
* `test_simplex.py`: every tableau of a textbook example, multiple iterations, degeneracy, alternative optima,
  unboundedness, minimisation, negative RHS, a clear rejection of '>=' problems, Beale's cycling example, and 120
  random problems against SciPy.
* `test_bigm_two_phase.py`: symbolic M arithmetic, the initial Big M tableau, the Two-Phase structure,
  infeasible, unbounded, redundant equality constraints, and 150 random problems where Big M, Two-Phase,
  graphical and SciPy must all agree (including detection of multiple optima).
* `test_step_verifier.py`: every scenario in the specification (fully correct, wrong final value, wrong entering
  and leaving variable, wrong ratio, pivot, normalisation and row operation, wrong Zj, premature stop,
  unboundedness and infeasibility misclassified, missed alternative optima, a valid different path, different
  variable naming, incomplete and contradictory answers), Big M / Two-Phase answers, graphical answers, plus a
  matrix of 9 mistakes × 4 problems × 3 table formats.
* `test_transport_assignment.py`: the textbook values NWC 1015 / LCM 814 / VAM 779 / optimum 743, unbalanced,
  degenerate, profit and prohibited cases, 160 random transportation and 120 random assignment problems
  against SciPy, and answer verification.
* `test_reference.py`, `test_integration.py`: textbook retrieval with real PDF page numbers, and the whole
  API including every web sample.

## API

| Endpoint | Body / result |
|---|---|
| `POST /api/analyze` | `{question, answer, method?, ai_source?}` → full report (`report.summary`, `report.steps`, `report.tableaus`, `solution`, `textbook`, `cross_check`, `ml`, plus the original fields `verdict`, `truth`, `claim`, `plot`, `checks`, `reasons`, `references`) |
| `POST /api/solve` | `{question, method?}` → MathGuard's own working only |
| `GET /api/methods` | Available methods and AI sources |
| `GET /api/examples`, `GET /api/metrics`, `GET /api/health` | Samples, evaluation metrics, status |
| `POST /api/extract`, `POST /api/reference/upload`, `GET` / `DELETE /api/reference` | Unchanged from the original |

`method` is one of `auto`, `graphical`, `simplex`, `bigm`, `two_phase`, `transportation`, `assignment`.

## Known limitations

* LP variables are assumed non-negative. Free and integer variables are not supported (integer programming is
  not implemented). Constraints must be linear and written without parentheses.
* The graphical method requires exactly two variables. The standard Simplex method requires '<=' constraints
  with non-negative right-hand sides; otherwise MathGuard explains why and points to Big M or Two-Phase.
* Tableau parsing needs a recognisable header row (Basis / CB / variable names / RHS). Very unusual layouts,
  images of tables, or tables split across messages are reported as unverifiable.
* For transportation, the final allocation, totals and the initial-solution cost are verified. Individual
  MODI u / v values written by the AI are not yet checked cell by cell. For assignment, the row-reduced and
  column-reduced matrices, the assignment and the total are verified; intermediate line covers are not.
* Textbook retrieval is TF-IDF over the extracted text. Scanned PDFs need OCR first. Page numbers are PDF page
  indices, plus printed page labels when the PDF defines them.
* The independent evaluation sets are small (70 answers). Larger collections of real AI outputs would give a
  more reliable estimate of parser coverage.

## Dataset findings (from the original README, confirmed by the new solvers)

* **S02** has alternative optima: (5, 0, 3) and (2, 6, 0) both give Z = 34. A non-basic variable has
  Cj - Zj = 0 in the optimal tableau.
* **G02**: the axis vertex is (0, 4), not (0, 4.5). The optimum (2, 2), Z = 10, is correct.
