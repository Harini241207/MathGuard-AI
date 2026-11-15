"""Train and *honestly* evaluate the experimental ML component.

What the model does: a Random Forest re-predicts the final-answer category (correct / wrong_status /
wrong_value / suboptimal_solution / infeasible_solution / missed_multiple_optima) from 28 numeric features.
Those features are computed from MathGuard's exact solver, and the training labels come from the
deterministic rule judge (features.judge).  The model therefore learns to imitate rules that are already
exact, so near-perfect agreement on generated data is expected *by construction* and is not evidence of
real-world generalisation.

How it is evaluated here:
1. Synthetic problems are split by problem (60 % train / 20 % validation / 20 % test); no problem appears
   in two splits.  The validation split chooses the tree depth.
2. The 20 solve-and-verify problems of the original dataset are never used for training; they form a
   second test set (answers generated, labels from the rules).
3. data/independent_eval.jsonl contains hand-written answers in styles the generator never produces, with
   *manually assigned* labels.  This is the only test whose labels are independent of the rules.
4. Every metric is reported for the ML model AND for the rule-based verifier on the same data, so the
   ML model's added value (if any) is visible.
5. The step-level verifier is evaluated separately on data/mathguard_extended_dataset.jsonl, whose labels
   are the mistakes deliberately injected by the generator (independent of the verifier's logic).

Run from the project root:   python -m backend.train
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import datagen, features, solver
from .answer_parser import parse_answer
from .lp_parser import parse_problem

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mathguard_30q_dataset_step2.jsonl"
INDEP = ROOT / "data" / "independent_eval.jsonl"
HOLDOUT = ROOT / "data" / "independent_holdout.jsonl"
STEPS = ROOT / "data" / "mathguard_extended_dataset.jsonl"
MODEL_PATH = ROOT / "models" / "mathguard_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
LABELS = features.LABELS


def dataset_problem_text(p):
    o = p["objective"]
    return f"{'Maximize' if o['sense'] == 'max' else 'Minimize'} Z = {o['expression']} subject to " + ", ".join(
        p["constraints"]) + ", " + ", ".join(p["nonnegativity"])


def load_dataset_problems():
    out = []
    for line in DATA.read_text().splitlines():
        r = json.loads(line)
        if r["task_type"] == "solve_and_verify":
            out.append((r["id"], dataset_problem_text(r["problem"])))
    return out


def rows_for_problem(pid, prob, rng, n_answers, source):
    truth = solver.solve(prob)
    rows = []
    for _ in range(n_answers):
        beh = datagen.sample_behavior(rng)
        spec = datagen.make_claim(rng, prob, truth, beh)
        text = datagen.render_answer(prob, spec, rng, truth)
        claim = parse_answer(text, prob["variables"])
        f, _d = features.build(prob, truth, claim)
        lab = features.judge(f)
        if lab == "unclear":
            continue
        rows.append({"group": pid, "source": source, "intended": beh, "label": lab, "text": text, **f})
    return rows


def build_synthetic(seed=7, quotas=None, exclude=()):
    rng = random.Random(seed)
    quotas = quotas or {"random": 800, "multi": 250, "infeasible": 220, "unbounded": 220}
    seen = set(exclude)
    rows = []
    pid = 0
    for kind, q in quotas.items():
        got, tries = 0, 0
        while got < q and tries < q * 20:
            tries += 1
            _, prob = datagen.make_problem(rng, kind)
            key = datagen.canon(prob)
            if key in seen:
                continue
            truth = solver.solve(prob)
            if kind == "multi" and truth["status"] != "multiple_optima":
                continue
            if kind == "infeasible" and truth["status"] != "infeasible":
                continue
            if kind == "unbounded" and truth["status"] != "unbounded":
                continue
            seen.add(key)
            rows += rows_for_problem(f"syn{pid}", prob, rng, 4, "synthetic")
            pid += 1
            got += 1
    return pd.DataFrame(rows)


def build_dataset_variants(n_answers=30, seed=99):
    rng = random.Random(seed)
    rows = []
    for pid, text in load_dataset_problems():
        prob = parse_problem(text)
        rows += rows_for_problem(pid, prob, rng, n_answers, "dataset")
    return pd.DataFrame(rows)


def build_independent(path=None):
    path = path or INDEP
    rows = []
    if not path.exists():
        return pd.DataFrame(rows)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        prob = parse_problem(r["problem"])
        truth = solver.solve(prob)
        f, _ = features.build(prob, truth, parse_answer(r["answer"], prob["variables"]))
        rows.append({"group": r["problem_id"], "source": "independent", "label": r["label"], "rule": features.judge(f), **f})
    return pd.DataFrame(rows)


def make_model(depth=None):
    return RandomForestClassifier(n_estimators=300, max_depth=depth, class_weight="balanced_subsample", n_jobs=-1,
                                  random_state=0)


def scores(y, pred):
    labs = [l for l in LABELS if l in set(y) | set(pred)]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labs, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, labels=labs, average="macro", zero_division=0)
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
        "per_class": {l: {"precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
                      for l, a, b, c, d in zip(labs, p, r, f, s)},
        "confusion_matrix": {"labels": labs, "matrix": confusion_matrix(y, pred, labels=labs).tolist()},
    }


def evaluate(model, df, name, rule_col=None):
    X = df[features.FEATURES].values
    y = df["label"].values
    ml = model.predict(X)
    rules = df[rule_col].values if rule_col else np.array([features.judge(dict(zip(features.FEATURES, x))) for x in X])
    out = {"ml": scores(y, ml), "rules": scores(y, rules), "label_counts": dict(Counter(y))}
    print(f"\n=== {name} (n={len(df)}) ===  ML accuracy {out['ml']['accuracy']:.4f} | rules accuracy {out['rules']['accuracy']:.4f}"
          f" | ML macro-F1 {out['ml']['f1_macro']:.4f} | rules macro-F1 {out['rules']['f1_macro']:.4f}")
    return out


def evaluate_step_verifier():
    """Detection quality of the step verifier against injected-mistake labels."""
    if not STEPS.exists():
        return None
    rows = [json.loads(l) for l in STEPS.read_text().splitlines()]
    rows = [r for r in rows if r.get("source") == "generated"]
    per = {}
    tp = fp = fn = tn = 0
    for r in rows:
        want = set(r["error_category"])
        got = set(r["verification_status"]["detected_codes"])
        key = r["injected_mistake"]
        d = per.setdefault(key, {"n": 0, "detected": 0, "exact_code": 0, "false_alarm": 0})
        d["n"] += 1
        if want:
            if got:
                tp += 1
                d["detected"] += 1
                if want <= got:
                    d["exact_code"] += 1
            else:
                fn += 1
        else:
            if got:
                fp += 1
                d["false_alarm"] += 1
            else:
                tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    res = {"n": len(rows), "error_detection": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
                                               "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0},
           "by_injected_mistake": per,
           "note": "Labels are the mistakes injected by backend/ai_render.py; the verifier never sees them. "
                   "'exact_code' counts cases where the expected error category was reported."}
    print(f"\n=== Step verifier on {len(rows)} generated answers ===  precision {prec:.4f} recall {rec:.4f}")
    return res


def main():
    t0 = time.time()
    ds_probs = load_dataset_problems()
    ds_keys = {datagen.canon(parse_problem(t)) for _, t in ds_probs}
    syn = build_synthetic(exclude=ds_keys)
    ds = build_dataset_variants()
    ind = build_independent()
    print(f"synthetic rows: {len(syn)}  dataset-derived rows: {len(ds)}  independent hand-labelled rows: {len(ind)}")

    groups = np.array(sorted(syn["group"].unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    n = len(groups)
    g_train, g_val, g_test = set(groups[: int(.6 * n)]), set(groups[int(.6 * n): int(.8 * n)]), set(groups[int(.8 * n):])
    tr, va, te = (syn[syn["group"].isin(g)] for g in (g_train, g_val, g_test))
    assert not (g_train & g_val) and not (g_train & g_test) and not (g_val & g_test)

    # model selection on the validation split
    best_depth, best_f1 = None, -1
    for depth in (None, 12, 6):
        m = make_model(depth).fit(tr[features.FEATURES].values, tr["label"].values)
        f1 = scores(va["label"].values, m.predict(va[features.FEATURES].values))["f1_macro"]
        print(f"validation macro-F1 depth={depth}: {f1:.4f}")
        if f1 > best_f1 + 1e-9:
            best_depth, best_f1 = depth, f1
    model = make_model(best_depth).fit(tr[features.FEATURES].values, tr["label"].values)
    m_test = evaluate(model, te, "Held-out synthetic problems (test split)")
    m_ds = evaluate(model, ds, "Original-dataset problems (never trained on)")
    m_ind = evaluate(model, ind, "Independent hand-labelled answers", rule_col="rule") if len(ind) else None
    hold = build_independent(HOLDOUT)
    m_hold = evaluate(model, hold, "Untouched hand-labelled hold-out", rule_col="rule") if len(hold) else None
    step_eval = evaluate_step_verifier()

    # final model: synthetic train + validation only (dataset problems and the independent set stay unseen)
    final_df = pd.concat([tr, va], ignore_index=True)
    final = make_model(best_depth).fit(final_df[features.FEATURES].values, final_df["label"].values)
    imp = sorted(zip(features.FEATURES, final.feature_importances_), key=lambda t: -t[1])[:10]
    joblib.dump({"model": final, "features": features.FEATURES, "classes": list(final.classes_)}, MODEL_PATH)

    gain = None
    if m_ind:
        gain = m_ind["ml"]["accuracy"] - m_ind["rules"]["accuracy"]
    conclusion = (
        "The ML model does not improve on the deterministic verifier: on the independent hand-labelled answers its accuracy "
        f"is {m_ind['ml']['accuracy']:.3f} versus {m_ind['rules']['accuracy']:.3f} for the rules. Its labels and features are "
        "derived from the same exact solver, so it can at best reproduce the rules. It is therefore kept only as an "
        "experimental, clearly labelled opinion and never overrides the mathematical verification."
        if m_ind and gain <= 0 else
        "The ML model scored higher than the rules on the independent set; inspect the confusion matrices before relying on it. "
        "It still never overrides the mathematical verification." if m_ind else "No independent set available.")
    print("\n" + conclusion)
    METRICS_PATH.write_text(json.dumps({
        "about": __doc__.strip(),
        "splits": {"train_problems": len(g_train), "validation_problems": len(g_val), "test_problems": len(g_test),
                   "train_rows": int(len(tr)), "validation_rows": int(len(va)), "test_rows": int(len(te))},
        "chosen_max_depth": best_depth,
        "heldout_synthetic": m_test,
        "dataset_problems_unseen": m_ds,
        "independent_hand_labelled": m_ind,
        "independent_hand_labelled_note": ("Used during development: it exposed 3 answer-parser bugs (fixed) and 2 hand-labelling "
                                           "slips (corrected). Agreement before those fixes was 45/50."),
        "untouched_holdout": m_hold,
        "untouched_holdout_note": "Written after development ended and evaluated once; no code was changed afterwards.",
        "step_verifier": step_eval,
        "ml_vs_rules_accuracy_gain_on_independent_set": gain,
        "conclusion": conclusion,
        "top_features": [[a, float(b)] for a, b in imp],
        "classes": list(final.classes_),
        "trained_seconds": round(time.time() - t0, 1),
    }, indent=2))
    print("saved", MODEL_PATH, "in", round(time.time() - t0, 1), "s")
    return final


if __name__ == "__main__":
    main()
