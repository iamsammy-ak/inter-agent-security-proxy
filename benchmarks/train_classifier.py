"""Train the injection classifiers on the deepset/prompt-injections *train* split only.

For each embedder (bge-small via fastembed, and the download-free hashing embedder) this script:
1. embeds the 546 training texts (cached under .cache/embeddings);
2. picks the regularisation strength C by 5-fold stratified cross-validation (out-of-fold F1);
3. picks the decision threshold that maximises F1 on the out-of-fold probabilities;
4. refits on the whole train split and writes a JSON artefact into src/agent_proxy/data/.

The test split is never read here. Requires scikit-learn (dev extra) and, for bge-small, fastembed.

usage: python benchmarks/train_classifier.py [--only hashing|bge-small]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_proxy.datasets import DATASET_ID, load_split  # noqa: E402
from agent_proxy.injection.classifier import ARTEFACT_FILES, ClassifierArtefact  # noqa: E402
from agent_proxy.injection.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder  # noqa: E402

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SEED = 13


def make_embedder(name: str) -> Embedder:
    return HashingEmbedder() if name == "hashing" else FastEmbedEmbedder()


def train(name: str) -> ClassifierArtefact:
    rows = load_split("train")
    texts = [r.text for r in rows]
    y = np.array([r.label for r in rows])
    embedder = make_embedder(name)
    cache = ROOT / ".cache" / "embeddings" / f"{name}-train.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        X = embedder.embed(texts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def model(c: float):  # noqa: ANN202 - sklearn pipeline
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, class_weight="balanced"))

    best_c, best_f1, best_oof = C_GRID[0], -1.0, np.zeros(len(y))
    for c in C_GRID:
        oof = cross_val_predict(model(c), X, y, cv=folds, method="predict_proba")[:, 1]
        f1 = f1_score(y, oof >= 0.5)
        print(f"  {name} C={c:<5} out-of-fold F1@0.5 = {f1:.3f}")
        if f1 > best_f1:
            best_c, best_f1, best_oof = c, f1, oof
    thresholds = np.round(np.arange(0.2, 0.81, 0.05), 2)
    scores = [f1_score(y, best_oof >= t) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(scores))])
    pred = best_oof >= threshold
    cv = {
        "f1": round(float(f1_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
    }
    final = model(best_c).fit(X, y)
    scaler, clf = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
    print(f"  {name}: C={best_c}, threshold={threshold}, out-of-fold {cv}")
    return ClassifierArtefact(
        name=name,
        embedder=embedder.name,
        dim=embedder.dim,
        mean=[round(float(v), 6) for v in scaler.mean_],
        scale=[round(float(v), 6) for v in scaler.scale_],
        coef=[round(float(v), 6) for v in clf.coef_[0]],
        intercept=round(float(clf.intercept_[0]), 6),
        threshold=threshold,
        C=best_c,
        trained_on=f"{DATASET_ID} train split",
        n_train=len(y),
        cv_metrics=cv,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(ARTEFACT_FILES), default=None)
    args = parser.parse_args()
    for name in [args.only] if args.only else sorted(ARTEFACT_FILES):
        artefact = train(name)
        out = ROOT / "src" / "agent_proxy" / "data" / ARTEFACT_FILES[name]
        out.write_text(artefact.model_dump_json(indent=None), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
