"""Compare broad VAHTA salary models on fixed, employer-grouped folds."""
import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from train import ALPHAS, CARD, HERE, fit, predict, score

CONTEXT = ["unit", "tax", "quote_kind"]
SPECS = ("card", "rules", "rules_nlp", "card_char", "full_text", "ridge", "specialists", "blend")


def with_context(x, y):
    x = x.copy()
    for c in CONTEXT:
        x[c] = y[c].astype(object)
    return x


def text_fit(text):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=45_000, sublinear_tf=True, dtype=np.float32)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=25_000,
                           sublinear_tf=True, dtype=np.float32)
    return (char, word), (char.fit_transform(text), word.fit_transform(text))


def text_transform(text, encoders):
    return tuple(v.transform(text) for v in encoders)


def dense_text(train, test, index_train, index_test):
    encoders, a, b = [], {}, {}
    with threadpool_limits(limits=3):
        for kind, dim, first, second in zip(("char", "word"), (32, 24), train, test, strict=True):
            enc = TruncatedSVD(n_components=min(dim, first.shape[1] - 1), n_iter=5, random_state=42)
            names = [f"text_{kind}_{i}" for i in range(enc.n_components)]
            a.update(zip(names, enc.fit_transform(first).T, strict=True))
            b.update(zip(names, enc.transform(second).T, strict=True))
            encoders.append((enc, names))
    return pd.DataFrame(a, index=index_train), pd.DataFrame(b, index=index_test), encoders


def sparse_features(x, text, cols, encoder=None):
    cat = [c for c in cols if x[c].dtype == object]
    num = [c for c in cols if c not in cat]
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=3, dtype=np.float32)
        one_hot = encoder.fit_transform(x[cat])
    else:
        one_hot = encoder.transform(x[cat])
    values = x[num].fillna(0)
    numeric = csr_matrix(np.sign(values) * np.log1p(values.abs()))
    return hstack([one_hot, numeric, *text], format="csr"), encoder


def quantile_base(train, test):
    keys = [["unit", "quote_kind", "tax", "profession_resolved"], CONTEXT,
            ["unit", "quote_kind"], ["unit"]]
    tables = []
    for k in keys:
        t = {}
        for key, d in train.groupby(k):
            if len(k) == 1 or (len(d) >= 10 and d.group.nunique() >= 3):
                t[key] = np.quantile(d.y, ALPHAS)
        tables.append(t)
    return np.vstack([
        next((t[key] for k, t in zip(keys, tables, strict=True)
              if (key := tuple(row[c] for c in k)) in t), np.full(3, np.nan))
        for row in test.to_dict("records")
    ])


def ridge_bundle(x, labels, text, cols, fold):
    split = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=100 + fold)
    left, right = next(split.split(x, labels.y, labels.group))
    a, b = x.iloc[left], x.iloc[right]
    dictionary, a_text = text_fit(text.loc[a.index])
    b_text = text_transform(text.loc[b.index], dictionary)
    ax, encoder = sparse_features(a, a_text, cols)
    bx, _ = sparse_features(b, b_text, cols, encoder)
    model = Ridge(alpha=8, solver="lsqr")
    model.fit(ax, np.log(labels.loc[a.index].y))
    residual = pd.Series(np.log(labels.loc[b.index].y) - model.predict(bx), index=b.index)
    offsets = {"*": np.quantile(residual, [.25, .75])}
    for unit, values in residual.groupby(labels.loc[b.index].unit):
        if len(values) >= 30:
            offsets[unit] = np.quantile(values, [.25, .75])
    return model, dictionary, encoder, offsets


def ridge_predict(bundle, x, text, cols, unit):
    model, dictionary, encoder, offsets = bundle
    matrix, _ = sparse_features(x, text_transform(text, dictionary), cols, encoder)
    p50 = model.predict(matrix)
    band = np.vstack([offsets.get(u, offsets["*"]) for u in unit])
    return np.exp(np.column_stack([p50 + np.minimum(band[:, 0], 0),
                                   p50, p50 + np.maximum(band[:, 1], 0)]))


def aggregate_metrics(rows, predictions):
    records = []
    for variant, p in predictions.items():
        has = np.isfinite(p).all(axis=1)
        published_shift = rows.unit.eq("shift").to_numpy()
        expected_amount = (rows[["low", "high"]].min(axis=1).ge(300)
                           & rows[["low", "high"]].max(axis=1).le(50_000)).to_numpy()
        selections = [
            ("total", "all", np.ones(len(rows), dtype=bool)),
            ("shift_plausibility", "300_to_50000", published_shift & expected_amount),
            ("shift_plausibility", "outside_300_to_50000", published_shift & ~expected_amount),
        ]
        for dim, key, keep in selections + [
            (dim, str(key), values.to_numpy() == key)
            for dim, values in {"unit": rows.unit, "unit_shape": rows.unit + "|" + rows.quote_kind,
                                "route": rows.unit + "|" + rows.quote_kind + "|" + rows.tax,
                                "source": rows.source_kind, "military": rows.military}.items()
            for key in values.unique()
        ]:
            take = keep & has
            if take.any():
                records.append(dict(variant=variant, dimension=dim, key=key,
                                    possible=int(keep.sum()),
                                    **score(rows.y.to_numpy()[take], p[take], rows.group.to_numpy()[take])))
    return pd.DataFrame(records)


def format_labels(labels):
    return labels.unit.where(labels.unit.isin({"shift", "month", "rotation", "unspecified", "no_amount"}), "other")


def classify_formats(out, labels, estimates):
    actual = format_labels(labels)
    freq = labels.groupby("fold").size()
    scores, rows = [], labels[["id", "group", "fold"]].copy()
    rows["actual"] = actual
    for variant, pred in estimates.items():
        rows[variant] = pred
        scores.append(dict(variant=variant, n=int(freq.sum()), accuracy=accuracy_score(actual, pred),
                           balanced_accuracy=balanced_accuracy_score(actual, pred),
                           macro_f1=f1_score(actual, pred, average="macro")))
    rows.to_csv(out / "format_oof.csv", index=False)
    pd.DataFrame(scores).to_csv(out / "format_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "full_results")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    for file in ("labels.csv", "dataset.json"):
        if (args.cache / file).resolve() != (args.out / file).resolve():
            shutil.copyfile(args.cache / file, args.out / file)
    labels, x, text, _ = pd.read_pickle(args.cache / "features.local.pkl")
    card = CARD + ["military_type"]
    rules = [c for c in x if not c.startswith("nlp_")]
    specs = {"card": card, "rules": rules, "rules_nlp": list(x),
             "card_char": card, "full_text": list(x)}
    rows = labels.loc[labels.y.notna()].copy()
    rows["position"] = np.arange(len(rows))
    x_full = with_context(x, labels)
    predictions = {key: np.full((len(rows), 3), np.nan) for key in ("empirical", *SPECS)}
    format_estimates = {key: np.empty(len(labels), dtype=object) for key in ("majority", "logistic_text")}
    format_values = format_labels(labels)
    for fold in sorted(labels.fold.unique()):
        print(f"Training fold {fold}/5", flush=True)
        all_train = labels.loc[~labels.fold.eq(fold)]
        all_test = labels.loc[labels.fold.eq(fold)]
        train, test = rows.loc[~rows.fold.eq(fold)], rows.loc[rows.fold.eq(fold)]
        assert set(all_train.group).isdisjoint(all_test.group)
        pos = test.position.to_numpy()
        predictions["empirical"][pos] = quantile_base(train, test)

        # Encoder sees every training card, including those without a numeric quote.
        dictionary, train_text = text_fit(text.loc[all_train.index])
        test_text = text_transform(text.loc[all_test.index], dictionary)
        a, b, _ = dense_text(train_text, test_text, all_train.index, all_test.index)
        ax, bx = x_full.loc[a.index].join(a), x_full.loc[b.index].join(b)
        for name, cols in specs.items():
            encoded = [c for c in a if c.startswith("text_char_")] if name == "card_char" else list(a) if name == "full_text" else []
            columns = cols + CONTEXT + encoded
            model = fit(ax.loc[train.index], columns, train.y)
            predictions[name][pos] = predict(model, bx.loc[test.index], columns)

        specialist = predictions["rules"][pos].copy()
        for unit, case in train.groupby("unit"):
            same = test.unit.eq(unit)
            if len(case) >= 40 and case.group.nunique() >= 12 and same.any():
                cols = rules + CONTEXT
                model = fit(ax.loc[case.index], cols, case.y)
                specialist[same] = predict(model, bx.loc[test.loc[same].index], cols)
        predictions["specialists"][pos] = specialist
        ridge_cols = rules + CONTEXT
        bundle = ridge_bundle(x_full.loc[train.index], train, text, ridge_cols, fold)
        predictions["ridge"][pos] = ridge_predict(bundle, x_full.loc[test.index], text.loc[test.index], ridge_cols, test.unit)
        predictions["blend"][pos] = np.sqrt(predictions["full_text"][pos] * predictions["ridge"][pos])

        ux, enc = sparse_features(x.loc[a.index], train_text, card)
        vx, _ = sparse_features(x.loc[b.index], test_text, card, enc)
        classifier = LogisticRegression(C=2.0, max_iter=400, class_weight="balanced")
        classifier.fit(ux, format_values.loc[a.index])
        format_estimates["logistic_text"][all_test.index] = classifier.predict(vx)
        format_estimates["majority"][all_test.index] = format_values.loc[a.index].mode().iat[0]
        print(f"Finished fold {fold}/5", flush=True)

    outputs = []
    for variant, p in predictions.items():
        assert np.all(np.isnan(p) | (p > 0))
        p[:] = np.sort(p, axis=1)
        pred = rows[["id", "group", "fold", "unit", "tax", "quote_kind", "source_kind", "y"]].copy()
        pred["variant"] = variant
        pred[["p25", "p50", "p75"]] = p
        outputs.append(pred)
    pd.concat(outputs).to_csv(args.out / "predictions_oof.csv", index=False)
    metrics = aggregate_metrics(rows, predictions)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    classify_formats(args.out, labels, format_estimates)
    print(metrics.loc[metrics.dimension.eq("total")].round(4).to_string(index=False), flush=True)

    # Refit useful paired candidates on all labeled cases. Publish schemas with weights.
    vocab, all_text = text_fit(text)
    a, _, reductions = dense_text(all_text, all_text, x.index, x.index)
    all_x = x_full.join(a).loc[rows.index]
    for name, cols in specs.items():
        encoded = [c for c in a if c.startswith("text_char_")] if name == "card_char" else list(a) if name == "full_text" else []
        cols = cols + CONTEXT + encoded
        model = fit(all_x, cols, rows.y)
        model.save_model(str(args.out / f"{name}.cbm"))
        (args.out / f"{name}.columns.json").write_text(json.dumps(cols, ensure_ascii=False), encoding="utf-8")
    for unit, case in rows.groupby("unit"):
        if len(case) >= 40 and case.group.nunique() >= 12:
            name = f"specialist_{unit}"
            cols = rules + CONTEXT
            fit(all_x.loc[case.index], cols, case.y).save_model(str(args.out / f"{name}.cbm"))
            (args.out / f"{name}.columns.json").write_text(json.dumps(cols, ensure_ascii=False), encoding="utf-8")
    joblib.dump((vocab, reductions), args.out / "text_encoders.joblib", compress=3)
    joblib.dump((ridge_bundle(x_full.loc[rows.index], rows, text, rules + CONTEXT, 0), rules + CONTEXT),
                args.out / "ridge.joblib", compress=3)
    matrix, encoder = sparse_features(x, all_text, card)
    classifier = LogisticRegression(C=2.0, max_iter=400, class_weight="balanced")
    classifier.fit(matrix, format_values)
    joblib.dump((classifier, encoder, card), args.out / "format_classifier.joblib", compress=3)


if __name__ == "__main__":
    main()
