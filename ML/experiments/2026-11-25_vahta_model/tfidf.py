"""Measure the incremental contribution of character TF-IDF on fixed folds."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from calibrate import widen
from train import HERE, ROOT, fit, make_features, predict, rules, score, strict_redact

TEXT = ["title", "description", "Что получают работники"]


def safe_text(value):
    return re.sub(r"\d+(?:[ ,.]\d+)*", " ", strict_redact(rules.clean_description(value)))


def fold_vectors(train, test):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=150_000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(train)
    svd = TruncatedSVD(n_components=min(32, matrix.shape[1] - 1, matrix.shape[0] - 1), random_state=42)
    with threadpool_limits(limits=3):
        a = svd.fit_transform(matrix)
        b = svd.transform(vectorizer.transform(test))
    columns = [f"tfidf_svd_{j:02d}" for j in range(a.shape[1])]
    return (pd.DataFrame(a, index=train.index, columns=columns),
            pd.DataFrame(b, index=test.index, columns=columns), (vectorizer, svd, columns))


def transform(text, encoder):
    vectorizer, svd, columns = encoder
    values = svd.transform(vectorizer.transform(text))
    return pd.DataFrame(values, index=text.index, columns=columns)


def calibrated_model(rows, blocks, text, cols, seed):
    split = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    fit_idx, cal_idx = next(split.split(rows, rows.y, rows.group))
    training, calibration = rows.iloc[fit_idx], rows.iloc[cal_idx]
    assert set(training.group).isdisjoint(calibration.group)
    a, b, encoder = fold_vectors(text.loc[training.index], text.loc[calibration.index])
    all_cols = cols + encoder[2]
    model = fit(blocks.loc[a.index].join(a), all_cols, training.y)
    p = np.log(predict(model, blocks.loc[b.index].join(b), all_cols))
    y = np.log(calibration.y.to_numpy())
    residuals = np.maximum(p[:, 0] - y, y - p[:, 2])
    k = min(len(residuals) - 1, int(np.ceil((len(residuals) + 1) * .50)) - 1)
    return model, encoder, max(0., float(np.sort(residuals)[k]))


def evaluate_calibration(folder, d, blocks, text, cols):
    raw, calibrated = np.empty((len(d), 3)), np.empty((len(d), 3))
    for fold in sorted(d.fold.unique()):
        test = d.fold.eq(fold)
        model, encoder, radius = calibrated_model(d.loc[~test], blocks, text, cols, 42 + int(fold))
        vectors = transform(text.loc[d.loc[test].index], encoder)
        p = predict(model, blocks.loc[vectors.index].join(vectors), cols + encoder[2])
        raw[test], calibrated[test] = p, widen(p, radius)
    cohort = "shift_net_lower_bound"
    variants = [(raw, "card_char_tfidf_split_raw"), (calibrated, "card_char_tfidf_calibrated")]
    summary = pd.DataFrame([dict(cohort=cohort, variant=name, **score(d.y.to_numpy(), p, d.group.to_numpy()))
                            for p, name in variants])
    outputs = []
    for p, name in variants:
        out = d[["group", "y", "origin", "fold", "cohort"]].copy()
        out["variant"] = name
        out[["p25", "p50", "p75"]] = p
        outputs.append(out)
    pd.concat(outputs).to_csv(folder / "predictions_tfidf_calibrated_oof.csv")
    summary.to_csv(folder / "metrics_tfidf_calibrated.csv", index=False)
    model, encoder, radius = calibrated_model(d, blocks, text, cols, 42)
    stem = folder / f"{cohort}__card_char_tfidf_calibrated"
    model.save_model(str(stem.with_suffix(".cbm")))
    joblib.dump(encoder, stem.with_suffix(".encoder.joblib"), compress=3)
    stem.with_suffix(".calibration.json").write_text(
        json.dumps({"column_variant": "card", "log_radius": radius, "tfidf": True}, indent=2), encoding="utf-8"
    )
    print(summary.round(4).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "Data/dataset/vacancies.csv")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate the card + TF-IDF model for quoted minima")
    args = parser.parse_args()
    folder = args.results
    rows = pd.read_csv(folder / "predictions_oof.csv", dtype={"id": str})
    rows = rows.loc[rows.variant.eq("role_median")].set_index("id")
    raw = pd.read_csv(args.data, dtype=str, keep_default_na=False).set_index("id").loc[rows.index]
    blocks, _ = make_features(raw)
    raw_text = raw[TEXT].agg("\n".join, axis=1).map(safe_text)
    schema = json.loads((folder / "feature_schema.json").read_text(encoding="utf-8"))
    if args.calibrate:
        evaluate_calibration(folder, rows.loc[rows.cohort.eq("shift_net_lower_bound")],
                             blocks, raw_text, schema["card"])
        return
    metrics, predictions = [], []
    for cohort, d in rows.groupby("cohort", sort=False):
        variants = ["card", "plus_m4_conditions"]
        pred = {key: np.empty((len(d), 3)) for key in variants}
        for fold in sorted(d.fold.unique()):
            test = d.fold.eq(fold)
            train_rows, test_rows = d.loc[~test], d.loc[test]
            assert set(train_rows.group).isdisjoint(test_rows.group)
            a, b, encoder = fold_vectors(raw_text.loc[train_rows.index], raw_text.loc[test_rows.index])
            x_train = blocks.loc[a.index].join(a)
            x_test = blocks.loc[b.index].join(b)
            for key in variants:
                cols = schema[key] + encoder[2]
                model = fit(x_train, cols, train_rows.y)
                pred[key][test] = predict(model, x_test, cols)
            print(cohort, "fold", fold, "done", flush=True)
        for key, p in pred.items():
            variant = key + "_char_tfidf"
            metrics.append(dict(cohort=cohort, variant=variant,
                                **score(d.y.to_numpy(), p, d.group.to_numpy())))
            out = d[["group", "y", "origin", "fold", "cohort"]].copy()
            out["variant"] = variant
            out[["p25", "p50", "p75"]] = p
            predictions.append(out)
    pd.concat(predictions).to_csv(folder / "predictions_tfidf_oof.csv")
    summary = pd.DataFrame(metrics)
    summary.to_csv(folder / "metrics_tfidf.csv", index=False)
    print(summary.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
