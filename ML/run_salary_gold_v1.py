"""Evaluate the analysts' 176-hour gross scenario on identical vacancy IDs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re

import numpy as np
import pandas as pd

import salary_gold as g
from run_salary_gold import RUN, VARIANTS, add_predictions, eval_predictions, paired, scores_by_scope


def main():
    out = g.ROOT / "ML/experiments" / RUN
    out.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((g.ROOT / "Analytics/salary_normalization_v1.ipynb").read_text(encoding="utf-8"))
    ns = dict(Counter=Counter, dataclass=dataclass, math=math, re=re, np=np, pd=pd, __name__="__main__")
    exec(compile("".join(notebook["cells"][5]["source"]), "analytics normalization v1", "exec"), ns)
    canonical, _ = g.load_snapshot()
    data = g.prepare(canonical)
    data = data[data.evaluation_fold.ge(0) & data.unit.eq("shift") & data.tax.eq("net")].copy()
    hours = data[ns["HOURS_COLUMN"]].map(ns["_parse_hours"])
    data["v1_hours"] = hours.map(lambda value: value[0][0] if value[1] is None and len(value[0]) == 1 else np.nan)
    data = data.loc[data.v1_hours.notna()]
    config = ns["SalaryConfig"](unspecified_as_month=False, split_range_by_hours=False, impute_missing_hours=False)
    normalized = ns["normalize_salaries"](data, config, print_stats=False).set_index(["source", "id"])
    assert normalized.index.is_unique
    idx = pd.MultiIndex.from_frame(data[["source", "id"]])
    y = normalized["salary_month_gross_normalized"].reindex(idx).to_numpy(dtype=float)
    check = np.array([ns["_annual_net_to_gross"](value * 176 / hours * 12) / 12
                      for value, hours in zip(data.y, data.v1_hours)])
    np.testing.assert_allclose(check[np.isfinite(y)], y[np.isfinite(y)])
    data = data.assign(v1_gross=y, label=data.y)
    data = data[data.v1_gross.notna()]
    normalized_rows = []
    for fold in range(3):
        train, val, test = g.split(data, fold)
        m = VARIANTS["cat_detail"]
        native = g.fit(train, val, **m)
        standardized = g.fit(train.assign(label=train.v1_gross), val.assign(label=val.v1_gross), **m)
        prediction = g.predict(test, native)
        mapped = np.array([[ns["_annual_net_to_gross"](value * 176 / hours * 12) / 12
                            for value in row] for row, hours in zip(prediction, test.v1_hours)])
        for name, values in [("native_to_v1_gross", mapped), ("v1_gross_target", g.predict(test, standardized))]:
            add_predictions(normalized_rows, test, values, name=name, fold=fold,
                            branch="v1_gross_176h", target=test.v1_gross,
                            v1_hours=test.v1_hours)
        print(f"V1 fold {fold}: {len(train)}/{len(val)}/{len(test)}", flush=True)
    results = pd.concat(normalized_rows, ignore_index=True)
    results.to_csv(out / "v1_predictions.csv", index=False)
    eval_predictions(results).to_csv(out / "v1_metrics.csv", index=False)
    pairs = [paired(results, "v1_gross_176h", "v1_gross_target", "native_to_v1_gross", scope, mask)
             for scope, mask in scores_by_scope(results)]
    pd.DataFrame([r for r in pairs if r]).to_csv(out / "v1_paired.csv", index=False)


if __name__ == "__main__":
    main()
