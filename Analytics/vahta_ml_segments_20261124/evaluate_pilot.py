"""Small fixed-model experiment on published-ad labels, with grouped OOF validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'

def weighted_median(x, w):
    order = np.argsort(x)
    return float(np.asarray(x)[order][np.searchsorted(np.cumsum(np.asarray(w)[order]), np.sum(w) / 2)])

def main():
    allowlist = json.loads((OUT / 'feature_allowlist.json').read_text())
    predictions, metrics = [], []
    for segment in ['loading', 'packing', 'picking', 'warehouse_pool']:
        for target, suffix in [('y_floor_net_shift_rub', ''), ('y_midpoint_closed_net_shift_rub', '_point')]:
            d = pd.read_csv(OUT / (segment + suffix + '.csv'), keep_default_na=False)
            y = d[target].astype(float).to_numpy()
            weight = d.training_weight.to_numpy(float)
            x = d[allowlist].astype(str).to_dict('records')
            estimates = {name: np.full(len(d), np.nan) for name in ['median', 'ridge', 'forest']}
            for fold in sorted(d.cv_fold.unique()):
                train, test = np.flatnonzero(d.cv_fold != fold), np.flatnonzero(d.cv_fold == fold)
                assert not set(d.iloc[train].split_group) & set(d.iloc[test].split_group)
                fallback = weighted_median(y[train], weight[train])
                role_medians = {}
                for role in d.iloc[train].segment_role.unique():
                    ii = train[d.iloc[train].segment_role.to_numpy() == role]
                    role_medians[role] = weighted_median(y[ii], weight[ii])
                estimates['median'][test] = [role_medians.get(d.iloc[i].segment_role, fallback) for i in test]
                vectorizer = DictVectorizer()
                xt = vectorizer.fit_transform([x[i] for i in train])
                xv = vectorizer.transform([x[i] for i in test])
                for name, model in [('ridge', Ridge(alpha=10, solver='lsqr')),
                                    ('forest', RandomForestRegressor(n_estimators=120, max_depth=5,
                                                                   min_samples_leaf=5, random_state=20261124, n_jobs=2))]:
                    model.fit(xt, np.log(y[train]), sample_weight=weight[train])
                    estimates[name][test] = np.exp(model.predict(xv))
            # In a closed-only view, rebalance weights over the actual evaluated rows.
            ew = (1 / d.groupby('split_group').id.transform('size')).to_numpy()
            base_error = np.abs(estimates['median'] - y)
            baseline_mae = np.average(base_error, weights=ew)
            for name, p in estimates.items():
                assert np.isfinite(p).all()
                err = np.abs(p - y)
                groups = d.assign(error=err, base_error=base_error).groupby('split_group')[['error', 'base_error']].mean()
                rng = np.random.default_rng(20261124)
                idx = rng.integers(0, len(groups), size=(1000, len(groups)))
                delta = (groups.base_error.to_numpy()[idx].mean(axis=1) - groups.error.to_numpy()[idx].mean(axis=1))
                lo, hi = np.quantile(delta, [.025, .975])
                metrics.append({'segment': segment, 'target': target, 'model': name, 'rows': len(d),
                                'groups': d.split_group.nunique(), 'folds': d.cv_fold.nunique(),
                                'mae_rub_group_weighted': np.average(err, weights=ew),
                                'median_ape_group_weighted_pct': weighted_median(err / y, ew) * 100,
                                'wmape_group_weighted_pct': np.average(err, weights=ew) / np.average(y, weights=ew) * 100,
                                'mae_improvement_vs_median_pct': (1 - np.average(err, weights=ew) / baseline_mae) * 100,
                                'mae_gain_rub_bootstrap_low': lo, 'mae_gain_rub_bootstrap_high': hi,
                                'baseline': 'weighted_train_role_median_fallback_global_train',
                                'scope': 'advertised labels; not factual income accuracy'})
                for i, row in d.iterrows():
                    predictions.append({'segment': segment, 'target': target, 'model': name, 'id': row.id,
                                        'split_group': row.split_group, 'cv_fold': row.cv_fold,
                                        'y': y[i], 'prediction_oof': p[i], 'eval_weight': ew[i]})
    pd.DataFrame(predictions).to_csv(OUT / 'pilot_oof_predictions.csv', index=False)
    m = pd.DataFrame(metrics)
    m.to_csv(OUT / 'pilot_metrics.csv', index=False)
    print(m[['segment', 'target', 'model', 'rows', 'groups', 'mae_rub_group_weighted',
             'median_ape_group_weighted_pct', 'mae_improvement_vs_median_pct']].round(2).to_string(index=False))

if __name__ == '__main__':
    main()
