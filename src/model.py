"""LightGBM training with KFold OOF for threshold calibration."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold


def get_params():
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "verbose": -1,
        "is_unbalance": True,
        "seed": 42,
    }


def train_lgb(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Train with KFold OOF. Returns (models, oof_predictions)."""
    oof = np.zeros(len(y))
    models = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    params = get_params()
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        dtrain = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
        dvalid = lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx])
        model = lgb.train(
            params, dtrain, num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )
        oof[va_idx] = model.predict(X.iloc[va_idx])
        models.append(model)
        print(f"Fold {fold}: best_iter={model.best_iteration}")
    
    return models, oof