"""Un modelo lineal sencillo y dos configuraciones pequeñas de LightGBM."""
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from .features import FEATURES, CATEGORICAL


def build_model(name, config):
    if name == "baseline_semanal":
        return None
    cat_cols = [FEATURES.index(col) for col in CATEGORICAL]
    if name == "ridge":
        numeric = [i for i in range(len(FEATURES)) if i not in cat_cols]
        preprocessing = ColumnTransformer([
            ("numericas", StandardScaler(), numeric),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", dtype="float32"), cat_cols)
        ], sparse_threshold=1.0)
        return Pipeline([("preprocesado", preprocessing),
                         ("modelo", Ridge(alpha=config["ridge_alpha"], solver="lsqr", tol=1e-4, max_iter=300))])
    params = config["lightgbm_candidates"][name]
    return LGBMRegressor(objective="regression_l1", learning_rate=config["learning_rate"],
                         min_child_samples=config["min_child_samples"], n_jobs=config["n_jobs"],
                         random_state=config["random_state"], deterministic=True, force_col_wise=True,
                         verbosity=-1, **params)


def fit_model(model, name, x, y):
    if name.startswith("lightgbm"):
        model.fit(x, y, feature_name=FEATURES,
                  categorical_feature=[FEATURES.index(c) for c in CATEGORICAL])
    elif model is not None:
        model.fit(x, y)
    return model
