import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    classification_report, RocCurveDisplay,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import (TRAIN_END_YEAR, TEST_START_YEAR, CV_SPLITS, RANDOM_SEED)
from features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from preprocessing import make_preprocessor

warnings.filterwarnings("ignore")

PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C":        [0.01, 0.1, 1, 10],
        "clf__penalty":  ["l1", "l2"],
        "clf__solver":   ["liblinear"],
        "clf__max_iter": [500],
    },
    "Random Forest": {
        "clf__n_estimators":     [100, 300],
        "clf__max_depth":        [None, 10],
        "clf__min_samples_leaf": [5, 20],
    },
    "XGBoost": {
        "clf__n_estimators":  [100, 300],
        "clf__max_depth":     [3, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample":     [0.8, 1.0],
    },
}


def split(df: pd.DataFrame):
    train = df[df["year"] <= TRAIN_END_YEAR].copy()
    test  = df[df["year"] >= TEST_START_YEAR].copy()
    X_train, y_train = train[ALL_FEATURES], train["y"]
    X_test,  y_test  = test[ALL_FEATURES],  test["y"]
    print(f"Train ≤{TRAIN_END_YEAR}: {len(train):,}  |  Test ≥{TEST_START_YEAR}: {len(test):,}")
    for name, y in [("train", y_train), ("test", y_test)]:
        if y.nunique() < 2:
            raise ValueError(f"Single class in {name} — check your data.")
    return X_train, X_test, y_train, y_test


def train_model(name: str, estimator, param_grid: dict, X_train, y_train):
    print(f"\n {name}\n")
    pipe = Pipeline([("pre", make_preprocessor()), ("clf", estimator)])
    grid = GridSearchCV(pipe, param_grid,
                        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
                        scoring="roc_auc", n_jobs=-1, verbose=0, refit=True)
    grid.fit(X_train, y_train)
    print(f"  Best params : {grid.best_params_}")
    print(f"  CV AUC      : {grid.best_score_:.4f}")
    cal = CalibratedClassifierCV(grid.best_estimator_, method="isotonic", cv="prefit")
    cal.fit(X_train, y_train)
    return cal


def train_all_models(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split(df)

    models_config = [
        ("Logistic Regression",
         LogisticRegression(random_state=RANDOM_SEED),
         PARAM_GRIDS["Logistic Regression"]),
        ("Random Forest",
         RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
         PARAM_GRIDS["Random Forest"]),
    ]
    if XGBOOST_AVAILABLE:
        models_config.append((
            "XGBoost",
            XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss",
                          tree_method="hist"),
            PARAM_GRIDS["XGBoost"],
        ))

    trained = {}
    for name, est, grid in models_config:
        trained[name] = train_model(name, est, grid, X_train, y_train)

    return trained, X_train, X_test, y_train, y_test


def baseline_rank(X_test, y_test):
    prob = 1 / (1 + np.exp(X_test["rank_diff"].fillna(0).values / 100))
    auc  = roc_auc_score(y_test, prob)
    print(f"  Baseline (rank only): AUC = {auc:.4f}")
    return auc, prob


def evaluate(name: str, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    ll    = log_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"\n  {name}  —  AUC {auc:.4f}  LogLoss {ll:.4f}  Brier {brier:.4f}")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=["A loses", "A wins"], digits=3))
    return {"name": name, "auc": auc, "logloss": ll, "brier": brier,
            "proba": proba, "pred": model.predict(X_test)}


def leakage_diagnostic(model, X_test, y_test):
    diff_cols = [c for c in X_test.columns if c.endswith("_diff")]
    X_inv = X_test.copy()
    X_inv[diff_cols] *= -1
    y_inv = 1 - y_test
    auc_n = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    auc_i = roc_auc_score(y_inv,  model.predict_proba(X_inv)[:, 1])
    print(f"  AUC normal   : {auc_n:.4f}")
    print(f"  AUC inverted : {auc_i:.4f}  (difference: {abs(auc_n - auc_i):.4f})")
    print(" No mirror leakage" if abs(auc_n - auc_i) < 0.02 else "!leakage?")


def plot_results(results: list, baseline_auc: float, baseline_prob, y_test):
    colors = ["#1a73e8", "#e8710a", "#0f9d58"]
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax = fig.add_subplot(gs[0])
    for res, color in zip(results, colors):
        RocCurveDisplay.from_predictions(y_test, res["proba"],
                                          name=f"{res['name']} ({res['auc']:.3f})",
                                          ax=ax, color=color)
    RocCurveDisplay.from_predictions(y_test, baseline_prob,
                                      name=f"Baseline ({baseline_auc:.3f})",
                                      ax=ax, color="#aaa", linestyle="--")
    ax.set_title("ROC curves"); ax.legend(fontsize=7)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for res, color in zip(results, colors):
        frac, mean_pred = calibration_curve(y_test, res["proba"], n_bins=10)
        ax2.plot(mean_pred, frac, "o-", color=color, label=res["name"], ms=4)
    ax2.set_title("Calibration")
    ax2.set_xlabel("Predicted probability"); ax2.set_ylabel("Actual frequency")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2]); ax3.axis("off")
    rows_data = [[r["name"], f"{r['auc']:.4f}", f"{r['logloss']:.4f}", f"{r['brier']:.4f}"]
                 for r in results]
    rows_data.append(["Baseline", f"{baseline_auc:.4f}", "—", "—"])
    tbl = ax3.table(cellText=rows_data, colLabels=["Model", "AUC", "LogLoss", "Brier"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
    ax3.set_title("Summary", pad=20)

    plt.suptitle("ATP Match Prediction", fontsize=13)
    plt.savefig("results/figures/model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, top_n: int = 20):
    try:
        inner     = model.calibrated_classifiers_[0].estimator
        xgb       = inner.named_steps["clf"]
        pre       = inner.named_steps["pre"]
        cat_names = (pre.named_transformers_["cat"]
                        .named_steps["ohe"]
                        .get_feature_names_out(CATEGORICAL_FEATURES).tolist())
        fi = pd.Series(xgb.feature_importances_, index=NUMERIC_FEATURES + cat_names)
        fi = fi.sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(9, 6))
        fi.sort_values().plot(kind="barh", ax=ax, color="#1a73e8")
        ax.set_title(f"Top {top_n} feature importances — XGBoost")
        plt.tight_layout()
        plt.savefig("results/figures/feature_importance.png", dpi=150)
        plt.show()
    except Exception as e:
        print(f"error: {e}")
