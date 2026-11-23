"""sklearn preprocessing pipeline (ColumnTransformer).

The preprocessor is always placed INSIDE a Pipeline together with the model, so that
imputation medians, scaling statistics and one-hot categories are learned from the
training rows only (in cross-validation: from each fold's training rows only).

* History/calendar numeric features: median imputation + missing-value indicator
  (a missing lag is informative), optional standard scaling (needed by linear models).
* Survey numeric features: median imputation (no per-column indicator; the single
  ``survey_available`` flag already says whether the survey can be used).
* Survey categorical features: missing -> "Unavailable", then one-hot encoding.
  ``handle_unknown="ignore"`` means a category never seen in training is encoded as all
  zeros instead of crashing at prediction time; rare categories (< 20 rows) are grouped.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import FEATURE_SETS, FORBIDDEN_FEATURES, SURVEY_NUMERIC


def build_preprocessor(feature_set: str = "history_only", scale: bool = False) -> ColumnTransformer:
    spec = FEATURE_SETS[feature_set]
    numeric, categorical = spec["numeric"], spec["categorical"]
    bad = (set(numeric) | set(categorical)) & FORBIDDEN_FEATURES
    if bad:
        raise ValueError(f"Forbidden columns in feature set: {bad}")

    history_like = [c for c in numeric if c not in SURVEY_NUMERIC]
    survey_num = [c for c in numeric if c in SURVEY_NUMERIC]

    def num_pipe(indicator: bool) -> Pipeline:
        # keep_empty_features=True: a column that is entirely missing in the training rows is kept
        # (filled with 0) instead of silently dropped. This happens for survey columns in early
        # time-ordered folds, when no household had been interviewed yet (survey_available = 0 for
        # all rows). Dropping columns would change the model's input shape between folds.
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=indicator, keep_empty_features=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers = [("history", num_pipe(indicator=True), history_like)]
    if survey_num:
        transformers.append(("survey_num", num_pipe(indicator=False), survey_num))
    if categorical:
        transformers.append(("survey_cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unavailable", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                     sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def model_input_columns(feature_set: str) -> list[str]:
    spec = FEATURE_SETS[feature_set]
    return spec["numeric"] + spec["categorical"]
