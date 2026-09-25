import os
import json
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(ROOT, "data", "train-ticket-price.csv")
MODEL_DIR = os.path.join(ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))

from feature_engineering import add_features


# =========================================================
# 1. LOAD DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA)

print("Original shape:", df.shape)

df = df.drop_duplicates()

df["price"] = pd.to_numeric(df["price"], errors="coerce")

df = df.dropna(subset=["price"]).copy()

print("Cleaned shape:", df.shape)


# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

y_train = train_df["price"]
y_test = test_df["price"]


# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================

print("\nRunning feature engineering...")

X_train = add_features(train_df.copy())
X_test = add_features(test_df.copy())

print("Feature-engineered shape:", X_train.shape)


# =========================================================
# 4. DETERMINE DATA TYPES
# =========================================================

# Anything that is genuinely numeric goes to the
# numeric pipeline.

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()


# Everything else is treated as categorical.
#
# This is important because fields such as:
#
# MADRID
# BARCELONA
# MADRID → BARCELONA
#
# must NEVER go through a median imputer.

categorical_features = [
    col for col in X_train.columns
    if col not in numeric_features
]


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# =========================================================
# 5. SAFETY CHECK
# =========================================================

for col in numeric_features:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )


print("\nNumeric validation:")

for col in numeric_features:

    if not pd.api.types.is_numeric_dtype(X_train[col]):

        raise TypeError(
            f"Column '{col}' is not numeric."
        )

print("PASSED")


# =========================================================
# 6. NUMERIC PIPELINE
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            RobustScaler()
        )
    ]
)


# =========================================================
# 7. CATEGORICAL PIPELINE
# =========================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


# =========================================================
# 8. COLUMN TRANSFORMER
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# =========================================================
# 9. ML MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 10. COMPLETE ML PIPELINE
# =========================================================

pipe = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# =========================================================
# 11. TRAIN
# =========================================================

print("\nTraining model...")

pipe.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# 12. PREDICT
# =========================================================

predictions = pipe.predict(X_test)


# =========================================================
# 13. EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("=" * 50)


# =========================================================
# 14. SAVE MODEL
# =========================================================

model_path = os.path.join(
    MODEL_DIR,
    "price_model.joblib"
)

joblib.dump(
    pipe,
    model_path
)


# =========================================================
# 15. SAVE METRICS
# =========================================================

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

with open(
    os.path.join(MODEL_DIR, "metrics.json"),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print("\nModel saved:")
print(model_path)

print("\nTraining finished successfully.")