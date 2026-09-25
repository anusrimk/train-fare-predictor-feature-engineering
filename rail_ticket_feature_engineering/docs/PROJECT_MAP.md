# Project-to-Syllabus Map

| Course topic | Project implementation |
|---|---|
| Foundations | Raw CSV → engineered features → model |
| Data ingestion | pandas CSV loader |
| Train/test split | `train_test_split` |
| Missingness | profile + indicators |
| Scaling | RobustScaler |
| Encoding | OneHotEncoder |
| Transformation | log1p on skewed price-related numeric inputs where appropriate |
| Time features | departure/arrival hour, weekday, month, season |
| Feature creation | duration, advance booking, route frequency, city frequency, interactions |
| Curse of dimensionality | high-cardinality route/categorical discussion |
| Filter selection | variance threshold + mutual information |
| Wrapper selection | optional RFE |
| Embedded selection | Random Forest / HistGradientBoosting importance |
| Interpretability | permutation importance; optional SHAP |
| Dimensionality reduction | PCA demo notebook |
| Leakage | split before fitting learned preprocessing; pipeline |
| Reproducibility | fixed random states + persisted pipeline |
| Feature consistency | same `FeatureEngineer` used during training/inference |
| Feature metadata | `docs/FEATURE_DICTIONARY.md` |
| ML evaluation | MAE, RMSE, R², cross-validation |
| Deployment preparation | joblib model + Streamlit interface |
| OCR interface | upload/camera → OCR → editable fields |

## Deliberately not forced
Some syllabus items are not naturally useful for this problem (e.g. image HOG/CNN features, text BoW, lag/rolling statistics for time-series). They are better kept as optional extensions rather than artificially inserted into the model.
