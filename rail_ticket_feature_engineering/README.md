# RailTicket FE — Feature Engineering + ML

An end-to-end Feature Engineering project built around a train-ticket price dataset.

## Important dataset note
The supplied `train-ticket-price.csv` is a Spanish rail ticket dataset (e.g. Madrid/Barcelona/Ponferrada-style routes), while `stations.json` and `trains.json` describe Indian railway stations/trains. Because the geographic systems do not share a common key, the project **does not falsely join them**. The Spanish ticket CSV is the primary ML dataset. The Indian JSON files are retained as an optional future route-enrichment module.

## Main objective
**Predict train ticket price** from raw booking/journey information while demonstrating substantial Feature Engineering.

### Feature Engineering topics covered
- Data ingestion and schema inspection
- Train/test split
- Missing-value analysis + missing indicators
- Duplicate/invalid record checks
- Date/time parsing
- Temporal features: hour, weekday, month, day-of-year, weekend, season
- Journey duration
- Advance booking time
- Route-level frequency features
- Origin/destination frequency features
- Interaction/cross features
- Log transformation of skewed numeric variables
- One-hot encoding
- Scaling with RobustScaler
- Variance filtering
- Mutual information feature selection
- Embedded feature importance (tree model)
- Permutation importance
- Optional RFE
- PCA demonstration for dimensionality reduction
- Leakage-safe sklearn Pipeline + ColumnTransformer
- Cross-validation and regression metrics
- Residual/error analysis
- Model persistence with joblib
- Streamlit inference UI
- Ticket image upload
- Ticket camera capture (works in Streamlit; Docker can be added later)
- OCR extraction scaffold with graceful fallback
- Explainable prediction output

## ML target
`price` (regression).

## App workflow
1. Upload a ticket image **or** use the camera input.
2. OCR attempts to extract ticket fields.
3. If OCR is unavailable/uncertain, the app lets you enter/edit the extracted fields manually.
4. Feature engineering transforms the ticket into model-ready features.
5. The trained model predicts the ticket price.
6. The app displays the extracted/entered journey details and prediction.

## Run
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
```

The training script creates `models/price_model.joblib`.

## Later Docker
Docker is intentionally **not included as a required part of the current implementation**, because the requested project is to be built first and containerized later.

## Suggested presentation title
**RailLens: Feature Engineering for Intelligent Train Ticket Analytics**
