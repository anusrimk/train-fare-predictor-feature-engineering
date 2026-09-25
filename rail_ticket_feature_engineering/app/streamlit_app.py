import os
import sys

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(ROOT, "src"))

from feature_engineering import add_features
from ocr import ocr_text, parse_ticket_text


MODEL_PATH = os.path.join(ROOT, "models", "price_model.joblib")


st.set_page_config(
    page_title="RailLens",
    page_icon="🚆",
    layout="wide"
)

st.title("🚆 RailLens")
st.caption(
    "Turn a railway ticket into structured travel intelligence"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model not found. Please run `python src/train.py` first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

defaults = {
    "origin": "",
    "destination": "",
    "train_type": "",
    "train_class": "",
    "start_date": "",
    "end_date": "",
    "insert_date": "",
    "fare": "",
    "price": "",
    "ocr_done": False,
    "ocr_raw": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("🎫 Ticket Input")

mode = st.sidebar.radio(
    "Choose ticket source",
    [
        "Upload from gallery",
        "Take a photo"
    ]
)


# ---------------------------------------------------------
# IMAGE INPUT
# ---------------------------------------------------------

if mode == "Upload from gallery":

    file = st.file_uploader(
        "Upload your railway ticket",
        type=["png", "jpg", "jpeg"]
    )

else:

    file = st.camera_input(
        "Take a photo of your railway ticket"
    )


# ---------------------------------------------------------
# OCR
# ---------------------------------------------------------

if file is not None:

    image = Image.open(file)

    st.image(
        image,
        caption="Ticket uploaded",
        width=700
    )

    # Run OCR only once for this uploaded image
    if not st.session_state.ocr_done:

        with st.spinner("🔎 Reading ticket..."):

            raw_text = ocr_text(image)

            st.session_state.ocr_raw = raw_text

            parsed = parse_ticket_text(raw_text)

            for key, value in parsed.items():

                if value:
                    st.session_state[key] = value

            st.session_state.ocr_done = True

        st.success("✅ Ticket information extracted!")

    else:
        st.success("✅ Ticket information loaded!")


# ---------------------------------------------------------
# OCR DEBUG
# ---------------------------------------------------------

if st.session_state.ocr_raw:

    with st.expander("🔍 View raw OCR text"):

        st.text(
            st.session_state.ocr_raw
        )


# ---------------------------------------------------------
# TICKET DETAILS
# ---------------------------------------------------------

st.divider()

st.subheader("🎫 Extracted Ticket Details")

st.caption(
    "OCR automatically fills these fields. "
    "You can edit them if the ticket was read incorrectly."
)


col1, col2, col3 = st.columns(3)


with col1:

    st.text_input(
        "Origin",
        key="origin",
        placeholder="e.g. MADRID"
    )

    st.text_input(
        "Destination",
        key="destination",
        placeholder="e.g. BARCELONA"
    )

    st.text_input(
        "Train Type",
        key="train_type",
        placeholder="e.g. AVE"
    )


with col2:

    st.text_input(
        "Departure",
        key="start_date",
        placeholder="YYYY-MM-DD HH:MM:SS"
    )

    st.text_input(
        "Arrival",
        key="end_date",
        placeholder="YYYY-MM-DD HH:MM:SS"
    )

    st.text_input(
        "Class",
        key="train_class",
        placeholder="e.g. Turista"
    )


with col3:

    st.text_input(
        "Booking / Listing Time",
        key="insert_date",
        placeholder="YYYY-MM-DD HH:MM:SS"
    )

    st.text_input(
        "Fare Type",
        key="fare",
        placeholder="e.g. Flexible"
    )

    st.text_input(
        "Actual Ticket Price",
        key="price",
        placeholder="Optional"
    )


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

st.divider()

st.subheader("⚙️ Feature Engineering")

if st.button(
    "⚙️ Generate Features",
    type="secondary"
):

    raw = pd.DataFrame([{

        "Unnamed: 0": 0,

        "insert_date":
            st.session_state.insert_date,

        "origin":
            st.session_state.origin,

        "destination":
            st.session_state.destination,

        "start_date":
            st.session_state.start_date,

        "end_date":
            st.session_state.end_date,

        "train_type":
            st.session_state.train_type,

        "train_class":
            st.session_state.train_class,

        "fare":
            st.session_state.fare,

        "price": 0

    }])

    features = add_features(raw)

    st.session_state.engineered_features = features


if "engineered_features" in st.session_state:

    features = st.session_state.engineered_features

    st.success(
        f"Generated {features.shape[1]} engineered features."
    )

    st.dataframe(
        features.T.rename(
            columns={0: "Value"}
        ),
        use_container_width=True
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.divider()

st.subheader("🤖 Machine Learning Prediction")

if st.button(
    "🔮 Predict Ticket Price",
    type="primary"
):

    raw = pd.DataFrame([{

        "Unnamed: 0": 0,

        "insert_date":
            st.session_state.insert_date,

        "origin":
            st.session_state.origin,

        "destination":
            st.session_state.destination,

        "start_date":
            st.session_state.start_date,

        "end_date":
            st.session_state.end_date,

        "train_type":
            st.session_state.train_type,

        "train_class":
            st.session_state.train_class,

        "fare":
            st.session_state.fare,

        "price": 0

    }])

    try:

        X = add_features(raw)

        prediction = float(
            model.predict(X)[0]
        )

        st.success(
            f"### Predicted Ticket Price: €{prediction:,.2f}"
        )

        st.info(
            "Prediction generated using the trained "
            "Feature Engineering + Random Forest pipeline."
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------

st.divider()

st.subheader("🧠 How RailLens Works")

st.markdown("""
```text
Ticket Image
     ↓
OCR
     ↓
Structured Ticket Data
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Temporal Features
     ↓
Route Features
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Random Forest
     ↓
Fare Prediction 
""")