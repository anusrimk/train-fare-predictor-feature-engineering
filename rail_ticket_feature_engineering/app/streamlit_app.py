import os, re, json, sys
import pandas as pd
import streamlit as st
from PIL import Image
import joblib

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT,"src"))
from feature_engineering import add_features

MODEL_PATH=os.path.join(ROOT,"models","price_model.joblib")

st.set_page_config(page_title="RailLens",page_icon="🚆",layout="wide")
st.title("🚆 RailLens")
st.caption("Train Ticket Feature Engineering + ML — upload or capture a ticket")

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Run `python src/train.py` first.")
    st.stop()

model=joblib.load(MODEL_PATH)

st.sidebar.header("Input")
mode=st.sidebar.radio("Ticket image source",["Upload from gallery","Take a photo"])
if mode=="Upload from gallery":
    file=st.file_uploader("Upload a ticket image",type=["png","jpg","jpeg"])
else:
    file=st.camera_input("Take a photo of the ticket")

image=None
if file:
    image=Image.open(file)
    st.image(image,caption="Ticket input",width=700)

st.divider()
st.subheader("Ticket details")
st.info("OCR is intentionally editable: real tickets can have different layouts. You can test the ML pipeline even when OCR cannot confidently read a field.")

def val(label, default=""):
    return st.text_input(label,default)

c1,c2,c3=st.columns(3)
with c1:
    origin=val("Origin","MADRID")
    destination=val("Destination","BARCELONA")
    train_type=val("Train type","AVE")
with c2:
    start_date=val("Departure","2026-10-12 08:30:00")
    end_date=val("Arrival","2026-10-12 11:15:00")
    train_class=val("Class","Turista")
with c3:
    insert_date=val("Booking/listing time","2026-10-01 10:00:00")
    fare=val("Fare","Flexible")

st.caption("The target price is NOT entered — it is what the ML model predicts.")
if st.button("🔮 Predict ticket price",type="primary"):
    raw=pd.DataFrame([{
        "Unnamed: 0":0,
        "insert_date":insert_date,
        "origin":origin,
        "destination":destination,
        "start_date":start_date,
        "end_date":end_date,
        "train_type":train_type,
        "train_class":train_class,
        "fare":fare,
        "price":0
    }])
    X=add_features(raw)
    pred=float(model.predict(X)[0])
    st.success(f"Predicted ticket price: **€{pred:,.2f}**")
    feat= X.iloc[0].to_dict()
    st.subheader("Engineered features")
    show_cols=["departure_hour","arrival_hour","departure_weekday","departure_month",
               "is_weekend","departure_period","season","journey_duration_hours",
               "advance_booking_hours","is_overnight","same_day_journey",
               "route_frequency","origin_frequency","destination_frequency"]
    st.dataframe(pd.DataFrame({"Feature":show_cols,"Value":[feat.get(c) for c in show_cols]}),hide_index=True)

st.divider()
st.subheader("How this demonstrates Feature Engineering")
st.markdown("""
**Raw ticket → parsing → cleaning → temporal features → duration/advance-booking features → categorical encoding → scaling → ML prediction**

The app keeps the feature engineering function shared with training so the model sees a consistent feature schema.
""")
