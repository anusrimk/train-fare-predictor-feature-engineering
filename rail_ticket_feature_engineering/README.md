
# 🚆 RailLens — Railway Ticket Intelligence
> **Turn every railway ticket into travel intelligence.**

RailLens is a machine-learning application that transforms railway ticket images into structured travel data using OCR, applies feature engineering to derive meaningful journey attributes, and uses a trained machine-learning model to estimate ticket prices.

Instead of treating a railway ticket as a static document, RailLens treats it as a **dataset** that can be extracted, transformed, analyzed, and used for machine learning.

---

## ✨ What Does RailLens Do?

A user can:

- 📁 Upload a railway ticket image from their gallery
- 📷 Capture a railway ticket using their camera
- 🔎 Extract ticket information using OCR
- 🧹 Clean and structure the extracted information
- ⚙️ Generate engineered features
- 🤖 Run the data through a machine-learning pipeline
- 💰 Predict the ticket price
- 📊 Inspect the generated features

### End-to-End Flow

```text
              Railway Ticket
                    │
                    ▼
             📷 Image Input
                    │
                    ▼
              🔎 OCR Engine
                    │
                    ▼
          Ticket Information
                    │
                    ▼
          Data Preprocessing
                    │
                    ▼
         ⚙️ Feature Engineering
                    │
                    ▼
        Machine Learning Pipeline
                    │
                    ▼
          🤖 Fare Prediction
                    │
                    ▼
             📊 Insights
````

---

# 🎯 Problem Statement

Railway tickets contain valuable information such as:

* Origin
* Destination
* Train type
* Class
* Departure time
* Arrival time
* Journey date
* Fare

However, this information is usually locked inside an image or document.

A railway ticket is designed primarily for **booking and verification**, rather than analysis.

RailLens explores how this unstructured ticket information can be converted into structured data and enriched through Feature Engineering.

### The central idea

> **A railway ticket is more than a receipt — it is a dataset.**

---

# 💡 Product Vision

RailLens aims to create a travel intelligence layer on top of railway tickets.

```text
                 Railway Ticket
                        │
                        ▼
                       OCR
                        │
                        ▼
                Structured Data
                        │
                        ▼
               Feature Engineering
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Journey        Pricing       Passenger
      Analytics      Insights      Analytics
          │             │             │
          └─────────────┼─────────────┘
                        ▼
               Travel Intelligence
```

The current prototype focuses on:

* OCR
* Ticket information extraction
* Feature Engineering
* Machine Learning
* Fare prediction

---

# 🧠 Feature Engineering

Feature Engineering is the core of this project.

Instead of directly passing raw ticket information into a machine-learning model, RailLens transforms the information into features that can better represent the journey.

## Raw Ticket Information

```text
Origin
Destination
Departure
Arrival
Train Type
Class
Fare
Price
```

## Engineered Features

Depending on the available ticket information, the pipeline can derive features such as:

```text
Departure hour
Arrival hour
Journey duration
Departure period
Arrival period
Overnight journey
Day of week
Month
Weekend indicator
Route characteristics
Fare-related features
```

The overall transformation is:

```text
RAW DATA
   │
   ▼
DATA CLEANING
   │
   ▼
FEATURE CREATION
   │
   ▼
ENCODING
   │
   ▼
SCALING
   │
   ▼
MACHINE LEARNING MODEL
```

---

# 🔬 Feature Engineering Techniques

## Data Cleaning

The pipeline includes:

* Duplicate removal
* Missing-value handling
* Data-type validation
* Numeric conversion

## Missing Value Handling

Numerical features use:

```text
Median Imputation
```

Categorical features use:

```text
Most-Frequent Imputation
```

Numerical missing indicators are also supported.

## Numerical Transformation

The training pipeline uses:

```text
RobustScaler
```

This reduces the influence of extreme numerical values.

## Categorical Encoding

Categorical variables are processed using:

```text
OrdinalEncoder
```

Unknown categories are handled safely during inference.

## Feature Creation

The feature-engineering layer is designed to derive:

* Temporal features
* Journey-duration features
* Route-related features
* Fare-related features

---

# 🤖 Machine Learning

The current prototype uses:

## Random Forest Regression

The model is trained to predict ticket price.

```python
RandomForestRegressor
```

Training configuration:

* 300 trees
* Maximum depth: 15
* Minimum samples per leaf: 2
* Random state: 42
* Parallel processing enabled

## Evaluation Metrics

The model is evaluated using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

The metrics are stored in:

```text
models/metrics.json
```

---

# 📷 OCR Pipeline

RailLens accepts ticket images and extracts their text using OCR.

The OCR stack is:

```text
Ticket Image
     ↓
Tesseract OCR
     ↓
pytesseract
     ↓
Raw Text
     ↓
Ticket Parser
     ↓
Structured Ticket Data
```

For example, OCR may extract:

```text
ORIGIN DESTINATION
MADRID BARCELONA

DEPARTURE ARRIVAL
2026-10-12 08:30:00
2026-10-12 11:15:00

TRAIN TYPE CLASS
AVE Turista

FARE PRICE
Flexible 89.50 EUR
```

This is converted into structured fields:

| Field       | Value            |
| ----------- | ---------------- |
| Origin      | MADRID           |
| Destination | BARCELONA        |
| Train Type  | AVE              |
| Class       | Turista          |
| Departure   | 2026-10-12 08:30 |
| Arrival     | 2026-10-12 11:15 |
| Fare Type   | Flexible         |
| Price       | 89.50 EUR        |

---

# 🖥️ Application

The interface is built using **Streamlit**.

The application provides:

### 📁 Upload from Gallery

Upload:

```text
PNG
JPG
JPEG
```

### 📷 Camera Input

Use the device camera to capture a ticket.

### 🔎 OCR Extraction

The application automatically extracts ticket information.

### ✏️ Editable Fields

Extracted fields can be manually corrected if OCR makes a mistake.

### ⚙️ Feature Engineering

The application displays the engineered features generated from the ticket.

### 🤖 Fare Prediction

The processed ticket is passed through the trained machine-learning pipeline.

---

# 🏗️ Project Structure

```text
rail_ticket_feature_engineering/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── train-ticket-price.csv
│
├── models/
│   ├── price_model.joblib
│   └── metrics.json
│
├── src/
│   ├── feature_engineering.py
│   ├── ocr.py
│   └── train.py
│
├── demo/
│   └── demo_ticket.png
│
├── requirements.txt
└── README.md
```

---

# 🔄 Complete ML Pipeline

```text
                    ┌─────────────────┐
                    │  Ticket Image   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Tesseract     │
                    │      OCR        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Ticket Parser   │
                    └────────┬────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Structured Ticket    │
                  │ Data                 │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Engineering │
                  └──────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
          Temporal       Numerical      Categorical
           Features       Features        Features
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Preprocessing        │
                  │ Pipeline             │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Random Forest        │
                  │ Regression           │
                  └──────────┬───────────┘
                             │
                             ▼
                    💰 Fare Prediction
```

---

# 📊 Dataset

The current fare-prediction training pipeline uses:

```text
data/train-ticket-price.csv
```

This dataset is used to train the machine-learning model for ticket-price prediction.

The project also includes the concept of integrating railway station and train datasets to enrich the ticket with additional journey-level features such as:

* Distance
* Route
* Journey duration
* Average speed
* Number of stops
* Origin/destination zones
* State-level route information

These enrichments represent the next stage of the project and should only be considered part of the prediction pipeline once the datasets are actually integrated.

---

# ⚙️ Installation & Setup

## Prerequisites

Make sure you have:

* Python 3.10+
* pip
* Git
* Tesseract OCR
* Homebrew — macOS only

---

# 🚀 Quick Start

Follow these commands from the project root.

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Then:

```bash
cd rail_ticket_feature_engineering
```

---

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

---

# 📦 3. Install Python Dependencies

Run:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies manually:

```bash
pip install pandas numpy scikit-learn joblib streamlit pillow opencv-python pytesseract
```

---

# 🔎 4. Install Tesseract OCR

The Python package `pytesseract` requires the Tesseract OCR engine to be installed separately.

## macOS

```bash
brew install tesseract
```

Verify the installation:

```bash
tesseract --version
```

You should see a Tesseract version printed in the terminal.

Verify that Python can access it:

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

---

# 🏋️ 5. Train the Machine Learning Model

From the project root:

```bash
python src/train.py
```

The training process:

```text
Dataset
   ↓
Train/Test Split
   ↓
Feature Engineering
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Random Forest Regression
   ↓
Evaluation
   ↓
Model Serialization
```

The trained model will be saved as:

```text
models/price_model.joblib
```

Metrics will be saved as:

```text
models/metrics.json
```

---

# 🚀 6. Start the Streamlit Application

Run:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

---

# 🧪 How to Use the Application

## Option 1 — Upload a Ticket

1. Start the Streamlit application.
2. Select **Upload from gallery**.
3. Upload a `.png`, `.jpg`, or `.jpeg` ticket.
4. Wait for OCR processing.
5. Review the extracted information.
6. Correct any OCR errors if necessary.
7. Click **Generate Features**.
8. Review the engineered features.
9. Click **Predict Ticket Price**.

---

## Option 2 — Use Camera

1. Start the Streamlit application.
2. Select **Take a photo**.
3. Allow browser camera access.
4. Capture the ticket.
5. Wait for OCR processing.
6. Review the extracted fields.
7. Generate features.
8. Run the prediction.

---

# 🎫 Demo Ticket

A demo ticket is included in the project for testing.

```text
demo/demo_ticket.png
```

Use it to verify that:

```text
Image
  ↓
OCR
  ↓
Ticket Extraction
  ↓
Feature Engineering
  ↓
Prediction
```

is working correctly.

---

# 🧪 Useful Commands

## Check Python version

```bash
python --version
```

## Check pip

```bash
pip --version
```

## Check installed packages

```bash
pip list
```

## Check Streamlit

```bash
streamlit --version
```

## Check Tesseract

```bash
tesseract --version
```

## Check the project Python files

```bash
find . -maxdepth 3 -type f \( -name "*.py" -o -name "*.pyw" \)
```

Expected:

```text
./app/streamlit_app.py
./src/feature_engineering.py
./src/ocr.py
./src/train.py
```

## Retrain the model

If the dataset or feature-engineering code changes:

```bash
rm -f models/price_model.joblib
rm -f models/metrics.json
python src/train.py
```

## Start the application again

```bash
streamlit run app/streamlit_app.py
```

## Stop Streamlit

Press:

```text
Ctrl + C
```

---

# 🛠️ Troubleshooting

## `streamlit: command not found`

Make sure the virtual environment is activated:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install streamlit
```

---

## `ModuleNotFoundError`

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install pandas numpy scikit-learn joblib streamlit pillow opencv-python pytesseract
```

---

## `tesseract: command not found`

Install Tesseract on macOS:

```bash
brew install tesseract
```

Then verify:

```bash
tesseract --version
```

---

## Model not found

Run:

```bash
python src/train.py
```

Then start the app:

```bash
streamlit run app/streamlit_app.py
```

---

## OCR is not extracting information

First check Tesseract:

```bash
tesseract --version
```

Then check Python:

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

Use a clear, well-lit ticket image with readable text.

The Streamlit application also provides a **Raw OCR Text** section that can be used to inspect what the OCR engine detected.

---

# 🛡️ ML Pipeline Design

RailLens uses a Scikit-learn `Pipeline` and `ColumnTransformer`.

This ensures that preprocessing remains consistent between:

```text
Training
   ↓
Testing
   ↓
Inference
```

### Numerical Pipeline

```text
Numerical Features
        ↓
Median Imputation
        ↓
Missing Indicators
        ↓
RobustScaler
```

### Categorical Pipeline

```text
Categorical Features
        ↓
Most-Frequent Imputation
        ↓
Ordinal Encoding
        ↓
Unknown Category Handling
```

Both pipelines feed into:

```text
Random Forest Regressor
```

---

# 📈 Future Roadmap

## Phase 1 — Current Prototype

* [x] Ticket image upload
* [x] Camera input
* [x] OCR
* [x] Ticket field extraction
* [x] Feature engineering
* [x] Machine-learning pipeline
* [x] Fare prediction
* [x] Streamlit interface

## Phase 2 — Journey Intelligence

Integrate railway station and train datasets to derive:

* [ ] Distance
* [ ] Route information
* [ ] Journey duration
* [ ] Average speed
* [ ] Number of stops
* [ ] Origin/destination zones
* [ ] State-level route information

## Phase 3 — Personal Travel Analytics

Build a personal travel dashboard:

```text
Total journeys
Total distance travelled
Total spending
Average fare/km
Most frequent routes
Longest journey
Monthly travel spending
```

## Phase 4 — AI Travel Assistant

Allow users to ask questions such as:

> "How much have I spent on railway travel this year?"

> "What is my most frequent route?"

> "What was my longest journey?"

> "How much did I spend travelling to Delhi?"

## Phase 5 — B2B Ticket Intelligence API

Potentially expose:

```text
Ticket Image / PDF
        ↓
OCR
        ↓
Structured Travel Data
```

for integration with:

* Corporate travel management
* Expense management
* Travel analytics
* Automated ticket processing

---

# 💼 Business Opportunity

RailLens can eventually be positioned as a **travel document intelligence platform**, rather than only a fare-prediction application.

### Consumer

Users could scan tickets and maintain a personal travel history.

### Corporate

Organizations could potentially automate railway-ticket information extraction for expense processing.

### Travel Platforms

A ticket-processing API could potentially be integrated into travel-management workflows.

### Long-Term Vision

> **Convert travel documents into structured, actionable intelligence.**

---

# 🔐 Privacy Considerations

Railway tickets can contain sensitive personal information.

A production version should consider:

* Secure image processing
* Data minimization
* Encryption
* Automatic deletion of uploaded images
* Avoiding unnecessary storage of passenger information
* Clear user consent
* Appropriate privacy policies

This project is a prototype and should not be considered a production-grade ticket-data security system.

---

# 🧰 Tech Stack

| Layer               | Technology                 |
| ------------------- | -------------------------- |
| Frontend            | Streamlit                  |
| Language            | Python                     |
| OCR                 | Tesseract / pytesseract    |
| Data Processing     | Pandas                     |
| Numerical Computing | NumPy                      |
| Machine Learning    | Scikit-learn               |
| ML Model            | Random Forest Regression   |
| Model Serialization | Joblib                     |
| Image Processing    | Pillow / OpenCV            |
| Environment         | Python Virtual Environment |

---

# 🎓 Feature Engineering Concepts Demonstrated

The project connects the application to the Feature Engineering workflow:

```text
Data Ingestion
      ↓
Data Cleaning
      ↓
Train/Test Split
      ↓
Missing Value Handling
      ↓
Feature Creation
      ↓
Categorical Encoding
      ↓
Feature Scaling
      ↓
ML Pipeline
      ↓
Model Training
      ↓
Evaluation
      ↓
Inference
```

The project demonstrates how:

> **Raw Data → Feature Engineering → Machine Learning → Product**

---

# 📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Understanding raw vs engineered features
* Data preprocessing
* Missing-value imputation
* Numerical scaling
* Categorical encoding
* Feature creation
* Temporal feature engineering
* Train/test splitting
* Scikit-learn pipelines
* ColumnTransformer
* Random Forest regression
* Model evaluation
* Model serialization
* OCR-based data extraction
* Deployment through Streamlit

---

# 🚧 Current Limitations

The current prototype has several limitations:

1. OCR accuracy depends on ticket image quality.
2. Ticket formats may vary significantly.
3. The parser currently expects specific field patterns.
4. The fare prediction model depends on the training dataset.
5. Railway network enrichment is planned but not yet fully integrated into the current fare model.
6. The current prototype is intended for demonstration and educational purposes.

---

# 🌱 Future Improvements

Potential improvements include:

* Better OCR preprocessing
* Support for multiple ticket formats
* Layout-aware OCR
* Automatic ticket-format detection
* Railway station database integration
* Train-route enrichment
* Distance calculation
* Historical fare analysis
* Personal travel dashboard
* Model comparison
* Explainable AI
* SHAP-based prediction explanations
* Docker deployment
* REST API
* Cloud deployment
* Model monitoring
* Data drift detection
* Automated model retraining

---

# 👩‍💻 Author

**Anusri Karmokar**

B.Tech Computer Science & Engineering
ITM Skills University

---

# ⭐ Project Summary

RailLens demonstrates how a seemingly simple railway ticket can become a complete machine-learning pipeline.

```text
          📷
      Ticket Image
           │
           ▼
          OCR
           │
           ▼
    Structured Data
           │
           ▼
  Feature Engineering
           │
           ▼
     ML Pipeline
           │
           ▼
    Fare Prediction
           │
           ▼
  Travel Intelligence
```

> **A ticket is not just a receipt. It's data waiting to be understood.**

---

## ⭐ If you found this project interesting

Feel free to fork the repository, experiment with the feature-engineering pipeline, and extend the application with additional railway and travel intelligence features.

````

### One important GitHub detail

Before pushing, replace this:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
````

with your actual GitHub repository URL.

And the **exact commands someone needs to run your current project** are:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rail_ticket_feature_engineering

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

brew install tesseract

python src/train.py

streamlit run app/streamlit_app.py
```

For Windows, the activation command is:

```powershell
.venv\Scripts\activate
```