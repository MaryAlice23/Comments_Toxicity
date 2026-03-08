'''
import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="AI Toxicity Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ------------------------------------------------
# Load Model
# ------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("models/toxicity_model.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()
MAX_LEN = 100

# ------------------------------------------------
# Text Cleaning
# ------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ------------------------------------------------
# Prediction Function
# ------------------------------------------------
def predict_text_list(text_list):
    cleaned = [clean_text(text) for text in text_list]
    seq = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    predictions = model.predict(padded)
    return predictions

# ------------------------------------------------
# UI Header
# ------------------------------------------------
st.title("🛡️ AI-Powered Comment Toxicity Detection System")
st.markdown("### Deep Learning based NLP Moderation Dashboard")

# ------------------------------------------------
# Single Comment Prediction
# ------------------------------------------------
st.subheader("🔍 Single Comment Analysis")

comment = st.text_area("Enter a comment")

if st.button("Analyze Comment"):
    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        score = predict_text_list([comment])[0][0]

        col1, col2 = st.columns(2)

        with col1:
            if score > 0.5:
                st.error("⚠️ Toxic Comment Detected")
            else:
                st.success("✅ Non-Toxic Comment")

        with col2:
            st.metric("Toxicity Score", f"{score:.2f}")

st.markdown("---")

# ------------------------------------------------
# Bulk CSV Prediction
# ------------------------------------------------
st.subheader("📂 Bulk CSV Analysis")

uploaded_file = st.file_uploader("Upload any CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    text_column = st.selectbox(
        "Select column containing comments:",
        df.columns
    )

    if st.button("Run Bulk Prediction"):
        predictions = predict_text_list(df[text_column])

        df["toxicity_score"] = predictions
        df["prediction"] = (df["toxicity_score"] > 0.5).astype(int)

        st.success("Prediction Completed ✅")

        # Metrics
        toxic_count = df["prediction"].sum()
        non_toxic_count = len(df) - toxic_count

        col1, col2 = st.columns(2)
        col1.metric("Toxic Comments", toxic_count)
        col2.metric("Non-Toxic Comments", non_toxic_count)

        # Pie Chart
        st.subheader("📊 Toxic vs Non-Toxic Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(
            [toxic_count, non_toxic_count],
            labels=["Toxic", "Non-Toxic"],
            autopct="%1.1f%%"
        )
        st.pyplot(fig1)

        # Confidence Histogram
        st.subheader("📈 Toxicity Score Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["toxicity_score"], bins=20)
        ax2.set_xlabel("Toxicity Score")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

        st.dataframe(df.head())

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="toxicity_predictions.csv",
            mime="text/csv"
        )

 '''
import streamlit as st
import pandas as pd
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Load Model and Tokenizer
# ----------------------------
model = load_model("models/toxicity_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# ----------------------------
# Prediction Function
# ----------------------------
def predict_comment(comment):

    comment = clean_text(comment)

    seq = tokenizer.texts_to_sequences([comment])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(pad)[0]

    return pred


# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(
    page_title="AI Comment Toxicity Detector",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Comment Toxicity Detection")

st.write(
    "This AI system detects toxic comments such as insults, threats, hate speech and obscene language."
)

# ============================
# Single Comment Prediction
# ============================

st.header("Analyze Single Comment")

comment = st.text_area("Enter Comment")

if st.button("Analyze"):

    if comment.strip() == "":
        st.warning("Please enter a comment")

    else:

        pred = predict_comment(comment)

        # Case 1: Binary model
        if len(pred) == 1:

            probability = float(pred[0])

            st.subheader("Prediction Result")

            if probability > 0.5:
                st.error(f"Toxic Comment Detected ({probability:.2f})")
            else:
                st.success(f"Safe Comment ({1-probability:.2f})")

            st.progress(probability)

        # Case 2: Multi-label model
        else:

            pred = pred[:len(labels)]

            result = pd.DataFrame({
                "Toxicity Type": labels,
                "Probability": pred
            })

            st.subheader("Toxicity Probabilities")

            st.bar_chart(result.set_index("Toxicity Type"))

# ============================
# Bulk CSV Analysis
# ============================

st.header("Bulk Comment Analysis")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Automatically detect text columns
    text_columns = df.select_dtypes(include=["object"]).columns

    if len(text_columns) == 0:
        st.error("No text column found in dataset")

    else:

        selected_column = st.selectbox(
            "Select Comment Column",
            text_columns
        )

        if st.button("Analyze Dataset"):

            predictions = []

            for text in df[selected_column]:

                pred = predict_comment(text)

                if len(pred) == 1:
                    predictions.append(float(pred[0]))
                else:
                    predictions.append(max(pred))

            df["toxicity_score"] = predictions

            st.success("Analysis Completed")

            st.dataframe(df.head())

            csv = df.to_csv(index=False)

            st.download_button(
                "Download Results",
                csv,
                "toxicity_results.csv",
                "text/csv"
            )