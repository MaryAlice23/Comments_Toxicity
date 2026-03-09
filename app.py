import streamlit as st
import pandas as pd
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model("models/Toxicity_model.h5", compile=False)

with open("models/Tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 150

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# =====================================================
# PREDICT COMMENT
# =====================================================

def predict_comment(comment):

    comment = clean_text(comment)

    seq = tokenizer.texts_to_sequences([comment])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(pad)

    return pred[0]

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="AI Toxic Comment Detector",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Toxic Comment Detection System")

st.write(
    "Detect toxic comments including insults, threats, obscene language and hate speech using Deep Learning."
)

# =====================================================
# SINGLE COMMENT
# =====================================================

st.header("Analyze Single Comment")

comment = st.text_area("Enter Comment")

if st.button("Analyze Comment"):

    if comment.strip() == "":
        st.warning("Please enter a comment")

    else:

        with st.spinner("Analyzing comment..."):

            pred = predict_comment(comment)

        result = pd.DataFrame({
            "Toxicity Type": labels,
            "Probability": pred
        })

        st.subheader("Toxicity Probabilities")

        st.bar_chart(result.set_index("Toxicity Type"))

        if max(pred) > 0.5:
            st.error("⚠️ Toxic Comment Detected")
        else:
            st.success("✅ Comment is Safe")

# =====================================================
# BULK CSV ANALYSIS
# =====================================================

st.header("Bulk Comment Analysis")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    text_columns = df.select_dtypes(include=["object"]).columns

    if len(text_columns) == 0:

        st.error("No text column found in dataset")

    else:

        selected_column = st.selectbox(
            "Select Comment Column",
            text_columns
        )

        if st.button("Analyze Dataset"):

            with st.spinner("Analyzing dataset..."):

                texts = df[selected_column].fillna("").astype(str)

                texts = texts.apply(clean_text)

                seq = tokenizer.texts_to_sequences(texts)

                pad = pad_sequences(seq, maxlen=MAX_LEN)

                preds = model.predict(pad, batch_size=128)

            pred_df = pd.DataFrame(preds, columns=labels)

            df = pd.concat([df, pred_df], axis=1)

            df["toxicity_score"] = pred_df.max(axis=1)

            st.success("Analysis Completed")

            st.dataframe(df.head())

            csv = df.to_csv(index=False)

            st.download_button(
                "Download Results",
                csv,
                "toxicity_predictions.csv",
                "text/csv"
            )

            # =====================================================
            # DASHBOARD
            # =====================================================

            st.header("Toxicity Analytics Dashboard")

            df["label"] = df["toxicity_score"].apply(
                lambda x: "Toxic" if x > 0.5 else "Safe"
            )

            st.subheader("Toxic vs Safe Comments")

            count = df["label"].value_counts()

            st.bar_chart(count)

            st.subheader("Toxicity Score Distribution")

            st.line_chart(df["toxicity_score"])

            st.subheader("Top 10 Most Toxic Comments")

            toxic_comments = df.sort_values(
                by="toxicity_score",
                ascending=False
            ).head(10)

            st.dataframe(toxic_comments)