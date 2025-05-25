import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("kmeans_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Load job data
df = pd.read_csv("job_data.csv")
X = vectorizer.transform(df["title"])
df["category"] = model.predict(X)

# Cluster labels
CLUSTER_LABELS = {
    0: "AI / Data Science",
    1: "Web / UI Development",
    2: "Testing / QA",
    3: "Management / Business",
    4: "Backend / DevOps"
}

st.title("Job Cluster Classifier")

selected = st.selectbox("Choose a Job Cluster:", options=list(CLUSTER_LABELS.keys()),
                        format_func=lambda x: CLUSTER_LABELS[x])

matched = df[df["category"] == selected]

st.success(f"Found {len(matched)} jobs in '{CLUSTER_LABELS[selected]}' cluster.")
st.dataframe(matched[["title", "company", "location", "experience"]])
