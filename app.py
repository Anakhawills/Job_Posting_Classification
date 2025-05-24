import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_titles
from scraper_selenium import scrape_jobs_karkidi  # Ensure this is correctly named

# Define cluster descriptions
cluster_descriptions = {
    0: "Data Science & Machine Learning roles",
    1: "Web Development & Frontend roles",
    2: "DevOps & Infrastructure",
    3: "Product Management & Business Analyst roles",
    4: "Backend Engineering & Systems Programming"
}

st.title("Job Clustering & Alert System (Karkidi.com)")

user_category = st.slider("Select your job interest cluster (0-4)", 0, 4)

# Show a dynamic description below the slider
st.markdown(f"**Selected Cluster {user_category}:** {cluster_descriptions[user_category]}")

if st.button("Fetch & Match Jobs"):
    scrape_jobs_karkidi()  # Scrape latest jobs
    df, X, _ = preprocess_titles("job_data.csv")
    kmeans = joblib.load("kmeans_model.joblib")
    df["category"] = kmeans.predict(X)

    matched = df[df["category"] == user_category]
    st.success(f"Found {len(matched)} matching jobs in your cluster.")
    st.dataframe(matched[["title", "company", "location", "experience"]])
