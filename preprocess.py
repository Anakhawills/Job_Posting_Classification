import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_titles(data_path="job_data.csv"):
    df = pd.read_csv(data_path)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["title"])  # Using job title for clustering
    return df, X, vectorizer
