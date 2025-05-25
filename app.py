from flask import Flask, render_template, request
import pandas as pd
import joblib
from scraper_selenium import scrape_jobs_karkidi
from preprocess import preprocess_titles

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    jobs = []
    selected_cluster = None

    if request.method == "POST":
        selected_cluster = int(request.form["cluster"])
        scrape_jobs_karkidi()
        df, X, _ = preprocess_titles("job_data.csv")
        model = joblib.load("kmeans_model.joblib")
        df["category"] = model.predict(X)
        jobs = df[df["category"] == selected_cluster].to_dict(orient="records")

    return render_template("index.html", jobs=jobs, selected=selected_cluster)

if __name__ == "__main__":
    app.run(debug=True)
