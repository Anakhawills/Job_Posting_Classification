from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load pre-saved model & vectorizer
model = joblib.load("kmeans_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Load static dataset (pre-scraped)
df = pd.read_csv("job_data.csv")
X = vectorizer.transform(df["title"])
df["category"] = model.predict(X)

# Optional: give names to clusters
CLUSTER_LABELS = {
    0: "AI / Data Science",
    1: "Web / UI Development",
    2: "Testing / QA",
    3: "Management / Business",
    4: "Backend / DevOps"
}

@app.route("/", methods=["GET", "POST"])
def home():
    jobs = []
    selected_cluster = None
    label = ""

    if request.method == "POST":
        selected_cluster = int(request.form["cluster"])
        jobs = df[df["category"] == selected_cluster].to_dict(orient="records")
        label = CLUSTER_LABELS.get(selected_cluster, "Unknown")

    return render_template("index.html", jobs=jobs, selected=selected_cluster, label=label)

if __name__ == "__main__":
    app.run(debug=True)
