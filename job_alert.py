import joblib
from scraper import scrape_jobs
from preprocess import preprocess_skills

def alert_user(preferred_category=1):
    df, X, _ = preprocess_skills("job_data.csv")
    kmeans = joblib.load("kmeans_model.joblib")
    df["category"] = kmeans.predict(X)

    matched_jobs = df[df["category"] == preferred_category]
    print("Matching Jobs:")
    print(matched_jobs[["title", "company", "skills"]])
