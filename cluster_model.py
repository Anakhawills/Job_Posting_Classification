from sklearn.cluster import KMeans
import joblib
from preprocess import preprocess_titles

def train_and_save_model():
    df, X, vectorizer = preprocess_titles()
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["category"] = kmeans.fit_predict(X)

    df.to_csv("categorized_jobs.csv", index=False)
    joblib.dump(kmeans, "kmeans_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

    print("✅ Model trained and saved.")
    return df

if __name__ == "__main__":
    train_and_save_model()
