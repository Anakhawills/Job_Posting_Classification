import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_jobs():
    url = "https://www.karkidi.com/job-opportunities"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    jobs = []

    for job_card in soup.select("div.col-md-12.job-container"):  # Check if this matches the job box
        try:
            title = job_card.select_one("h2.job-title").text.strip()
            company = job_card.select_one("h3.company-name").text.strip()
            skills_tag = job_card.select_one("div.skills")

            if skills_tag:
                skills = skills_tag.text.strip()
            else:
                skills = "N/A"

            jobs.append({"title": title, "company": company, "skills": skills})
        except Exception as e:
            print("Error extracting job:", e)
            continue

    print(f"✅ Total jobs scraped: {len(jobs)}")
    df = pd.DataFrame(jobs)
    df.to_csv("job_data.csv", index=False)
    return df

if __name__ == "__main__":
    scrape_jobs()
