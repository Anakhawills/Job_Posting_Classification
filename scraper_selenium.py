from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

def scrape_jobs_karkidi():
    path = r"D:\Predictive\Supervised\chromedriver.exe"  # Make sure this is correct
    service = Service(executable_path=path)
    
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=service, options=options)

    url = "https://www.karkidi.com/job-opportunities"
    driver.get(url)
    time.sleep(10)  # Let the page load completely

    job_elements = driver.find_elements(By.CSS_SELECTOR, "div.ads-details.bg-white.border2")
    print(f"✅ Found {len(job_elements)} jobs")

    jobs = []

    for job in job_elements:
        try:
            title = job.find_element(By.TAG_NAME, "h4").text.strip()
            company_links = job.find_elements(By.CSS_SELECTOR, ".cmp-info a")
            company = company_links[1].text.strip() if len(company_links) > 1 else "N/A"
            try:
                location = job.find_elements(By.CSS_SELECTOR, ".cmp-info p")[0].text.strip()
                experience = job.find_element(By.CSS_SELECTOR, "p.emp-exp").text.strip()
            except:
                location = "N/A"
                experience = "N/A"

            jobs.append({
                "title": title,
                "company": company,
                "location": location,
                "experience": experience
            })
        except Exception as e:
            print("⚠️ Error reading a job card:", e)

    driver.quit()

    df = pd.DataFrame(jobs)
    df.to_csv("job_data.csv", index=False, mode='w')

    print("✅ job_data.csv written successfully.")

if __name__ == "__main__":
    scrape_jobs_karkidi()
