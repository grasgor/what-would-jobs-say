import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm  

# URL of the interview page
# url = "https://allaboutstevejobs.com/verbatim/interviews/playboy_1985"
# url = "https://allaboutstevejobs.com/verbatim/interviews/newsweek_2006"
url = "https://allaboutstevejobs.com/verbatim/interviews/fortune_2008"

# Fetch the page content
response = requests.get(url)
response.raise_for_status()  # Raise error if request failed

# Parse the HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Extract questions and answers
questions = soup.find_all('p', class_='question')
answers = soup.find_all('p', class_='answer')

# Build the structured data with tqdm progress bar
interview_data = []
for q, a in tqdm(zip(questions, answers), total=len(questions), desc="Processing Q&A"):
    interview_data.append({
        "question": q.get_text(strip=True),
        "response": a.get_text(strip=True),
        "interview_name": "Steve Jobs: 2008 Fortune"
    })

# Save to JSON file
with open('2008_fortune.json', 'w', encoding='utf-8') as f:
    json.dump(interview_data, f, ensure_ascii=False, indent=2)

print("Scraping completed. Data saved to '2008_fortune.json'.")
