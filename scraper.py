import requests
from bs4 import BeautifulSoup
import json
import re

# Define the list to store scraped data
scraped_data = []

# Function to scrape Quranic verses
def scrape_quran():
    url = "https://quran.com/1"  # Surah Al-Fatiha
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content, "html.parser")
    # The site might use different class names for Arabic and translation text
    # Let's target typical div structures used by Quran.com
    verses = soup.find_all("div", class_=re.compile("TranslationText"))
    arabic_verses = soup.find_all("div", class_=re.compile("AyahText"))
    print(verses)
    print(arabic_verses)
    # Check if both Arabic and translations are fetched
    if len(verses) != len(arabic_verses):
        print("Mismatch between Arabic verses and translations!")
    
    for index, (arabic, translation) in enumerate(zip(arabic_verses, verses), start=1):
        arabic_text = arabic.get_text(strip=True)
        translated_text = translation.get_text(strip=True)
        reference = f"Surah Al-Fatiha, Ayah {index}"

        scraped_data.append({
            "id": f"quran_1_{index}",
            "arabic_text": arabic_text,
            "translation": translated_text,
            "source": "Quran",
            "reference": reference
        })

# Scrape data
scrape_quran()

# Save to JSON file
with open("islamic_data.json", "w", encoding="utf-8") as f:
    json.dump(scraped_data, f, ensure_ascii=False, indent=2)

print("Scraping completed and data saved to islamic_data.json")
