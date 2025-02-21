import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm  # Progress bar


# Hadith website URL (Replace with a valid Hadith website)
base_url = "https://sunnah.com/"  # Example URL

# Use Google Chrome User-Agent
headers = {
    "User-Agent": "Chrome/121.0.0.0"
}

all_hadiths = []
hadiths_reference = ["muslim", "nasai","abudawud","tirmidhi","ibnmajah","malik"]
total_hadiths_books = [56, 51, 43, 49, 37, 61]

for i,ref in enumerate(hadiths_reference):
    url = f"{base_url}{ref}/"
    print(url)
    for book_number in tqdm(range(1, total_hadiths_books[i]+1), desc=f"Scraping {ref} Books"):
        book_url = f"{url}{book_number}"
        response = requests.get(book_url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            hadith_entries = soup.find_all("div", class_="hadithTextContainers")

            if hadith_entries:
                for entry in hadith_entries:
                    reference = entry.find_previous("div", class_="hadith_reference_sticky")
                    narrator = entry.find("div", class_="hadith_narrated")
                    english_text = entry.find("div", class_="text_details")
                    arabic_text = entry.find("span", class_="arabic_text_details")

                    hadith_data = {
                        "Book Number": book_number,
                        "Reference": reference.text.strip() if reference else "N/A",
                        "Narrator": narrator.text.strip() if narrator else "N/A",
                        "English Text": english_text.text.strip() if english_text else "N/A",
                        "Arabic Text": arabic_text.text.strip() if arabic_text else "N/A"
                    }
                    all_hadiths.append(hadith_data)
            else:
                print(f"⚠ No Hadiths found for {ref} Book {book_number}")

        else:
            print(f"Failed to fetch Book {book_number}. HTTP Status: {response.status_code}")
        
# Save all Hadiths to JSON
with open("hadith.json", "w", encoding="utf-8") as json_file:
    json.dump(all_hadiths, json_file, ensure_ascii=False, indent=4)

print("✅ Scraping completed. Data saved to sahih_bukhari_hadith.json")
