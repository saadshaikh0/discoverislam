import requests
import json

# Base URL for Quran API
BASE_URL = "https://quran.com/api/proxy/content/api/qdc/verses/by_chapter/{}"
PARAMS = {
    "words": "true",
    "translation_fields": "resource_name,language_id",
    "per_page": 300,
    "fields": "text_uthmani,chapter_id,hizb_number,text_imlaei_simple",
    "translations": 85,
    "reciter": 7,
    "word_translation_language": "en",
    "page": 1,
    "word_fields": "verse_key,verse_id,page_number,location,text_uthmani,code_v1,qpc_uthmani_hafs",
    "mushaf": 1
}

# List to store all verses
scraped_data = []

# Iterate through all chapters (1 to 114)
for chapter_id in range(1, 115):
    response = requests.get(BASE_URL.format(chapter_id), params=PARAMS)

    if response.status_code == 200:
        data = response.json()

        for verse in data.get("verses", []):
            verse_data = {
                "id": f"quran_{chapter_id}_{verse['verse_number']}",
                "chapter_id": verse["chapter_id"],
                "verse_number": verse["verse_number"],
                "arabic_text": verse["text_uthmani"],
                "translation": verse["translations"][0]["text"] if verse.get("translations") else "",
                "reference": f"Surah {chapter_id}, Ayah {verse['verse_number']}"
            }
            scraped_data.append(verse_data)
    else:
        print(f"Failed to fetch data for chapter {chapter_id}")

# Save to JSON file
with open("quran_data.json", "w", encoding="utf-8") as f:
    json.dump(scraped_data, f, ensure_ascii=False, indent=2)

print("Fetching completed. Data saved to quran_data.json")
