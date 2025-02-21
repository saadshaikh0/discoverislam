import requests
import json
import time

# Base URL for Tafsir API
BASE_URL = "https://quran.com/api/proxy/content/api/qdc/tafsirs/en-tafsir-maarif-ul-quran/by_ayah/{surah}:{ayah}?locale=en&words=true"

# Total number of Surahs in the Quran
TOTAL_SURAHS = 114

# API to get total Ayahs per Surah
CHAPTERS_API = "https://quran.com/_next/data/iXKgDCNrm2Y6VuxobJ_Ab/en/1.json?startingVerse=1"

def get_total_ayahs():
    """Fetch total number of Ayahs for each Surah from API."""
    response = requests.get(CHAPTERS_API)
    if response.status_code != 200:
        print("Failed to fetch Ayah counts. Status Code:", response.status_code)
        return {}
    
    data = response.json()
    chapters_data = data.get("pageProps", {}).get("chaptersData", {})
    return {int(surah_id): chapter["versesCount"] for surah_id, chapter in chapters_data.items()}

# Storage list for Tafsir data
tafsir_data = []

# Fetch total Ayahs for each Surah
total_ayahs_per_surah = get_total_ayahs()
if not total_ayahs_per_surah:
    print("Error fetching Surah Ayah counts. Exiting.")
    exit()

# Iterate through all Surahs
for surah_id in range(1, TOTAL_SURAHS + 1):
    total_ayahs = total_ayahs_per_surah.get(surah_id, 0)
    if total_ayahs <= 0:
        print(f"Skipping Surah {surah_id} due to invalid Ayah count: {total_ayahs}")
        continue
    
    print(f"Fetching Tafsir for Surah {surah_id}, Total Ayahs: {total_ayahs}")
    processed_ayahs = set()  # Keep track of ayahs already covered

    for ayah_number in range(1, total_ayahs + 1):
        response = requests.get(BASE_URL.format(surah=surah_id, ayah=ayah_number))

        if response.status_code == 200:
            data = response.json()
            
            # Check if Tafsir exists for the current Ayah
            tafsir_content = data.get("tafsir", {}).get("text", "")
            verses = data.get("tafsir", {}).get("verses", {})

            if not tafsir_content or not verses:
                print(f"No Tafsir found for Surah {surah_id}, Ayah {ayah_number}")
                continue  # Skip if no Tafsir data exists

            # Get the list of Ayahs that share the same Tafsir
            ayah_keys = list(verses.keys())
            first_ayah = int(ayah_keys[0].split(":")[1])
            last_ayah = int(ayah_keys[-1].split(":")[1])  # The last Ayah covered in this Tafsir

            # Check if we are stuck on the same Tafsir range repeatedly
            if first_ayah in processed_ayahs:
                print(f"Skipping redundant Tafsir for Surah {surah_id}, Ayah {first_ayah}-{last_ayah}")
                continue

            # Store Tafsir only once for the entire range of Ayahs
            tafsir_data.append({
                "id": f"tafsir_{surah_id}_{first_ayah}_to_{last_ayah}" if first_ayah != last_ayah else f"tafsir_{surah_id}_{first_ayah}",
                "surah_id": surah_id,
                "ayah_range": f"{first_ayah}-{last_ayah}" if first_ayah != last_ayah else str(first_ayah),
                "tafsir_text": tafsir_content,
                "reference": f"Surah {surah_id}, Ayah {first_ayah}-{last_ayah}" if first_ayah != last_ayah else f"Surah {surah_id}, Ayah {first_ayah}",
                "source": "Tafsir Maarif-ul-Quran"
            })

            print(f"Fetched Tafsir for Surah {surah_id}, Ayah {first_ayah}-{last_ayah}")

            # Add covered ayahs to processed set
            for i in range(first_ayah, last_ayah + 1):
                processed_ayahs.add(i)

        else:
            print(f"Failed to fetch Tafsir for Surah {surah_id}, Ayah {ayah_number}")

        time.sleep(1)  # Small delay to avoid hitting rate limits

    # **Save JSON file after each Surah**
    with open("tafsir_data_maarif-ul-quran.json", "w", encoding="utf-8") as f:
        json.dump(tafsir_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved Tafsir data after Surah {surah_id}")

print("🎉 Fetching completed. Data saved to tafsir_data.json")
