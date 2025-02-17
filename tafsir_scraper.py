import requests
import json
import time

# Base URL for Tafsir API
BASE_URL = "https://quran.com/api/proxy/content/api/qdc/tafsirs/en-tafisr-ibn-kathir/by_ayah/{surah}:{ayah}?locale=en&words=true&word_fields=verse_key%2Cverse_id%2Cpage_number%2Clocation%2Ctext_uthmani%2Ccode_v1%2Cqpc_uthmani_hafs&mushaf=2"

# Total number of Surahs in the Quran
TOTAL_SURAHS = 114

# Storage list for Tafsir data
tafsir_data = []

# Iterate through all Surahs
for surah_id in range(1, TOTAL_SURAHS + 1):
    ayah_number = 1  # Start from Ayah 1 for each Surah
    processed_ayahs = set()  # Keep track of ayahs already covered

    while True:
        response = requests.get(BASE_URL.format(surah=surah_id, ayah=ayah_number))

        if response.status_code == 200:
            data = response.json()
            
            # Check if Tafsir exists for the current Ayah
            tafsir_content = data.get("tafsir", {}).get("text", "")
            verses = data.get("tafsir", {}).get("verses", {})

            if not tafsir_content or not verses:
                print(f"No more Tafsir found for Surah {surah_id}, stopping at Ayah {ayah_number}")
                break  # Stop fetching Tafsir for this Surah if no more data

            # Get the list of Ayahs that share the same Tafsir
            ayah_keys = list(verses.keys())
            first_ayah = int(ayah_keys[0].split(":")[1])
            last_ayah = int(ayah_keys[-1].split(":")[1])  # The last Ayah covered in this Tafsir

            # Check if we are stuck on the same Tafsir range repeatedly
            if first_ayah in processed_ayahs:
                print(f"Skipping redundant Tafsir for Surah {surah_id}, Ayah {first_ayah}-{last_ayah}")
                ayah_number += 1  # Move forward manually to prevent infinite loop
                continue

            # Store Tafsir only once for the entire range of Ayahs
            tafsir_data.append({
                "id": f"tafsir_{surah_id}_{first_ayah}_to_{last_ayah}" if first_ayah != last_ayah else f"tafsir_{surah_id}_{first_ayah}",
                "surah_id": surah_id,
                "ayah_range": f"{first_ayah}-{last_ayah}" if first_ayah != last_ayah else str(first_ayah),
                "tafsir_text": tafsir_content,
                "reference": f"Surah {surah_id}, Ayah {first_ayah}-{last_ayah}" if first_ayah != last_ayah else f"Surah {surah_id}, Ayah {first_ayah}",
                "source": "Tafsir Ibn Kathir"
            })

            print(f"Fetched Tafsir for Surah {surah_id}, Ayah {first_ayah}-{last_ayah}")

            # Add covered ayahs to processed set
            for i in range(first_ayah, last_ayah + 1):
                processed_ayahs.add(i)

            # Move to the next Ayah after the last Ayah in this chunk
            ayah_number = last_ayah + 1

        else:
            print(f"Failed to fetch Tafsir for Surah {surah_id}, Ayah {ayah_number}")
            break  # Stop if API fails for an Ayah

        time.sleep(1)  # Small delay to avoid hitting rate limits

    # **Save JSON file after each Surah**
    with open("tafsir_data.json", "w", encoding="utf-8") as f:
        json.dump(tafsir_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved Tafsir data after Surah {surah_id}")

print("🎉 Fetching completed. Data saved to tafsir_data.json")
