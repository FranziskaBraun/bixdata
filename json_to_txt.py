import json
import os

# Define the result directory containing JSON files
result_dir = "./audio_data_bix/audio_processed/result"

# Ensure the directory exists
if not os.path.exists(result_dir):
    print(f"Error: Directory '{result_dir}' does not exist.")
    exit(1)

# Iterate over all JSON files in the result directory
for filename in os.listdir(result_dir):
    if filename.endswith(".json"):  # Process only JSON files
        json_path = os.path.join(result_dir, filename)

        try:
            # Read the JSON file
            with open(json_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            # Extract transcript text
            text = ""
            for result in data['result']:
                text += result['transcript'] + " "

            # Define the output TXT file path (same directory as JSON)
            txt_path = os.path.splitext(json_path)[0] + ".txt"

            # Save the transcript as a TXT file
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)

            print(f"Transcript saved to: {txt_path}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {filename}: {e}")

