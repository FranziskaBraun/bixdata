import json
import os


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    tenths = int((seconds - int(seconds)) * 10)
    return f"{minutes:02}:{secs:02}:{tenths}"


def get_speaker(speaker_data, time_point):
    time = 0.0
    for segment in speaker_data:
        if time <= time_point < segment['time']:
            return segment['speaker']
        time = segment['time']
    return "U"  # U for Unknown if no match


result_dir = "../llm_data/trans/transcripts_nsc"
speaker_dir = "../llm_data/trans/gt_transcripts_nsc"

if not os.path.exists(result_dir):
    print(f"Error: Directory '{result_dir}' does not exist.")
    exit(1)

if not os.path.exists(result_dir):
    print(f"Error: Directory '{speaker_dir}' does not exist.")
    exit(1)

for filename in os.listdir(result_dir):
    if filename.endswith("Interview.json"):
        json_path = os.path.join(result_dir, filename)
        speaker_filename = f"interview{filename.split('_')[1]}.json"
        speaker_path = os.path.join(speaker_dir, speaker_filename)
        try:
            with open(json_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            with open(speaker_path, "r", encoding="utf-8") as speaker_file:
                speaker_data = json.load(speaker_file)

            text = ""
            end = 0.0
            current_speaker = None

            for result in data['result']:
                for word in result['words_formatted']:
                    start, stop = word['interval']
                    speaker = get_speaker(speaker_data, start)

                    if speaker != current_speaker:
                        if text:
                            text += "\n"
                        text += f"{speaker}: "
                        current_speaker = speaker

                    pause = start - end
                    if pause >= 1:
                        text += f"({round(pause)}) "

                    text += word['word'] + " "
                    end = stop

                timestamp = format_time(end)
                text += f"#{timestamp}#\n"

            text_path = os.path.join(result_dir, filename.replace(".json", ".txt"))
            with open(text_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
