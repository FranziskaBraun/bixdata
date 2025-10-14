import os
import re
import pandas as pd
import jiwer
from num2words import num2words
from utils import get_german_hesitations
from bixdata.dataloading import load_meta_data
import chardet

# ---------- Konfiguration ----------
model = "owsm"
gold_dir = "bix-txt"      # Ordner mit Gold-Transkripten (.txt)
asr_dir = f"bix-{model}"        # Ordner mit ASR-Outputs (.txt)
output_csv = f"{model}_wer_results.csv"
encoding = "utf-8"
# ---------- Ende Konfiguration ----------

# Regex für Dresing-&-Pehl Markup
SPEAKER_RE = re.compile(r"^[A-ZÄÖÜß]{1,3}:\s*", re.MULTILINE)  # A:, B:, MA:
TIMESTAMP_RE = re.compile(r"#\d{2}:\d{2}:\d{2}-\d+#")          # #00:00:14-3#
BRACKETS_RE = re.compile(r"\([^)]*\)")                         # alles in Klammern (inkl. Pausen, lachen, (11))
WORDCUT_RE = re.compile(r"\b(\w+)/")                           # mu/ -> mu
PUNCT_RE = re.compile(r"[—–\-\"'“”„,.:;?!()/]+")               # punctuation incl. /
MULTI_SPACE_RE = re.compile(r"\s+")
# hesitation words (German)
# untertitelung des zdf 2020, vielen dank, untertitelung des zdf für funk 2017
# HESITATION_RE = re.compile(r"\b(ähm?|hm+|mhm+|mh+|öh+|hmm+|äh+|oh+|ah+|uh+|aha+|oho+)\b", re.IGNORECASE)
HESITATION_RE = re.compile(r"\b(ä+h+|ä+h+m+|h+m+|m+h+m+|m+h+|ö+h+|h+m+|a+h+|o+h+|u+h+|a+h+a+|o+h+o+)\b", re.IGNORECASE)
FILLER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(get_german_hesitations(), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)
NOISE_RE = re.compile(r"\b(musik|applaus|lachen|geräusch|noise|unk|vielen dank|untertitelung des zdf)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def normalize_text(s: str) -> str:
    t = s
    # remove speaker labels
    t = SPEAKER_RE.sub("", t)
    # remove timestamps
    t = TIMESTAMP_RE.sub("", t)
    # fix truncated words mu/ -> mu
    t = WORDCUT_RE.sub(r"\1", t)
    # remove annotations in brackets
    t = BRACKETS_RE.sub("", t)
    # remove hesitations
    t = HESITATION_RE.sub("", t)
    # remove fillerwords
    t = FILLER_RE.sub("", t)
    # remove punctuation
    t = PUNCT_RE.sub("", t)
    t = NOISE_RE.sub("", t)
    # normalize numbers to words
    def num_to_words(m):
        try:
            return num2words(int(m.group()), lang="de")
        except Exception:
            return m.group()
    t = NUMBER_RE.sub(num_to_words, t)
    t = YEAR_RE.sub("", t)
    # lowercase + normalize spaces
    t = t.lower()
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def read_file(path):
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def main():
    gold_files = sorted([f for f in os.listdir(gold_dir) if f.endswith(".txt")])
    asr_files = sorted([f for f in os.listdir(asr_dir) if f.endswith(".txt")])
    common = [f for f in gold_files if f in asr_files]

    rows = []
    for fn in common:
        ref = normalize_text(read_file(os.path.join(gold_dir, fn))).strip()
        hyp = normalize_text(read_file(os.path.join(asr_dir, fn))).strip()
        print(f"ref={ref}")
        print(f"hyp={hyp}")

        # skip empty files
        if not ref:
            continue

        m = jiwer.compute_measures([ref], [hyp])

        rows.append({
            "subject": int(fn.split("_")[1]),
            "test": fn.split("_")[2],
            "tester": int(fn.split("_")[3]),
            "assessment": fn.split("_")[5],
            "timestamp": fn.split("_")[6],
            "filename": fn,
            f"{model}_wer": m["wer"],
            f"{model}_substitutions": m["substitutions"],
            f"{model}_deletions": m["deletions"],
            f"{model}_insertions": m["insertions"],
            f"{model}_hits": m["hits"],
            f"{model}_reference": ref,
            f"{model}_hypothesis": hyp,
        })

        print(f"{fn}: WER={m['wer']:.4f}")

    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nErgebnisse gespeichert in {output_csv}")


def to_utf8(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, "rb") as f:
                raw_data = f.read()

            detected = chardet.detect(raw_data)
            enc = detected["encoding"]
            print(f"Converting {filename} from {enc} to UTF-8")

            try:
                text = raw_data.decode(enc)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                print(f"Skipping {filename}: {e}")


if __name__ == "__main__":
    # to_utf8("audio_data_bix/metadata3")
    # main()
    meta, asses = load_meta_data("audio_data_bix/metadata3")
    meta = meta.sort_values(by=['subject']).reset_index(drop=True)
    result_picture = pd.read_csv("results/bix_llm_picture_description_Mistral-Small-3.2-24B-Instruct-2506.csv")
    # result_story = pd.read_csv("results/bix_llm_story_reading_Mistral-Small-3.2-24B-Instruct-2506.csv")
    merged = meta.merge(result_picture, on='subject', how='inner')
    # merged = merged.merge(result_story, on='subject', how='inner')
    merged = merged.sort_values(by=['subject']).reset_index(drop=True)

    # wer_whisper = pd.read_csv("whisper_wer_results.csv")
    # wer_owsm = pd.read_csv("owsm_wer_results.csv")
    # wer_parakeet = pd.read_csv("parakeet_wer_results.csv")
    # merged = meta.merge(wer_whisper, on='subject', how='inner')
    # merged = merged.merge(wer_owsm, on='filename', how='inner')
    # merged = merged.merge(wer_parakeet, on='filename', how='inner')
    # merged = merged.sort_values(by=['subject']).reset_index(drop=True)
    # pd.DataFrame(merged).to_csv("bix_wer.csv", index=False, encoding="utf-8")
    print("Finished")
