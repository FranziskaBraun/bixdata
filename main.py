import os
import re
import pandas as pd
import jiwer
from num2words import num2words
from utils import get_german_hesitations
from bixdata.dataloading import load_meta_data

# ---------- Konfiguration ----------
gold_dir = "bix-txt"      # Ordner mit Gold-Transkripten (.txt)
asr_dir = "bix-whisper"        # Ordner mit ASR-Outputs (.txt)
output_csv = "wer_results.csv"
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

        if ref and hyp:
            # normal case
            m = jiwer.compute_measures([ref], [hyp])

        elif not ref and not hyp:
            # both empty → perfect match
            m = {
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
            }

        elif ref and not hyp:
            # reference non-empty, hypothesis empty → everything deleted
            ref_words = ref.split()
            m = {
                "wer": 1.0,
                "substitutions": 0,
                "deletions": len(ref_words),
                "insertions": 0,
                "hits": 0,
            }

        elif not ref and hyp:
            # reference empty, hypothesis non-empty → everything inserted
            hyp_words = hyp.split()
            m = {
                "wer": 1.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": len(hyp_words),
                "hits": 0,
            }

        rows.append({
            "subject": int(fn.split("_")[1]),
            "test": fn.split("_")[2],
            "center": int(fn.split("_")[3]),
            "assessment": fn.split("_")[5],
            "timestamp": fn.split("_")[6],
            "filename": fn,
            "wer": m["wer"],
            "substitutions": m["substitutions"],
            "deletions": m["deletions"],
            "insertions": m["insertions"],
            "hits": m["hits"],
            "reference": ref,
            "hypothesis": hyp,
        })

        print(f"{fn}: WER={m['wer']:.4f}")

    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nErgebnisse gespeichert in {output_csv}")


if __name__ == "__main__":
    main()
    meta, _ = load_meta_data("audio_data_bix/metadata")
    wer = pd.read_csv("wer_results.csv")
    merged = meta.merge(wer, on='subject', how='inner').sort_values(by=['subject']).reset_index(drop=True)
    pd.DataFrame(merged).to_csv("bix_wer.csv", index=False, encoding="utf-8")
    print()
