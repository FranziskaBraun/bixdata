import os
import re
import pandas as pd
import jiwer
# from nltk.corpus import stopwords
from utils import german_word_to_number, get_german_hesitations

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
# Interpunktion + Schrägstriche
PUNCT_RE = re.compile(r"[—–\-\"'“”„,.:;?!()/]+")               # inkl. / und //
MULTI_SPACE_RE = re.compile(r"\s+")


def preprocess_text(text):
    """
    Preprocess the text by substituting German words with numbers and synonyms,
    removing stopwords, and lowercasing the text.

    Parameters:
    text (str): The input text to preprocess.
    symbol_synonyms (dict): The dictionary of symbol synonyms for preprocessing.
    german_stopwords (set): Set of German stopwords.

    Returns:
    str: The preprocessed text.
    """
    if not isinstance(text, str):
        return ""  # Return an empty string for non-string input

    # Tokenize the text
    tokens = text.split()
    processed_tokens = []

    hesitations = get_german_hesitations()
    # german_stopwords = set(stopwords.words('german'))
    for token in tokens:
        # Skip German stopwords
        # if token in german_stopwords or token in hesitations:
        if token in hesitations:
            continue

        # Substitute numbers
        number = german_word_to_number(token)
        if isinstance(number, int):  # If it's a number
            processed_tokens.append(str(number))

    return " ".join(processed_tokens)


def normalize_text(s: str) -> str:
    t = s
    # Sprecherlabels entfernen
    t = SPEAKER_RE.sub(" ", t)
    # Zeitmarken entfernen
    t = TIMESTAMP_RE.sub(" ", t)
    # Abgebrochene Wörter "mu/" -> "mu"
    t = WORDCUT_RE.sub(r"\1", t)
    # Annotationen & Pausen in Klammern entfernen
    t = BRACKETS_RE.sub(" ", t)
    # Interpunktion + // entfernen
    t = PUNCT_RE.sub(" ", t)
    # Kleinschreibung + Whitespaces normalisieren
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
        ref = preprocess_text(normalize_text(read_file(os.path.join(gold_dir, fn))))
        hyp = preprocess_text(normalize_text(read_file(os.path.join(asr_dir, fn))))

        m = jiwer.compute_measures([ref], [hyp])

        rows.append({
            "filename": fn,
            "reference": ref,
            "hypothesis": hyp,
            "wer": m["wer"],
            "substitutions": m["substitutions"],
            "deletions": m["deletions"],
            "insertions": m["insertions"],
            "hits": m["hits"],
        })

        print(f"{fn}: WER={m['wer']:.4f}")

    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nErgebnisse gespeichert in {output_csv}")


if __name__ == "__main__":
    main()
