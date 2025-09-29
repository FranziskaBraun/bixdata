"""
compute_wer_dresing.py
Berechnet WER zwischen Gold-Transkripten (Dresing&Pehl) und ASR-Outputs.
Optionen:
  - keep_annotations: wenn True, wandeln wir bestimmte Annotationen in Token (z.B. <laughter>),
    sonst werden alle Annotationen entfernt (empfohlen: False für Clean-WER).
"""

import os
import re
from jiwer import wer
from jiwer.measures import compute_measures


# ---------- Konfiguration ----------
gold_dir = "gold_txt"      # txt-Dateien deiner manuellen Transkripte (eine Datei = ein Interview)
asr_dir = "asr_txt"       # txt-Dateien der ASR-Modelle (Dateinamen müssen matchen)
encoding = "utf-8"
# keep_annotations: False = entferne (lacht),(unv.), Zeitmarken; True = ersetze z.B. (lacht) -> <laughter>
keep_annotations = False
# ---------- Ende Konfiguration ----------

# Regex-Patterns (Dresing & Pehl typische Markups)
BRACKET_ANNOTATION_RE = re.compile(r"\((?:unv\.|unverständlich|lacht|hustet|kichert|schluckt|atmung| Geräusch|[^\)]+?\?)\)", flags=re.I)
GENERAL_BRACKETS_RE = re.compile(r"\([^)]*\)")   # alle runden Klammern
TIMESTAMP_RE = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?")  # [00:01:23] oder 00:01
ELLIPSIS_RE = re.compile(r"\.{2,}")              # … oder ...
PUNCT_RE = re.compile(r"[—–\-\"'“”„,.:;?!()]")   # Interpunktion (Klammern bereits entfernt)
MULTI_SPACE_RE = re.compile(r"\s+")

# Optionale mapping von Annotationen zu Tokens (wenn keep_annotations True)
ANNOTATION_MAP = {
    "lacht": "<laughter>",
    "hustet": "<cough>",
    "schluckt": "<swallow>",
    "atmung": "<breath>",
    "unv.": "<unv>",
}


def normalize_text(s: str, keep_annotations: bool = False) -> str:
    """
    Normalisiert ein Transkript gemäß Empfehlungen:
      - optional: (lacht), (unv.) -> <laughter>/<unv> (wenn keep_annotations True)
      - sonst: entferne runde Klammern, Zeitmarken, Ellipsen, Interpunktion
      - lowercase, strip, collapse spaces
    """
    t = s.strip()

    # 1) Zeitmarken entfernen
    t = TIMESTAMP_RE.sub(" ", t)

    if keep_annotations:
        # Ersetze bekannte Annotationen in Tokens
        def _map_ann(m):
            text = m.group(0).strip("()").lower()
            # einfache match/lookup
            text_key = text.replace(" ", "")
            mapped = ANNOTATION_MAP.get(text_key)
            if mapped:
                return " " + mapped + " "
            # wenn unbekannt: entferne oder bleibe? --> entferne
            return " "
        t = BRACKET_ANNOTATION_RE.sub(_map_ann, t)
        # entferne sonstige (unbekannte) Klammerinhalte
        t = GENERAL_BRACKETS_RE.sub(" ", t)
    else:
        # Entferne *alle* runden Klammern und Inhalte darin (Dresing z.B. (unv.), (lacht))
        t = GENERAL_BRACKETS_RE.sub(" ", t)

    # 2) Ellipsen -> einzelnes Token (optional) oder entfernen
    t = ELLIPSIS_RE.sub(" ", t)

    # 3) Interpunktion (Komma, Punkt, Fragezeichen, Anführungszeichen etc.) entfernen
    t = PUNCT_RE.sub(" ", t)

    # 4) Zahlen: (OPTION) du könntest Zahlen normalisieren (z.B. "zwölf" statt "12").
    # Für Konsistenz: wir belassen Ziffern so wie sie sind, falls ASR ebenfalls Ziffern ausgibt.
    # (wenn du willst, füge hier eine mapping-Funktion hinzu)

    # 5) Kleinschreibung und whitespace collapse
    t = t.lower()
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def read_text_file(path):
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def list_matching_files(gold_dir, asr_dir):
    gold_files = sorted([f for f in os.listdir(gold_dir) if f.lower().endswith(".txt")])
    asr_files = sorted([f for f in os.listdir(asr_dir) if f.lower().endswith(".txt")])
    # Match by filename - best practice: filenames identical in both dirs
    common = [f for f in gold_files if f in asr_files]
    only_gold = [f for f in gold_files if f not in asr_files]
    only_asr = [f for f in asr_files if f not in gold_files]
    return common, only_gold, only_asr


def main():
    common, only_gold, only_asr = list_matching_files(gold_dir, asr_dir)
    if not common:
        print("Keine passenden Dateipaarungen gefunden. Stelle sicher, dass Dateinamen übereinstimmen.")
        print("Gold-only:", only_gold)
        print("ASR-only: ", only_asr)
        return

    per_file_measures = []
    refs = []
    hyps = []

    for fn in common:
        gold_path = os.path.join(gold_dir, fn)
        asr_path = os.path.join(asr_dir, fn)

        gold_text = read_text_file(gold_path)
        asr_text = read_text_file(asr_path)

        gold_norm = normalize_text(gold_text, keep_annotations=keep_annotations)
        asr_norm = normalize_text(asr_text,  keep_annotations=keep_annotations)

        refs.append(gold_norm)
        hyps.append(asr_norm)

        m = compute_measures([gold_norm], [asr_norm])
        per_file_measures.append((fn, m['wer'], m['substitutions'], m['deletions'], m['insertions']))

        print(f"{fn}  WER={m['wer']:.4f}  S={m['substitutions']} D={m['deletions']} I={m['insertions']}")

    # Gesamte WER (über alle Paare, jiwer kann Liste verarbeiten)
    overall = compute_measures(refs, hyps)
    print("\n=== Gesamt ===")
    print(f"Files compared: {len(common)}")
    print(f"Overall WER = {overall['wer']:.4f}")
    print(f"S={overall['substitutions']} D={overall['deletions']} I={overall['insertions']}")


if __name__ == "__main__":
    main()
