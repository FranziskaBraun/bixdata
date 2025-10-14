import json
from pathlib import Path
import base64
import pandas as pd

from bixdata.dataloading import load_text_file
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, Optional
from tqdm import tqdm
import re
import ast


# Define JSON schema for structured output
class CognitiveAssessment(BaseModel):
    cognitive_status: Literal["HC", "MCI", "DEM"]
    # reasoning: Optional[str] = None  # optional explanation


# Set API connection to local vLLM (update your IP/port)
openai_api_key = "EMPTY"
# openai_api_base_mistral = "http://127.0.0.1:8073/v1"
openai_api_base_mistral_vllm_1 = "http://127.0.0.1:8091/v1"
openai_api_base_mistral_vllm_2 = "http://127.0.0.1:8092/v1"
openai_api_base_deephermes = "http://127.0.0.1:8090/v1"
openai_api_base_magistral = "http://127.0.0.1:8093/v1"
# openai_api_base_mxbai = "http://127.0.0.1:8094/v1"
# openai_api_base_phi4 = "http://127.0.0.1:8099/v1"
# openai_api_base_qwen = "http://127.0.0.1:8090/v1"
# openai_api_base_qwen_omni = "http://127.0.0.1:8090/v1"

# Modify OpenAI's API key and API base to use vLLM's API server.
# Model: https://huggingface.co/neuralmagic/Mistral-Small-24B-Instruct-2501-FP8-Dynamic
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base_mistral_vllm_2,
)

# Optionally print model list
models = client.models.list()
model = models.data[0].id
print(f"Using model: {model}")

# Create JSON schema for output constraint
json_schema = CognitiveAssessment.model_json_schema()


# Sie sind ein Experte für die sprachliche Diagnostik im medizinischen Bereich.
# Ihre Aufgabe ist es, anhand transkribierter Gespräche mögliche Anzeichen von Demenz und Depression zu erkennen.
# Build prompt for classification
# den Text in Bezug auf Kohärenz (text_coherence), lexikalische Vielfalt (lexical_diversity), Satzlänge (sentence_length) und Wortfindungsschwierigkeiten (word_finding_difficulties) mit Hilfe von Punkten zwischen 0 und 1 zu bewerten.
def get_user_prompt_picture_recall(picture_description, delayed_recall):
    prompt = f"""
    Die Person beschreibt das gegebene Bild (Bildbeschreibung).
    Nach Ablenkung muss die Person das Bild aus dem Gedächtnis beschreiben (Verzögerter Recall).
    Das Bild kann mit acht Konzepten beschrieben werden:
    - Eine Berglandschaft
    - Wanderer suchen den Weg
    - Eine Brücke führt über den Fluss
    - Kinder sitzen auf einem Baumstamm
    - Ein Junge ruft seine Eltern
    - Ein Mädchen fällt ins Wasser
    - Gewitterwolken mit Blitzen
    - Ein Haus in der Ferne
    
    Ihre Aufgabe darin, anhand der Bildbeschreibung und des verzögerten Recalls den kognitiven Status der Person in eine der drei Kategorien zu klassifizieren:
    - HC = Healthy Control
    - MCI = Mild Cognitive Impairment
    - DEM = Mild to Severe Dementia

    Hier ist die Bildbeschreibung:
    \"\"\"{picture_description}\"\"\"
    
    Hier ist der verzögerte Recall:
    \"\"\"{delayed_recall}\"\"\"
    
    Geben Sie die Bewertung NUR im JSON-Format entsprechend des gegebenen Schemas an. Keine anderen Ausgaben.
    """
    return prompt


def get_user_prompt_picture(picture_description):
    prompt = f"""
    Die Person beschreibt das gegebene Bild (Bildbeschreibung).
    Das Bild kann mit acht Konzepten beschrieben werden:
    - Eine Berglandschaft
    - Wanderer suchen den Weg
    - Eine Brücke führt über den Fluss
    - Kinder sitzen auf einem Baumstamm
    - Ein Junge ruft seine Eltern
    - Ein Mädchen fällt ins Wasser
    - Gewitterwolken mit Blitzen
    - Ein Haus in der Ferne

    Ihre Aufgabe darin, anhand der Bildbeschreibung den kognitiven Status der Person in eine der drei Kategorien zu klassifizieren:
    - HC = Healthy Control
    - MCI = Mild Cognitive Impairment
    - DEM = Mild to Severe Dementia

    Hier ist die Bildbeschreibung:
    \"\"\"{picture_description}\"\"\"

    Geben Sie die Bewertung NUR im JSON-Format entsprechend des gegebenen Schemas an. Keine anderen Ausgaben.
    """
    return prompt


def get_user_prompt_story_recall(story_reading, delayed_recall):
    prompt = f"""
    Die Person liest eine Geschichte vor (Vorlesung).
    Nach Ablenkung erzählt die Person die Geschichte aus dem Gedächtnis mit eigenen Worten nach (Verzögerter Recall).
    
    Ihre Aufgabe besteht darin, anhand der Vorlesung und des verzögerten Recalls den kognitiven Status der Person in eine der drei Kategorien zu klassifizieren:
    - NCI = No/age-related Cognitive Impairment
    - MCI = Mild Cognitive Impairment
    - DEM = Mild to Severe Dementia

    Hier ist die Vorlesung:
    \"\"\"{story_reading}\"\"\"

    Hier ist der verzögerte Recall:
    \"\"\"{delayed_recall}\"\"\"

    Geben Sie die Bewertung NUR im JSON-Format entsprechend des gegebenen Schemas an. Keine anderen Ausgaben.
    """
    return prompt


def get_user_prompt_story(story_reading):
    prompt = f"""
    Die Person liest eine Geschichte vor (Vorlesung).
    
    Ihre Aufgabe besteht darin, anhand der Vorlesung den kognitiven Status der Person in eine der drei Kategorien zu klassifizieren:
    - NCI = No/age-related Cognitive Impairment
    - MCI = Mild Cognitive Impairment
    - DEM = Mild to Severe Dementia

    Hier ist die Vorlesung:
    \"\"\"{story_reading}\"\"\"

    Geben Sie die Bewertung NUR im JSON-Format entsprechend des gegebenen Schemas an. Keine anderen Ausgaben.
    """
    return prompt


def get_response(messages):
    # Send request
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # top_p=0.95,
        # temperature=0.7,
        # max_tokens=4096,
        max_tokens=4096,
        temperature=0.15,
        # response_format={"type": "json", "value": dict(json_schema)},
        extra_body={"guided_json": dict(json_schema)},
    )

    # TGI compliant https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance#constrain-with-pydantic
    # response_format={"type": "json", "value": dict(json_schema)},
    # VLLM compliant
    # extra_body={"guided_json": dict(json_schema)},
    # extra_body={"guided_json": dict(json_schema), "chat_template_kwargs": {"enable_thinking": True}},  # qwen3 no reasoning
    # response_format={"type": "json_object", "schema": dict(json_schema)},
    # temperature=0.15,  # mistral
    # temperature=0.6,  # qwen3
    # top_p=0.95,  # qwen3
    # temperature=0.7,  # qwen3 no reasoning
    # top_p=0.8,  # qwen3 no reasoning
    # presence_penalty=1.5,  # qwen3 no reasoning
    # max_tokens=16384,
    # temperature=0.01  # phi4
    # temperature=1.0, top_p=0.9  # qwen_omni

    # content = response.choices[0].message.content
    # reasoning_content = response.choices[0].message.reasoning_content
    # return content.strip() if content is not None else reasoning_content.strip()

    return response.choices[0].message.content.strip()


def load_json(js, i=""):
    try:
        js = json.loads(js)
        js = {f"{k}{i}": v for k, v in js.items()}
        return js
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        print(f"Problematic response: {js}")
        return None


def safe_parse_json(text):
    """
    Extrahiert und parsed JSON oder Python-Dict-ähnliche Antworten sicher.
    Entfernt Markdown, Backticks und whitespace.
    Gibt {} zurück, wenn Parsing fehlschlägt.
    """
    if not text:
        return {}

    # Extrahiere JSON-ähnlichen Teil (alles zwischen { ... })
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    text = text.strip()

    # Versuch 1: Normales JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Versuch 2: Python-Dict (mit einfachen Anführungszeichen)
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    print(f"JSON decoding failed. Problematic response:\n{text}\n")
    return {}


def extract_subject_id(path: Path) -> str:
    parts = path.stem.split('_')
    # Beispiel: ['subj', '123', '80ui0t50']
    if len(parts) > 1:
        return parts[1]
    return None


def get_matches(paths1, paths2):
    # Dictionaries für einfaches Matching
    dict1 = {extract_subject_id(p): p for p in paths1}
    dict2 = {extract_subject_id(p): p for p in paths2}

    # Schnittmenge der IDs finden
    common_ids = sorted(set(dict1.keys()) & set(dict2.keys()))
    print(f"{len(common_ids)} gemeinsame Subjects gefunden")

    # Liste aller zusammengehörigen Paare
    matched_pairs = [(sid, dict1[sid], dict2[sid]) for sid in common_ids]
    return matched_pairs


base_path = Path("bix-whisper-formatted")
image_path = Path("/Users/franziskabraun/PycharmProjects/bixdata/MSP.png")
image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
picture_description_paths = [t for t in base_path.glob('**/*80ui0t50*.txt')]
recall_picture_description_paths = [t for t in base_path.glob('**/*u7kztp6d*.txt')]

story_reading_paths = [t for t in base_path.glob('**/*h6uxbwun*.txt')]
recall_story_reading_paths = [t for t in base_path.glob('**/*yt1y8hou*.txt')]

resps = []
for sid, task_path, recall_path in tqdm(get_matches(picture_description_paths, recall_picture_description_paths), desc="Processing subjects"):
    task = load_text_file(task_path)
    recall = load_text_file(recall_path)

    response_json = {'subject': sid}

    messages = [
        {
            "role": "system",
            "content": (
                "Sie sind ein Experte für Alzheimer's Demenz, der den kognitiven Status von Personen anhand ihrer Sprachleistung klassifiziert."
                "Sie müssen ein JSON-Objekt zurückgeben, das sich strikt an dieses Schema hält:\n" + str(json_schema)
            ),
        },
        {
            "role": "user",
            # "content": get_user_prompt_story(task),
            "content": [
                {"type": "text", "text": get_user_prompt_picture(task)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},  # Bild übergeben
            ],
        },
    ]

    response_pred = get_response(messages)
    pred_json = load_json(response_pred)
    print(pred_json)
    if pred_json == {}:
        print("Skipped subject: " + sid)
        continue
    response_json = {**response_json, **pred_json}
    resps.append(response_json)

results = pd.DataFrame(resps)
results.to_csv(f'{base_path}/bix_llm_picture_description_{model.split("/")[1]}_only.csv', index=False)

# "You are a precise clinical language model that classifies cognitive status."
# "First draft your thinking process (inner monologue) until you arrive at a response. Format your response using valid JSON. Write both your thoughts and the response in the same language as the input."
# "Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK] Here, only provide a valid JSON object that strictly follows this schema:\n" + str(json_schema)

# "Sie sind ein präzises klinisches Sprachmodell, das den kognitiven Status klassifiziert."
# "Sie müssen ein JSON-Objekt zurückgeben, das sich strikt an dieses Schema hält:\n" + str(json_schema)

# f"""
#     Der Originaltext der Geschichte lautet:
#     Johanna nahm den Flug von Frankfurt nach Barcelona, um ihren Cousin Hans zu besuchen.
#     Dort angekommen ging sie zur U-Bahn-Haltestelle und kaufte sich am Treppenaufgang ein Ticket.
#     Am Eingang sah sie das enge Drehkreuz.
#     Sie wusste nicht recht, wie sie mit dem Gepäck dort hindurchpassen würde.
#     Daher nahm sie den Wanderrucksack ab und warf ihn auf die andere Seite des Drehkreuzes.
#     Zusammen mit ihrem Rollkoffer zwängte sie sich durch das Drehkreuz.
#     Ein Pförtner beobachtete die Szene und lachte.
#     Er verwies auf die Schranke, die er hätte öffnen können.
#     Johanna ärgerte sich.
# """