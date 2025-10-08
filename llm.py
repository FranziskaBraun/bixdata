import json
from pathlib import Path
import base64
import pandas as pd

from bixdata.dataloading import load_text_file
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, Optional
from tqdm import tqdm


# Define JSON schema for structured output
class CognitiveAssessment(BaseModel):
    cognitive_status: Literal["NCI", "MCI", "DEM"]
    # reasoning: Optional[str] = None  # optional explanation


# Set API connection to local vLLM (update your IP/port)
openai_api_key = "EMPTY"
# openai_api_base_mistral = "http://127.0.0.1:8073/v1"
openai_api_base_mistral_vllm_1 = "http://127.0.0.1:8091/v1"
openai_api_base_mistral_vllm_2 = "http://127.0.0.1:8092/v1"
openai_api_base_deephermes = "http://127.0.0.1:8090/v1"
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

# Build prompt for classification
# den Text in Bezug auf Kohärenz (text_coherence), lexikalische Vielfalt (lexical_diversity), Satzlänge (sentence_length) und Wortfindungsschwierigkeiten (word_finding_difficulties) mit Hilfe von Punkten zwischen 0 und 1 zu bewerten.
def get_user_prompt(picture_description, delayed_recall):
    prompt = f"""
    Sie sind ein klinischer Experte für Alzheimer's Demenz, der den kognitiven Status von Personen anhand ihrer Sprachleistung klassifiziert.
    Person A beschreibt das gegebene Bild auf dem eine Bergszene zu sehen ist (Bildbeschreibung).
    Nach Ablenkung muss die Person das Bild aus dem Gedächtnis beschreiben (Verzögerter Recall).
    Das Bild kann mit acht exemplarischen Konzepten beschrieben werden:
    - Eine Berglandschaft
    - Wanderer suchen den Weg
    - Eine Brücke führt über den Fluss
    - Kinder sitzen auf einem Baumstamm
    - Ein Junge ruft seine Eltern
    - Ein Mädchen fällt ins Wasser
    - Gewitterwolken mit Blitzen
    - Ein Haus in der Ferne
    
    Ihre Aufgabe darin, anhand der Bildbeschreibung und des verzögerten Recalls den kognitiven Status der Person in eine der drei Kategorien zu klassifizieren:
    - NCI = No Cognitive Impairment
    - MCI = Mild Cognitive Impairment
    - DEM = Mild to Severe Dementia
    
    Hier ist die Bildbeschreibung:
    \"\"\"{picture_description}\"\"\"
    
    Hier ist der verzögerte Recall:
    \"\"\"{delayed_recall}\"\"\"
    
    Geben Sie die Bewertung NUR im JSON-Format entsprechend des gegebenen Schemas an. Keine anderen Ausgaben.
    """
    return prompt


def get_user_prompt_story(story_reading, delayed_recall):
    prompt = f"""
    Sie sind ein klinischer Experte für Alzheimer's Demenz, der den kognitiven Status von Personen anhand ihrer Sprachleistung klassifiziert.
    Person A liest eine Geschichte Namens "Johanna Subway" vor (Geschichte Vorlesen).
    Nach Ablenkung muss die Person die Geschichte aus dem Gedächtnis erzählen (Verzögerter Recall).
    Der Orginaltext der Geschichte lautet:
    "Johanna nahm den Flug von Frankfurt nach Barcelona, um ihren Cousin Hans zu besuchen.
    Dort angekommen ging sie zur U-Bahn-Haltestelle und kaufte sich am Treppenaufgang ein Ticket.
    Am Eingang sah sie das enge Drehkreuz.
    Sie wusste nicht recht, wie sie mit dem Gepäck dort hindurchpassen würde.
    Daher nahm sie den Wanderrucksack ab und warf ihn auf die andere Seite des Drehkreuzes.
    Zusammen mit ihrem Rollkoffer zwängte sie sich durch das Drehkreuz. 
    Ein Pförtner beobachtete die Szene und lachte.
    Er verwies auf die Schranke, die er hätte öffnen können.
    Johanna ärgerte sich."

    Ihre Aufgabe besteht darin, anhand der vorgelsenen Geschichte und des verzögerten Recalls den kognitiven Status der Person in eine der drei Kategorien zu klassifizieren:
    - NCI = No Cognitive Impairment
    - MCI = Mild Cognitive Impairment
    - DEM = Mild to Severe Dementia

    Hier ist die vorgelesene Geschichte:
    \"\"\"{story_reading}\"\"\"

    Hier ist der verzögerte Recall:
    \"\"\"{delayed_recall}\"\"\"

    Geben Sie die Bewertung NUR im JSON-Format entsprechend des gegebenen Schemas an. Keine anderen Ausgaben.
    """
    return prompt


def get_response(messages):
    # Send request
    response = client.chat.completions.create(
        model=model,
        messages=messages,
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

    # return response.choices[0].message.reasoning_content.strip()
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
                    "Sie sind ein präzises klinisches Sprachmodell, das den kognitiven Status klassifiziert."
                    "Sie müssen ein JSON-Objekt zurückgeben, das sich strikt an dieses Schema hält:\n" + str(json_schema)
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": get_user_prompt(task, recall)},
                # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},  # Bild übergeben
            ],
        },
    ]

    response_pred = get_response(messages)
    pred_json = load_json(response_pred)
    response_json = {**response_json, **pred_json}
    resps.append(response_json)

results = pd.DataFrame(resps)
results.to_csv(f'{base_path}/bix_llm_picture_description_{model.split("/")[1]}_v2.csv', index=False)









