"""Classify cognitive status (healthy / MCI / dementia) using a Vision-LLM and the OpenAI-compatible vLLM API"""
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, Optional

# Define JSON schema for structured output
class CognitiveAssessment(BaseModel):
    cognitive_status: Literal["healthy", "MCI", "dementia"]
    reasoning: Optional[str] = None  # optional explanation

# Set API connection to local vLLM (update your IP/port)
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"  # adapt to your running vLLM endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Optionally print model list
models = client.models.list()
model = models.data[0].id
print(f"Using model: {model}")

# Create JSON schema for output constraint
json_schema = CognitiveAssessment.model_json_schema()

# Example input data (replace with your patient data)
image_description = (
    "Eine ältere Frau steht in einer Küche und hält einen Löffel über einem Topf, "
    "während ein Mann ein Glas Wasser trinkt. Ein Kind steht neben ihr und schaut zu."
)
delayed_recall = (
    "Die Frau war am Kochen, und ein Kind war da, aber ich glaube, sie hat etwas verschüttet."
)

# Build prompt for classification
prompt = f"""
Du bist ein klinischer Assistent, der Patienten anhand ihrer Sprachleistung klassifiziert.

Hier ist eine Bildbeschreibung eines Patienten:
\"\"\"{image_description}\"\"\"

Hier ist der verzögerte Recall desselben Patienten:
\"\"\"{delayed_recall}\"\"\"

Klassifiziere den kognitiven Status in eine der drei Kategorien:
- healthy = keine kognitiven Auffälligkeiten
- MCI = milde kognitive Beeinträchtigung
- dementia = deutliche kognitive Einschränkung

Antworte NUR im JSON-Format entsprechend des gegebenen Schemas.
"""

messages = [
    {
        "role": "system",
        "content": (
            "You are a precise clinical language model that classifies cognitive status. "
            "Return only a JSON object following this schema:\n" + str(json_schema)
        ),
    },
    {"role": "user", "content": prompt},
]

# Send request
response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=512,
    temperature=0.2,
    response_format={"type": "json", "value": dict(json_schema)},
)

# Parse and print structured result
print(response.model_dump_json(indent=2))
