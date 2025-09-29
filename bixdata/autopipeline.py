from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch


class CustomAudioPipeline:
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = None  # Initialize your classifier here

    def preprocess_audio(self, audio):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000, padding=True)
        return inputs

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Determine which hidden states to use based on model architecture
        if hasattr(outputs, 'encoder_last_hidden_state'):
            hidden_states = outputs.encoder_last_hidden_state
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError("Model does not provide 'encoder_last_hidden_state' or 'last_hidden_state'")

        return hidden_states

    def classify(self, hidden_states):
        if self.classifier is None:
            raise ValueError("Classifier not initialized")

        # Example: classify using a simple linear layer
        logits = self.classifier(hidden_states)
        return logits


class CustomTextPipeline:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = None  # Initialize your classifier here

    def preprocess_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Determine which hidden states to use based on model architecture
        if hasattr(outputs, 'encoder_last_hidden_state'):
            hidden_states = outputs.encoder_last_hidden_state
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError("Model does not provide 'encoder_last_hidden_state' or 'last_hidden_state'")

        return hidden_states

    def classify(self, hidden_states):
        if self.classifier is None:
            raise ValueError("Classifier not initialized")

        # Example: classify using a simple linear layer
        logits = self.classifier(hidden_states)
        return logits


class CustomSeqTextPipeline:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = None  # Initialize your labels or class mapping here

    def preprocess_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        return logits

# # Example usage for audio pipeline
# model_name = "openai/whisper-small"
# audio_pipeline = CustomAudioPipeline(model_name)
#
# # Example audio input (replace with your audio loading/preprocessing logic)
# audio = torch.randn(1, 16000)  # Example audio tensor
#
# # Preprocess audio
# inputs = audio_pipeline.preprocess_audio(audio)
#
# # Forward pass through the model
# hidden_states = audio_pipeline.forward(inputs)
#
# # Perform classification
# logits = audio_pipeline.classify(hidden_states)
