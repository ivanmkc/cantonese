import logging
from typing import List, Optional

from google.cloud import aiplatform

from tenacity import retry, stop_after_attempt
from lib.models.model import Model

# Constants
max_tokens = 60
max_tokens_description = 90
temperature = 0.0
top_p = 1.0
top_k = 1
raw_response = True

logger = logging.getLogger()


class VertexAIEndpointModel(Model):
    def __init__(self, endpoint: str, location: str, source_language: str, target_language: str, additional_context: Optional[str] = None):
        self.source_language = source_language
        self.target_language = target_language
        self.endpoint = aiplatform.Endpoint(endpoint, location=location)
        self.additional_context = additional_context

    @property
    def system_instructions(self) -> str:
        return f"""
# Instruction
Translate the following text to the target language. Only return the translated text and nothing else. Do not include explanations.

# Source language: {self.source_language}
# Target language: {self.target_language}
{self.additional_context or ""}

# Format

Text: <text_to_translate>
Translation: <translated_text>

# Translate the below text
        """

    def format_prompt(self, text: str) -> str:
        return f'{self.system_instructions}Text: "{text}"\n\nTranslation: '

    def extract_translation_from_response(self, text: str) -> str:
        return text.strip()

    def translate(self, text: str) -> Optional[str]:
        prompt = self.format_prompt(text=text)

        instances = [
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "raw_response": raw_response,
            },
        ]
        response = self.endpoint.predict(instances=instances)

        response_text = response.predictions[0]

        translation = self.extract_translation_from_response(text=response_text)
        # logger.debug(f"\nPrompt: '{prompt}'\n\tResponse raw text: '{response_text}'\n\tInferred translation: '{translation}'\n")
        logger.debug(f"\nResponse raw text: '{response_text}'\n\tInferred translation: '{translation}'\n")

        return translation