import logging
from typing import Optional

from vertexai.generative_models import (GenerationConfig, GenerativeModel,
                                        HarmBlockThreshold, HarmCategory)

from lib.models.model import Model

logger = logging.getLogger()


class VertexAIGenerativeModel(Model):
    # https://github.com/googleapis/python-aiplatform/blob/main/vertexai/generative_models/_generative_models.py
    def __init__(self, model_name: str, source_language: str, target_language: str, additional_context: Optional[str] = None):
        self.source_language = source_language
        self.target_language = target_language
        self.additional_context = additional_context
        self.model = GenerativeModel(
            model_name,
            system_instruction=self.system_instructions,
        )

    @property
    def system_instructions(self) -> str:
        return f"""
# Instruction
Translate the following text to the target language. Only return the translated text and nothing else.

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
        # Set model parameters
        generation_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=30,
        )

        # Set safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Set contents to send to the model
        contents = [f"{text}\nTranslation:\n"]

        logger.debug(f"Contents:\n{text}")

        # Prompt the model to generate content
        response = self.model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        response_text = response.text

        translation = self.extract_translation_from_response(text=response_text)
        # logger.debug(f"\nPrompt: '{prompt}'\n\tResponse raw text: '{response_text}'\n\tInferred translation: '{translation}'\n")
        logger.debug(f"\nResponse raw text: '{response_text}'\n\tInferred translation: '{translation}'\n")

        return translation