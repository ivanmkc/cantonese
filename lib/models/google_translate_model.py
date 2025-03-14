from google.cloud import translate_v2 as translate
from lib.models.model import Model
from typing import Optional

class GoogleTranslateModel(Model):    
    def __init__(self, target_language: str):
        self.client = translate.Client()
        self.target_language = target_language
    
    def translate(self, text: str) -> Optional[str]:
        try:
            result = self.client.translate(text, target_language=self.target_language)
            return result['translatedText']
        except Exception as e:
            return f"Error during translation: {e}"

