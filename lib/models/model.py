import abc
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_random, wait_exponential
import concurrent

class Model(abc.ABC):
    @abc.abstractmethod
    def translate(self, text: str) -> Optional[str]:
        pass
    
    def translate_batch(self, texts: list[str]) -> list[Optional[str]]:
        @retry(stop=stop_after_attempt(3), 
               wait=wait_random(min=1, max=2), 
               retry_error_callback=lambda _: None)

        def _translate_text(text: str) -> str:
            return self.translate(text=text)
        
        predictions = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the thread pool
            futures = [executor.submit(_translate_text, text) for text in texts]

            for future in futures:
                predicted_translation = future.result()  # Wait for the result

                predictions.append(predicted_translation)

        return predictions        