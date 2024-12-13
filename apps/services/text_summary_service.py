# Created by guxu at 10/24/24
from ..llama.llama_worker import LlamaWorker

class TextSummaryService:
    def __init__(self, model_name="gemma2:2b"):
        self.worker = LlamaWorker(model_name)

    def summarize_text(self, text):
        self.worker.load_text(text)
        return self.worker.summarize_with_my_map_reduce()