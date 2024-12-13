from .llama_worker import LlamaWorker
from .. import PROJECT_BASE_PATH
import os

if __name__ == '__main__':
    worker = LlamaWorker("gemma2:2b")
    documents_path = os.path.join(PROJECT_BASE_PATH, "data", "output")
    file_name = "toefl.txt"
    with open (os.path.join(documents_path, file_name), "r") as myfile:
        data=myfile.read()

    worker.load_text(data)
    summary = worker.summarize_with_my_map_reduce()
    print(summary)