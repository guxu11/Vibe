# Created by guxu at 10/19/24
import time
from abc import ABC, abstractmethod
import chromadb
import pprint
from . import DATABASE_PATH


class Knowledge(ABC):
    def __init__(self, source_type, source_path=""):
        self.source_type = source_type
        self.source_path = source_path
        self.chroma_client = chromadb.PersistentClient(path=DATABASE_PATH)
        self.data = []

    @abstractmethod
    def build_knowledge(self):
        raise NotImplementedError("build_knowledge is not implemented")

    def make_ids(self, new_data):
        return [str(time.time()) + str(i) for i in range(1, len(new_data)+1)]
    def add_text_knowledge_to_db(self):
        self.build_knowledge()
        if not self.data:
            raise RuntimeError("knowledge data is empty")

        collection =self.chroma_client.get_or_create_collection(name="fitness")
        collection.add(
            documents=self.data,
            ids=self.make_ids(self.data)
        )

    def print_knowledge(self):
        print(self.chroma_client.list_collections())
        all_result = self.chroma_client.get_collection(name="fitness").get()
        pprint.pprint(all_result)

    def search_with_fitness_type(self, fitness_type):
        if not self.chroma_client:
            raise RuntimeError("chroma client is None")

        result = self.chroma_client.get_collection("fitness").query(
            query_texts=fitness_type,
            n_results=10,
            where_document={"$contains": fitness_type}
        )
        return result["documents"][0]


