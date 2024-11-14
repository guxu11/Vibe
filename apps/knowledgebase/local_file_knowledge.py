# Created by guxu at 10/19/24
from .knowledge import Knowledge

class LocalFileKnowledge(Knowledge):
    def __init__(self, src_type, src_path=""):
        super().__init__(src_type, src_path)

    def build_knowledge(self):
        if not self.source_path:
            raise RuntimeError("No source path is found")
        with open(self.source_path, 'r') as f:
            line = f.readline()
            while line:
                line_strip = line.strip()
                if line_strip:
                    self.data.append(line_strip)
                line = f.readline()


if __name__ == '__main__':
    local_file_knowledges = LocalFileKnowledge("local", "knowledgebase/knowledge")
    # local_file_knowledges.add_text_knowledge_to_db()
    # local_file_knowledges.print_knowledge()
    print("result is ", local_file_knowledges.search_with_fitness_type("squat"))