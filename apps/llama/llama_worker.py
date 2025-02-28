import os.path
from .. import PROJECT_BASE_PATH

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import JsonOutputParser


class LlamaWorker:
    def __init__(self, model_name="llama3.2"):
        """
        Initialize the LlamaWork class.

        Parameters:
        - model_name (str): The name of the Llama model.
        """
        self.model_name = model_name
        # Initialize the Ollama LLM
        self.llm = OllamaLLM(model=self.model_name, num_threads=8, temperature=0)
        self.documents = []
        self.supportive_documents = []

    def load_text(self, text):
        """
        Load and store text content.

        Parameters:
        - text (str): The text content to be loaded.
        """
        self.documents.append(Document(page_content=text))

    def load_pdf(self, pdf_path):
        """
        Load and store a PDF document.

        Parameters:
        - pdf_path (str): The file path to the PDF document.
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
            loader = PyPDFLoader(pdf_path, extract_images=True)
            pdf_documents = loader.load()
            print("PDF documents: ", pdf_documents)
            self.documents.extend(pdf_documents)
        except Exception as e:
            print(f"Error loading PDF document: {e}")

    def load_image(self, image_path):
        """
        Load and store an image.

        Parameters:
        - image_path (str): The file path to the image.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found at path: {image_path}")
            loader = UnstructuredImageLoader(image_path)
            data = loader.load()
            print("Image data: ", data)
            self.documents.extend(data)
        except Exception as e:
            print(f"Error loading image: {e}")

    def summarize(self):
        """
        Extract key information, knowledge points, key tasks, and time points from the documents.

        Returns:
        - key_info (str): The extracted key information.
        """
        if not self.documents:
            return "No documents to extract information from."

        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)
        docs = text_splitter.split_documents(self.documents)
        # chunk_parser = JsonOutputParser(pydantic_object=ChunkSummary)
        chunk_parser = JsonOutputParser()
        chunk_prompt = PromptTemplate(
            template = """
            Please read the following text segment carefully and provide a concise summary that captures 
            the main ideas and key details. Your summary should be clear, coherent, and written in complete sentences. 
            The summary should only contain the content and should not start or end with any additional text.
            
            {text}
            
            The output should be formatted as a JSON instance that conforms to the JSON schema below. The output must only contain a json object without any additional text.
            '{{"summary": ""}}'
            
            """,
            input_variables = ["text"],
            # partial_variables = {"format_instructions": chunk_parser.get_format_instructions()}
        )

        # Create an LLMChain
        chunk_chain = chunk_prompt | self.llm | chunk_parser

        key_infos = []
        for doc in docs:
            try:
                key_info = chunk_chain.invoke({"text": doc.page_content})
            except Exception as e:
                key_info = {"summary": ""}
                print(f"Error summarizing chunk: {e}")
            print("key_info: ", key_info)
            key_infos.append(key_info)

        # Combine all key information
        final_key_info = "\n\n".join(key_info["summary"] for key_info in key_infos)
        # combine_parser = JsonOutputParser(pydantic_object=Summary)
        combine_parser = JsonOutputParser()
        combine_prompt = PromptTemplate(
            template="""You have a list of summaries extracted from ordered document chunks. These summaries are parts of a larger coherent document. Your task is to combine these summaries into one comprehensive summary.

            Here is the input summaries:
            \n\n{text}

            You need to:
            1. Identify the context as a concise descriptor of the setting. Avoid speculative phrases like "this content is likely from".
            2. Determine the topic as a short phrase that clearly summarizes the subject matter.
            3. Extract the key terms, which are significant keywords or terms central to the content.
            4. Create a combined summary by merging the input summaries into one. The summary must be comprehensive, clear, coherent, and written in complete sentences. Avoid redundancy while preserving all important information.

            The output must be formatted as a JSON object that adheres to the schema below. Do not include any text outside the JSON object.

            {{
                "context": "",
                "topic": "",
                "key_terms": [],
                "summary": ""
            }}
            """,
            input_variables=["text"]
        )

        combine_chain = combine_prompt | self.llm | combine_parser
        combined_summary = combine_chain.invoke({"text": final_key_info})
        return combined_summary


    # Placeholder methods for future integration with tarily and ChromaDB
    def set_vector_store(self, vector_store):
        """
        Set the vector store for Retrieval Augmented Generation (RAG).

        Parameters:
        - vector_store: The vector store instance (e.g., ChromaDB).
        """
        self.vector_store = vector_store

    def set_retrieval_augmentation(self, retriever):
        """
        Set the retriever for RAG.

        Parameters:
        - retriever: The retriever instance.
        """
        self.retriever = retriever


if __name__ == '__main__':
    from .summary_measure import SummaryMeasurement

    # Example usage of the LlamaWork class
    model_list = [
        "opencoder:1.5b", "codegemma:2b", "qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:0.5b",
        "nemotron-mini:latest", "phi3.5:latest", "llama3.2:1b", "llama3.2:3b", "gemma2:2b"
    ]
    metric_collection = {}
    summary_measurement = SummaryMeasurement()
    documents_path = os.path.join(PROJECT_BASE_PATH, "data", "output")
    document_list = os.listdir(documents_path)
    num_documents = len(document_list)
    for i, document in enumerate(document_list):
        print(f"========{i+1} / {num_documents} is in process============")
        print(document)
        with open(os.path.join(documents_path, document), "r") as f:
            text_content = f.read()

        summaries = {}
        for model in model_list:
            print(f"======== {model} is running ... ========")
            llama_worker = LlamaWorker(model)

            # Load text content
            llama_worker.load_text(text_content)
        # llama_worker.load_image(os.path.join(PROJECT_BASE_PATH, "data", "image", "img.png"))
        # llama_worker.load_pdf(os.path.join(PROJECT_BASE_PATH, "data", "pdf", "meta_BQ_preparation.pdf"))
        # print(llama_worker.documents)

        # Load a PDF document
        # pdf_path = "sample.pdf"
        # llama_work.load_pdf(pdf_path)
        # Summarize the loaded content
            try:
                output = llama_worker.summarize_with_my_map_reduce()
                summary = output["summary"]
                summaries[model] = summary
            except Exception as e:
                print(e)
                summaries[model] = "No summary"

        measurements = summary_measurement.measure(summaries, text_content)
        if not metric_collection:
            metric_collection.update(measurements)
        else:
            for model, metrics in measurements.items():
                for m, point in metrics.items():
                    metric_collection[model][m] += point


    for model, metrics in metric_collection.items():
        for m in metrics:
            metrics[m] = round(metrics[m] / num_documents, 2)

    print(metric_collection)
    summary_measurement.draw_plot(metric_collection)

    res = {'opencoder:1.5b': {'Relevance': 1.0, 'Coherence': 1.0, 'Consistency': 1.0, 'Fluency': 1.0},
     'codegemma:2b': {'Relevance': 1.0, 'Coherence': 1.0, 'Consistency': 1.0, 'Fluency': 1.0},
     'qwen2.5:3b': {'Relevance': 3.62, 'Coherence': 3.88, 'Consistency': 3.38, 'Fluency': 3.0},
     'qwen2.5:1.5b': {'Relevance': 3.5, 'Coherence': 3.88, 'Consistency': 3.88, 'Fluency': 2.88},
     'qwen2.5:0.5b': {'Relevance': 3.12, 'Coherence': 3.25, 'Consistency': 3.75, 'Fluency': 3.0},
     'nemotron-mini:latest': {'Relevance': 3.0, 'Coherence': 3.25, 'Consistency': 4.0, 'Fluency': 3.0},
     'phi3.5:latest': {'Relevance': 4.25, 'Coherence': 4.38, 'Consistency': 4.12, 'Fluency': 2.88},
     'llama3.2:1b': {'Relevance': 2.62, 'Coherence': 2.62, 'Consistency': 3.25, 'Fluency': 2.5},
     'llama3.2:3b': {'Relevance': 3.75, 'Coherence': 3.75, 'Consistency': 3.88, 'Fluency': 3.0},
     'gemma2:2b': {'Relevance': 4.5, 'Coherence': 4.62, 'Consistency': 4.62, 'Fluency': 3.0}}
