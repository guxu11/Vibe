import os.path
from .. import PROJECT_BASE_PATH

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap

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
        loader = PyPDFLoader(pdf_path)
        pdf_documents = loader.load()
        self.documents.extend(pdf_documents)

    # Summarize the documents using MapReduce
    def summarize(self):
        """
        Summarize the loaded documents.

        Returns:
        - summary (str): The summarized content.
        """
        if not self.documents:
            return "No documents to summarize."

        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(self.documents)

        # map_prompt = ChatPromptTemplate.from_template(
        #     # SystemMessage(content="You are a professional summarization assistant."),
        #     # HumanMessage(content="Please summarize the following content:\n\n {text}")
        #     "You are a professional summarization assistant. Please summarize the following content:\n\n {text}"
        # )
        #
        combine_prompt = ChatPromptTemplate.from_template(
            "You are the reduce phase of a Map-Reduce summarization process. Please summarize the summaries from the map phase's output:\n\n{text} \n\n Please just give me a comprehesive summary without any other extra words."
        )

        collapse_prompt = ChatPromptTemplate.from_template(
             "You are the collapse phase of a Map-Reduce summarization process. Please compress the map phase's output, making it shorter than 1000 tokens, if it's already within 1000 tokens do nothing:\n\n{text}"
        )

        summary_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            # verbose=True,
            collapse_prompt=collapse_prompt,
            combine_prompt=combine_prompt
        )

        output_summary = summary_chain.invoke(input={"input_documents": docs})
        return output_summary["output_text"]

    def extract_key_info(self):
        """
        Extract key information, knowledge points, key tasks, and time points from the documents.

        Returns:
        - key_info (str): The extracted key information.
        """
        if not self.documents:
            return "No documents to extract information from."

        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(self.documents)


        prompt = ChatPromptTemplate.from_template(
            "You are an information extraction expert. Please extract key information, knowledge points, key tasks, and time points from the following content:\n\n {text}\n\nPlease list them in an organized manner."
        )

        # Create an LLMChain
        chain = prompt | self.llm

        key_infos = []
        for doc in docs:
            key_info = chain.invoke({"text": doc.page_content})
            key_infos.append(key_info)

        # Combine all key information
        final_key_info = "\n".join(key_infos)
        return final_key_info

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
    # Example usage of the LlamaWork class
    llama_worker = LlamaWorker()

    # Load text content
    with open(os.path.join(PROJECT_BASE_PATH, "data", "output", "output_text.txt"), "r") as f:
        text_content = f.read()
    llama_worker.load_text(text_content)
    # print(llama_worker.documents)

    # Load a PDF document
    # pdf_path = "sample.pdf"
    # llama_work.load_pdf(pdf_path)

    # Summarize the loaded content
    summary = llama_worker.summarize()
    print("Summarized content:")
    print(summary)

    # Extract key information from the documents
    # key_info = llama_worker.extract_key_info()
    # print("\nExtracted key information:")
    # print(key_info)