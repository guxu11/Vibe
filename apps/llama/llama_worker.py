import os.path
from typing import List

from pydantic import BaseModel, Field

from .. import PROJECT_BASE_PATH

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import JsonOutputParser


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
            loader = PyPDFLoader(pdf_path)
            pdf_documents = loader.load()
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
            print(data[0])
            self.documents.extend(data)
        except Exception as e:
            print(f"Error loading image: {e}")
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
        print(docs)

        parser = JsonOutputParser(pydantic_object=Summary)
        combine_prompt = ChatPromptTemplate.from_template(
            """You are the reduce phase of a Map-Reduce summarization process. 
            Please summarize the summaries from the map phase's output:\n\n{text} \n\n 
            Please just give me a comprehesive summary without any other extra words."""
        )
        #
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

        summary_chain = summary_chain | parser

        output_summary = summary_chain.invoke(input={"input_documents": docs})
        return output_summary["output_text"]

    def summarize_with_my_map_reduce(self):
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
            template = """You have a list of summaries which are extracted from ordered document chunks. Please combine and summarize them into a single summary:\n\n{text}
            
            You need to get the context, topic, key terms, and summary from the extracted key information.
            Context is the background or setting where the content occurs, e.g., math class, team meeting, casual conversation, etc.
            Topic is the main topic or subject matter of the content.
            Key terms are important keywords or terms extracted from the content.
            Summary must be a concise summary that captures the main ideas and key details. Your summary should be clear, coherent, and written in complete sentences. 
            
            The output should be formatted as a JSON instance that conforms to the JSON schema below. The output must only contain a json object without any additional text.
            
            Only give me the whole json object without any other extra words.
            {{
                "context": "",
                "topic": "",
                "key_terms": [],
                "summary": ""
            }}
            """,
            input_varialbes = ["text"],
            # partial_variables = {"format_instructions": combine_parser.get_format_instructions()}
        )

        combine_chain = combine_prompt | self.llm | combine_parser
        combined_summary = combine_chain.invoke({"text": final_key_info})
        return combined_summary

    # def supportive_document(self):


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

class Summary(BaseModel):
    context: str = Field(
        description="The background or setting where the content occurs, e.g., math class, team meeting, casual conversation, etc."
    )
    topic: str = Field(
        description="The main topic or subject matter of the content."
    )
    key_terms: List[str] = Field(
        description="Important keywords or terms extracted from the content."
    )
    summary: str = Field(
        description="A detailed summary of the content."
    )

class ChunkSummary(BaseModel):
    summary: str = Field(
        description="The summary of the chunk."
    )

if __name__ == '__main__':
    # Example usage of the LlamaWork class
    llama_worker = LlamaWorker()

    # Load text content
    with open(os.path.join(PROJECT_BASE_PATH, "data", "output", "output.txt"), "r") as f:
        text_content = f.read()
    llama_worker.load_text(text_content)
    # llama_worker.load_image(os.path.join(PROJECT_BASE_PATH, "data", "image", "img.png"))
    # llama_worker.load_pdf(os.path.join(PROJECT_BASE_PATH, "data", "pdf", "meta_BQ_preparation.pdf"))
    # print(llama_worker.documents)

    # Load a PDF document
    # pdf_path = "sample.pdf"
    # llama_work.load_pdf(pdf_path)

    # Summarize the loaded content
    summary = llama_worker.summarize_with_my_map_reduce()
    print("Summarized content:")
    print(summary)

    # Extract key information from the documents
    # key_info = llama_worker.extract_key_info()
    # print("\nExtracted key information:")
    # print(key_info)