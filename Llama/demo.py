# Created by guxu at 10/10/24

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define a class to encapsulate the LangChain process
class LangChainSummarizer:
    def __init__(self, model_name="llama3.2"):
        """
        Initializes the class with the specified model name.
        """
        self.template = '''Question: {question}
        Answer: Let's think step by step.
        '''
        self.model_name = model_name
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.model = OllamaLLM(model=model_name)
        self.chain = self.prompt | self.model

    def set_model(self, model_name):
        """
        Allows the user to change the model dynamically.
        """
        self.model_name = model_name
        self.model = OllamaLLM(model=model_name)
        self.chain = self.prompt | self.model

    def generate_response(self, question):
        """
        Generates a response from the model based on the input question.
        """
        output = self.chain.invoke({
            "question": question
        })
        return output

    def get_model_info(self):
        """
        Returns the current model name for reference.
        """
        return f"Current model: {self.model_name}"

# Example usage of the class
if __name__ == "__main__":
    # Initialize the summarizer with the default model (llama3.2)
    summarizer = LangChainSummarizer()

    # Sample question
    question = '''
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="llama3.1")

    chain = prompt | model

    chain.invoke({"question": "What is LangChain?"})
    '''

    # Generate a response
    print("Generating response with default model:")
    response = summarizer.generate_response(question)
    print(response)

    # Optionally, change the model to a different version (e.g., llama3.1)
    summarizer.set_model("llama3.2")
    print("\nGenerating response with new model (llama3.2):")
    response = summarizer.generate_response(question)
    print(response)

    # Output the current model info
    print("\n" + summarizer.get_model_info())
