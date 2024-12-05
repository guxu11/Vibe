# Created by guxu at 11/14/24
import json
import os

from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np

# Evaluation prompt template based on G-Eval
EVALUATION_PROMPT_TEMPLATE = """
                         You will be given one summary written for an article. Your task is to rate the summary on one metric.
                         Please make sure you read and understand these instructions very carefully. 
                         Please keep this document open while reviewing, and refer to it as needed.
                         
                         Evaluation Criteria:
                         
                         {criteria}
                         
                         Evaluation Steps:
                         
                         {steps}
                         
                         Example:
                         
                         Source Text:
                         
                         {document}
                         
                         Summary:
                         
                         {summary}
                         
                         Evaluation Form (scores ONLY):
                         
                         - {metric_name}
                         """

# Metric 1: Relevance

RELEVANCY_SCORE_CRITERIA = """
                       Relevance(1-5) - selection of important content from the source. \
                       The summary should include only important information from the source document. \
                       Annotators were instructed to penalize summaries which contained redundancies and excess information.
                       The result must be a json object in the following format "{"score": [1-5]}"
                       """

RELEVANCY_SCORE_STEPS = """
                    1. Read the summary and the source document carefully.
                    2. Compare the summary to the source document and identify the main points of the article.
                    3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
                    4. Assign a relevance score from 1 to 5.
                    """

# Metric 2: Coherence

COHERENCE_SCORE_CRITERIA = """
                       Coherence(1-5) - the collective quality of all sentences. \
                       We align this dimension with the DUC quality question of structure and coherence \
                       whereby "the summary should be well-structured and well-organized. \
                       The summary should not just be a heap of related information, but should build from sentence to a\
                       coherent body of information about a topic."
                       The result must be a json object in the following format "{"score": [1-5]}"
                       """

COHERENCE_SCORE_STEPS = """
                    1. Read the article carefully and identify the main topic and key points.
                    2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
                    and if it presents them in a clear and logical order.
                    3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
                    """

# Metric 3: Consistency

CONSISTENCY_SCORE_CRITERIA = """
                         Consistency(1-5) - the factual alignment between the summary and the summarized source. \
                         A factually consistent summary contains only statements that are entailed by the source document. \
                         Annotators were also asked to penalize summaries that contained hallucinated facts.
                         The result must be a json object in the following format "{"score": [1-5]}"
                         """

CONSISTENCY_SCORE_STEPS = """
                      1. Read the article carefully and identify the main facts and details it presents.
                      2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
                      3. Assign a score for consistency based on the Evaluation Criteria.
                      """

# Metric 4: Fluency

FLUENCY_SCORE_CRITERIA = """
                     Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
                     1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
                     2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
                     3: Good. The summary has few or no errors and is easy to read and follow.
                     The result must be a json object in the following format "{"score": [1-3]}"
                     """

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""
class SummaryMeasurement:
    def __init__(self):
        self.judge = OpenAI(
            # make sure add OPENAI_API_KEY to environment variables
            api_key =os.getenv("OPENAI_API_KEY")
        )
    def get_geval_score(
            self, criteria: str, steps: str, document: str, summary: str, metric_name: str
    ):
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            criteria=criteria,
            steps=steps,
            metric_name=metric_name,
            document=document,
            summary=summary,
        )
        response = self.judge.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"},
        )
        result_json = response.choices[0].message.content + "}"
        # print(result_json)
        try:
            score = json.loads(result_json)["score"]
        except Exception as e:
            score = 1
            print(e)
        return score

    def measure(self, summaries, document):
        evaluation_metrics = {
            "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
            "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
            "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
            "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
        }

        data = {}

        for model_name, summary in summaries.items():
            data[model_name] = {}
            for eval_type, (criteria, steps) in evaluation_metrics.items():
                result = self.get_geval_score(criteria, steps, document, summary, eval_type)
                data[model_name][eval_type] = int(result)
        return data

    def draw_plot(self, data):
        # Prepare data
        model_names = list(data.keys())
        categories = ['Relevance', 'Coherence', 'Consistency', 'Fluency']
        scores = [list(model.values()) for model in data.values()]

        # Bar positions and width
        x = np.arange(len(model_names))
        bar_width = 0.8

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bottoms = np.zeros(len(model_names))
        # Improved color palette
        colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0']

        for idx, category in enumerate(categories):
            heights = [score[idx] for score in scores]
            ax.bar(x, heights, bar_width, bottom=bottoms, label=category, color=colors[idx])
            bottoms += heights

        ax.set_xlabel('Model Name')
        ax.set_ylabel('Scores')
        ax.set_title('Performance Scores by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(title="Categories")

        # No gridlines
        ax.grid(False)

        # Show plot
        plt.tight_layout()
        plt.savefig(f"metrics.png")
        plt.show()



if __name__ == '__main__':
    data = {'opencoder:1.5b': {'Relevance': 1, 'Coherence': 1, 'Consistency': 1, 'Fluency': 1}, 'codegemma:2b': {'Relevance': 1, 'Coherence': 1, 'Consistency': 1, 'Fluency': 1}, 'qwen2.5:3b':  {'Relevance': 5, 'Coherence': 5, 'Consistency': 5, 'Fluency': 3}, 'qwen2.5:1.5b': {'Relevance': 4, 'Coherence': 5, 'Consistency': 4, 'Fluency': 3}, 'qwen2.5:0.5b': {'Relevance': 4, 'Coherence': 4, 'Consistency': 5, 'Fluency': 3}, 'nemotron-mini:latest': {'Relevance': 4, 'Coherence': 4, 'Consistency': 5, 'Fluency': 3}, 'phi3.5:latest':  {'Relevance': 5, 'Coherence': 5, 'Consistency': 5, 'Fluency': 3}, 'llama3.2:1b':  {'Relevance': 4, 'Coherence': 4, 'Consistency': 5, 'Fluency': 3}, 'llama3.2:3b':  {'Relevance': 4, 'Coherence': 4, 'Consistency': 3, 'Fluency': 3}, 'gemma2:2b': {'Relevance': 5, 'Coherence': 5, 'Consistency': 5, 'Fluency': 3}}
    sm = SummaryMeasurement()
    sm.draw_plot(data)