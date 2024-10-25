# Created by guxu at 9/19/24

import csv
import os

import whisper
import jiwer
import pandas as pd
import time
import matplotlib.pyplot as plt

BASE_PATH = "/Users/guxu/Documents/work space/SFSU courses/CSC899/Professor Wang/cv-corpus-19.0-delta-2024-09-13/en"
class SpeechToTextExperiment:
    def __init__(self, reference_text_file, duration_file, output_csv, models=["tiny", "tiny.en", "base", "base.en", "small", "small.en"]):
        """
        初始化实验类，加载不同的Whisper模型。
        :param reference_text_file: 包含音频文件路径和原始文本的TSV文件路径
        :param output_csv: 输出结果的CSV文件路径
        :param models: 需要比较的Whisper模型列表
        """
        self.reference_text_file = reference_text_file
        self.duration_file = duration_file
        self.transcripts = {}
        self.durations = {}
        self.output_csv = output_csv
        self.models = models
        self.loaded_models = {model_name: whisper.load_model(model_name) for model_name in models}

    def load_data(self):
        """
        从TSV文件加载音频文件路径和对应的文本。
        """
        real_path = os.path.join(BASE_PATH, self.reference_text_file)
        data = pd.read_csv(real_path, sep='\t', header=0)
        paths = data["path"]
        sentences = data["sentence"]
        self.transcripts = {k: v for k, v in zip(paths, sentences)}
        
    def load_duration(self):
        real_path = os.path.join(BASE_PATH, self.duration_file)
        data = pd.read_csv(real_path, sep='\t', header=0)
        paths = data["clip"]
        duration = data["duration[ms]"]
        self.durations = {k: v for k, v in zip(paths, duration)}

    def transcribe_audio(self, audio_path, model):
        """
        使用给定的Whisper模型转录音频，并记录执行时间。
        :param audio_path: 音频文件路径
        :param model: Whisper模型实例
        :return: 转录文本和执行时间
        """
        real_audio_path = os.path.join(os.path.join(BASE_PATH, 'clips'), audio_path)
        start_time = time.time()
        result = model.transcribe(real_audio_path)
        end_time = time.time()

        transcription = result['text']
        execution_time = end_time - start_time

        return transcription, execution_time

    def calculate_metrics(self, reference, hypothesis):
        """
        计算Word Error Rate (WER)等常见的评估指标。
        :param reference: 原始文本
        :param hypothesis: 转录文本
        :return: 评估指标字典
        """
        wer = jiwer.wer(reference, hypothesis)
        return {"WER": wer}

    @staticmethod
    def sort_by_tiny(df):
        pass
    @staticmethod
    def different_model_plot_results(csv_file, metric="WER", en=False):
        """
        从CSV文件中读取结果并绘制折线图，每个模型一条折线，表示不同音频的WER变化。
        :param csv_file: 包含实验结果的CSV文件路径
        """
        # 读取CSV文件到DataFrame
        df = pd.read_csv(csv_file)

        # 获取所有模型的唯一列表
        models = ['tiny', 'base', 'small']
        title = f'{metric} Comparison of Different Multilingual Whisper Models'
        if en:
            models = [model + '.en' for model in models]
            title = f'{metric} Comparison of Different English-only Whisper Models'

        # 初始化图形
        plt.figure(figsize=(10, 6))

        # 为每个模型绘制一条折线
        for model in models:
            # 过滤出该模型的所有数据点
            model_data = df[df['model'] == model][:100]

            model_data = model_data.reset_index()
            if metric == "WER":
                model_data = model_data[(model_data[metric] >= 0) & (model_data[metric] <= 1)]

            # 使用音频文件的索引作为X轴，WER作为Y轴绘制折线图
            plt.plot(model_data.index, model_data[metric], label=model)

        # 添加图例、标题和标签
        plt.title(title)
        plt.xlabel('Audio File Index')  # 仅显示X轴标签，不显示具体文件名
        plt.ylabel('WER')
        # plt.ylim(0, 1)  # WER 的范围应该在0到1之间
        plt.legend(title='Model')

        # 显示图形
        plt.tight_layout()
        plt.savefig(f"../data/figure/{title}.png")

    @staticmethod
    def mul_vs_en_plot_results(csv_file, m, metric="WER"):
        """
        从CSV文件中读取结果并绘制折线图，每个模型一条折线，表示不同音频的WER变化。
        :param csv_file: 包含实验结果的CSV文件路径
        """
        # 读取CSV文件到DataFrame
        df = pd.read_csv(csv_file)

        # 获取所有模型的唯一列表
        models = [m, m + '.en']

        # 初始化图形
        plt.figure(figsize=(10, 8))

        # 为每个模型绘制一条折线
        for model in models:
            # 过滤出该模型的所有数据点
            model_data = df[df['model'] == model][:100]
            model_data = model_data.reset_index()
            if metric == "WER":
                model_data = model_data[(model_data[metric] >= 0) & (model_data[metric] <= 1)]

            # 使用音频文件的索引作为X轴，WER作为Y轴绘制折线图
            plt.plot(model_data.index, model_data[metric], label=model)

        # 添加图例、标题和标签
        title = f'{metric} Comparison of {m} and {m}.en'
        plt.title(title)
        plt.xlabel('Audio File Index')  # 仅显示X轴标签，不显示具体文件名
        plt.ylabel('WER')
        # plt.ylim(0, 1)  # WER 的范围应该在0到1之间
        plt.legend(title='Model')

        # 显示图形
        plt.tight_layout()
        plt.savefig(f"../data/figure/{title}.png")

    def save_result(self, result):
        """
        将单个结果追加写入到CSV文件中。
        """
        # 检查文件是否存在，如果不存在则写入表头
        try:
            with open(self.output_csv, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['audio_file', 'duration', 'model', 'transcript',
                                                          'predicted_transcript', 'WER', 'execution_time'])
                # 如果文件是空的，写入表头
                if file.tell() == 0:
                    writer.writeheader()
                # 写入单行结果
                writer.writerow(result)
        except Exception as e:
            print(f"Failed to write result: {e}")

# 使用示例
# experiment = SpeechToTextExperiment(reference_text_file='other.tsv', duration_file="clip_durations.tsv",output_csv='results.csv')
# experiment.run_experiment()
# SpeechToTextExperiment.different_model_plot_results('results.csv', "execution_time", True)
for model in ['tiny', 'base', 'small']:
    for metric in ["WER", "execution_time"]:
        SpeechToTextExperiment.mul_vs_en_plot_results("results.csv", model, metric)


