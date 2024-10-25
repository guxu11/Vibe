# Created by guxu at 10/4/24
from model_measure_pipeline import ModelMeasurePipeline
from typing import List

import os
import pandas as pd
import whisper

class WhisperMeasurePipeline(ModelMeasurePipeline):
    BASE_PATH = "/Users/guxu/Documents/work space/SFSU courses/CSC899/CSC899-Professor Wang/cv-corpus-19.0-delta-2024-09-13/en"
    def __init__(self, size_list: List, **kwargs):
        super().__init__(model_name="Whisper", size_list=size_list, **kwargs)
        self.models = {model_name: whisper.load_model(model_name) for model_name in self.size_list}

    def load_data(self):
        print("Loading data...")
        reference_text_file = "other.tsv"
        real_path = os.path.join(WhisperMeasurePipeline.BASE_PATH, reference_text_file)
        data = pd.read_csv(real_path, sep='\t', header=0)
        paths = data["path"]
        sentences = data["sentence"]
        self.data = [{"audio_path": audio, "transcript": sentence} for audio, sentence in zip(paths, sentences)]
        print("Finish loading data")

    def _add_noise(self, audio_file_path):
        pass



    def preprocess(self, add_noise: bool = False):
        print("Preprocess data...")
        pass

        print("Finish preprocessing data")


    def _create_model_task(self, *args):
        pass

if __name__ == '__main__':
    whisper_measure_pipeline = WhisperMeasurePipeline([])
    print(whisper_measure_pipeline.data[0])



