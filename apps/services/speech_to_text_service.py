# Created by guxu at 10/24/24
from ..whisper_stt.whisper_worker import WhisperWorker
import os
from .. import PROJECT_BASE_PATH

class SpeechToTextService:
    def __init__(self):
        self.worker = WhisperWorker(output_dir=os.path.join(PROJECT_BASE_PATH, "data"))

    def speech_to_text(self, audio):
        return self.worker.transcribe_audio(audio)
