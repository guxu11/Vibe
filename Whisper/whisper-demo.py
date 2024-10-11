# Created by guxu at 9/12/24
import time

from openai import OpenAI
import whisper
from concurrent.futures import ThreadPoolExecutor
from utils.utils import *
import dotenv


def call_whisper_api():
  dotenv.load_dotenv()
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

  audio_file = open("../data/audio/demo.mp3", "rb")

  transcription = "No content"
  try:
    transcription = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file,
      response_format="text"
    )
  except Exception as e:
    print("Error", e)

  return transcription

def whisper_sdk_basic_demo(input_path, output_path, model_type="base"):
  start_time = time.time()
  model = whisper.load_model(model_type)
  result = model.transcribe(input_path)
  end_time = time.time()
  print(f"{model_type} model running time: {end_time-start_time:.1f}")
  write_output(output_path, result["text"])
  return result["text"]

def whisper_sdk_advanced_demo(input_path, output_path):
  model = whisper.load_model("base")

  # load audio and pad/trim it to fit 30 seconds
  audio = whisper.load_audio(input_path)
  audio = whisper.pad_or_trim(audio)

  # make log-Mel spectrogram and move to the same device as the model
  mel = whisper.log_mel_spectrogram(audio).to(model.device)

  # detect the spoken language
  _, probs = model.detect_language(mel)
  print(f"Detected language: {max(probs, key=probs.get)}")

  # decode the audio
  options = whisper.DecodingOptions()
  result = whisper.decode(model, mel, options)

  # print the recognized text
  write_output(output_path, result.text)
  return(result.text)

if __name__ == '__main__':
   input_path = "../data/audio/demo.mp3"
   model_sizes = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium']
   out_paths = [f'../data/output/demo_{model_size}.txt' for model_size in model_sizes]
   print(out_paths)

   with ThreadPoolExecutor() as executor:
     # submit tasks
     futures = []
     for index, model in enumerate(model_sizes):
      future = executor.submit(whisper_sdk_basic_demo, input_path, out_paths[index], model_sizes[index])
      futures.append(future)

      for future in futures:
        try:
          result = future.result()
          print("Task completed successfully:", result)
        except Exception as e:
          print(f"Task failed with exception: {e}")
