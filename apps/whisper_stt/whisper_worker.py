import os
import whisper
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
from . import WHISPER_MODEL_LIST
from .. import PROJECT_BASE_PATH

class WhisperWorker:
    def __init__(self, model_size="tiny", with_gpu=False, output_dir=None):
        """
        Initialize the WhisperWorker with the desired model size, GPU option, and optional output directory.
        """
        assert model_size in WHISPER_MODEL_LIST, f"Invalid model size: {model_size}"
        self.model_size = model_size

        # Set the output directory for temporary audio files and transcription results
        self.output_dir = output_dir or PROJECT_BASE_PATH  # Default to project base path if not provided
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # Create the directory if it doesn't exist

        # Load Whisper model with optional GPU acceleration
        if with_gpu:
            self.model = whisper.load_model(model_size).to("cuda")
        else:
            self.model = whisper.load_model(model_size)

    def transcribe_audio_chunk(self, chunk_path):
        """
        Transcribes a single audio chunk and returns the transcription result.
        """
        result = self.model.transcribe(chunk_path, temperature=0)
        return result['text']


    def transcribe_audio(self, audio_file, chunk_size=30):
        """
        Transcribes a single audio file by splitting it into chunks of the specified size (in seconds).
        """
        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000  # Audio duration in seconds

        chunks = []
        for i in range(0, int(duration), chunk_size):
            start_time = i * 1000  # Convert to milliseconds
            end_time = min((i + chunk_size) * 1000, len(audio))
            chunk = audio[start_time:end_time]
            chunks.append(chunk)

        # Process each chunk
        chunk_path_map = {}
        for i, chunk in enumerate(chunks):
            # Create a path for the chunk in the specified output directory
            chunk_path = os.path.join(self.output_dir, "temp", f"temp_chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_path_map[chunk_path] = chunk

        transcriptions = {}
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.transcribe_audio_chunk, chunk_path): chunk_path for chunk_path in chunk_path_map.keys()}
            for future in as_completed(futures):
                chunk_path = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error processing {chunk_path}: {e}")
                    result = ""
                transcriptions[chunk_path] = result
                os.remove(chunk_path)

        # Combine transcriptions of all chunks
        return self.combine_transcriptions(transcriptions)

    def combine_transcriptions(self, transcriptions):
        """
        Combines the transcriptions from multiple audio chunks into a coherent single text.
        """
        combined = []
        size = len(transcriptions)

        for i in range(size):
            chunk_path = os.path.join(self.output_dir, "temp", f"temp_chunk_{i}.wav")
            combined.append(transcriptions[chunk_path])

        # Return the combined transcription as a single string
        return " ".join(combined)

    def transcribe_multiple_files(self, file_list, chunk_size=30):
        """
        Processes multiple audio files and aggregates the transcription results for each file.
        """
        all_results = {}

        # Process each file in parallel
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.transcribe_audio, file, chunk_size): file for file in file_list}
            for future in futures:
                file_name = futures[future]
                try:
                    transcription = future.result()
                    all_results[file_name] = transcription
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        return all_results


# Example usage
if __name__ == "__main__":
    import time
    # Define output directory within the project base path
    output_path = os.path.join(PROJECT_BASE_PATH, "data")
    print(output_path)

    # Initialize WhisperWorker with the desired model and output directory
    whisper_worker = WhisperWorker(output_dir=output_path)

    # Process multiple audio files
    # input_path = os.path.join(PROJECT_BASE_PATH, "data/audio/L1.mp3")
    input_path = "/Users/guxu/Movies/playground/normal_playground/demo_youtube.mp3"
    start_time = time.time()
    files = [input_path]
    results = whisper_worker.transcribe_multiple_files(files, chunk_size=30)

    # Output the transcription results
    for file, transcription in results.items():
        print(f"Transcription for {file}:")
        print(transcription)
        # with open(os.path.join(output_path, "output", "output.txt"), "w") as f:
        #     f.write(transcription)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.1f} seconds")
    print("done")

