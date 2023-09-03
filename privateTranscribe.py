#!/usr/bin/env python3
import os
import time
import argparse
from dotenv import load_dotenv

from whisper_cpp_python import Whisper
# on macos
# update whisper_cpp.py to say
#    elif sys.platform == "darwin":
#        lib_ext = ".dylib"
from whisper_cpp_python.whisper_cpp import whisper_progress_callback

from files import save_file

model_path = os.environ.get('WHISPER_MODEL_PATH')
num_threads = int(os.environ.get('NUM_THREADS', 1))

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

def callback(ctx, state, i, p):
    print(f"{i}%")

def main():
    args = parse_arguments()
    model = Whisper(model_path=model_path, n_threads=num_threads)
    model.params.progress_callback = whisper_progress_callback(callback)
    with open(args.path, 'rb') as audio_file:
        start = time.time()
        output = model.transcribe(audio_file)
        end = time.time()
        print(f"\n> Transcription took {round(end - start, 2)}s.")
        new_file = args.path.split("/")[len(args.path.split("/"))-1].removesuffix(".wav").removesuffix(".mp3").removesuffix(".flac") + ".txt"
        save_file(output["text"], f"./source_documents/{new_file}")
    print("Now re-ingest your data and rerun the summary with the new context")

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateTranscribe: Transcribe audio offline')

    parser.add_argument('--path', default="", type=str,
                        help='the path to the file to be transcribed')

    return parser.parse_args()

if __name__ == "__main__":
    main()
