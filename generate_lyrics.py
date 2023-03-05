import argparse
import os
import time

import numpy as np
from tqdm import tqdm

whisper_path = "/Users/petrosmitseas/Documents/Programming/whisper.cpp"

def generate_transcripts(input_dir):
    for song in tqdm(sorted(os.listdir(input_dir))):
        if not os.path.isdir(os.path.join(input_dir, song)):
            continue

        for cover in os.listdir(os.path.join(input_dir, song)):
            if not cover.endswith(".mp3"):
                continue

            file_path = os.path.join(input_dir, song, cover)
            resampled_path = file_path.replace('.mp3', '.wav')
            transcript_path = file_path.replace('.mp3', '.txt')
            print(f"generating {transcript_path}")

            # convert wav
            os.system(f'ffmpeg -y -i "{file_path}" -ac 1 -ar 16000 -c:a pcm_s16le "{resampled_path}"')

            # Call whisper
            os.system(f'{whisper_path}/main -m {whisper_path}/models/ggml-small.bin -f "{resampled_path}" -t 8 -nt > "{transcript_path}"')


def regenerate_transcripts(input_dir):
    st = time.time()
    runtimes = []
    for song in tqdm(sorted(os.listdir(input_dir))):
        if not os.path.isdir(os.path.join(input_dir, song)):
            continue

        for cover in os.listdir(os.path.join(input_dir, song)):
            if not cover.endswith(".wav"):
                continue

            file_path = os.path.join(input_dir, song, cover)
            transcript_path = file_path.replace('.wav', '.txt')
            print(f"generating {transcript_path}")

            st_per_song = time.time()
            # Call whisper
            os.system(f'{whisper_path}/main -m {whisper_path}/models/ggml-small.bin -f "{file_path}" -t 8 -nt > "{transcript_path}"')
            et_per_song = time.time()
            runtimes.append(et_per_song-st_per_song)

    et = time.time()
    print(f"Total time: {et-st}, avg. per song {np.mean(runtimes)}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('-i', '--input', type=str, help='input file path')

    args = parser.parse_args()

    generate_transcripts(args.input)
    # regenerate_transcripts(args.input)