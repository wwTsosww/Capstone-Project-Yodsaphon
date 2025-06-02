
import os
import numpy as np
import librosa
from tqdm import tqdm

# Parameters
sr = 44100
n_fft = 2048
hop_length = 512
n_mels = 128
fmax = 8000
segment_duration = 10  # seconds
segment_samples = sr * segment_duration

# Paths
STEM_TRAIN_DIR = "Data_set/Stem/train"
STEM_TEST_DIR = "Data_set/Stem/test"
OUTPUT_SPEC_TRAIN = "Data_set/Spectrogram/train"
OUTPUT_SPEC_TEST = "Data_set/Spectrogram/test"

def extract_mel_segments(audio_path):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    segments = []
    for start in range(0, len(y), segment_samples):
        end = start + segment_samples
        y_segment = y[start:end]

        if len(y_segment) < segment_samples:
            pad_width = segment_samples - len(y_segment)
            y_segment = np.pad(y_segment, (0, pad_width))

        mel = librosa.feature.melspectrogram(
            y=y_segment,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.maximum(mel_db, -80)  # ป้องกัน NaN

        segments.append(mel_db)

    return segments

def process_directory(stem_dir, output_dir):
    for song_folder in tqdm(os.listdir(stem_dir), desc=f"Processing {stem_dir}"):
        song_input_path = os.path.join(stem_dir, song_folder)
        if not os.path.isdir(song_input_path):
            continue

        mix_path = os.path.join(song_input_path, "mix", "mix.wav")
        if not os.path.exists(mix_path):
            continue

        for stem_type in ["mix", "vocals", "drums", "bass", "other"]:
            if stem_type == "mix":
                input_path = mix_path
            else:
                input_path = os.path.join(song_input_path, "stems", f"{stem_type}.wav")

            if not os.path.exists(input_path):
                continue

            segments = extract_mel_segments(input_path)

            output_folder = os.path.join(output_dir, song_folder,
                                         "mix" if stem_type == "mix" else f"stems/{stem_type}")
            os.makedirs(output_folder, exist_ok=True)

            for i, mel_segment in enumerate(segments):
                np.save(os.path.join(output_folder, f"{i}.npy"), mel_segment)

            print(f"✅ {song_folder} [{stem_type}] => {len(segments)} segments")

# Run for both train and test
process_directory(STEM_TRAIN_DIR, OUTPUT_SPEC_TRAIN)
process_directory(STEM_TEST_DIR, OUTPUT_SPEC_TEST)
