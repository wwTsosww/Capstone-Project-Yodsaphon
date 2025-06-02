import os
import stempeg
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Paths
MUSDB_TRAIN_DIR = "musdb18/train"
MUSDB_TEST_DIR = "musdb18/test"

OUTPUT_STEM_TRAIN = "Data_set/Stem/train"
OUTPUT_STEM_TEST = "Data_set/Stem/test"

# Make output directories
os.makedirs(OUTPUT_STEM_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_STEM_TEST, exist_ok=True)

# Function to process stems
def process_stem(file_path, output_folder):
    # Load stems
    stems, rate = stempeg.read_stems(file_path)

    if stems.shape[0] < 5:
        print(f"⚠️ Not enough stems: {file_path}")
        return

    # Correct stem order according to MUSDB18 format
    mix    = stems[0]
    drums  = stems[1]
    bass   = stems[2]
    other  = stems[3]
    vocals = stems[4]

    # Prepare output paths
    song_name = os.path.splitext(os.path.basename(file_path))[0]
    song_folder = os.path.join(output_folder, song_name)
    stems_folder = os.path.join(song_folder, "stems")
    mix_folder = os.path.join(song_folder, "mix")

    os.makedirs(stems_folder, exist_ok=True)
    os.makedirs(mix_folder, exist_ok=True)

    # Save stems
    sf.write(os.path.join(stems_folder, "vocals.wav"), vocals, rate)
    sf.write(os.path.join(stems_folder, "drums.wav"), drums, rate)
    sf.write(os.path.join(stems_folder, "bass.wav"), bass, rate)
    sf.write(os.path.join(stems_folder, "other.wav"), other, rate)

    # Save mix
    sf.write(os.path.join(mix_folder, "mix.wav"), mix, rate)

# Process train files
train_files = [f for f in os.listdir(MUSDB_TRAIN_DIR) if f.endswith(".stem.mp4")]
for file_name in tqdm(train_files, desc="Processing train stems"):
    file_path = os.path.join(MUSDB_TRAIN_DIR, file_name)
    process_stem(file_path, OUTPUT_STEM_TRAIN)

# Process test files
test_files = [f for f in os.listdir(MUSDB_TEST_DIR) if f.endswith(".stem.mp4")]
for file_name in tqdm(test_files, desc="Processing test stems"):
    file_path = os.path.join(MUSDB_TEST_DIR, file_name)
    process_stem(file_path, OUTPUT_STEM_TEST)

print("✅ Done splitting stems and creating mix.wav!")
