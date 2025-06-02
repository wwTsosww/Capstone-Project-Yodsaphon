
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine

# พาธต่าง ๆ
AUDIO_FOLDER = r"C:\Users\YodsaphonKeddid\Desktop\Separation musical\Object"
MODEL_PATH = r"C:\Users\YodsaphonKeddid\Desktop\Separation musical\outputs\best_model_unet.keras"
OUTPUT_FOLDER = r"C:\Users\YodsaphonKeddid\Desktop\Separation musical\test_outputs"
GRAPH_FOLDER = os.path.join(OUTPUT_FOLDER, "graphs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# โหลดไฟล์เสียง
sr = 44100
ref_tracks = {}
for name in ["vocals", "drums", "bass", "other"]:
    ref_tracks[name], _ = librosa.load(os.path.join(AUDIO_FOLDER, f"{name}.wav"), sr=sr)
mix_audio, _ = librosa.load(os.path.join(AUDIO_FOLDER, "mix.wav"), sr=sr)

# สร้าง mel spectrogram
n_fft = 2048
hop_length = 512
n_mels = 128
fmax = 8000
segment_frames = 864

mel = librosa.feature.melspectrogram(y=mix_audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_db = np.expand_dims(mel_db, axis=(0, -1))

# Padding ให้ครบเพลง
def pad_mel_to_fit(mel_db, segment_frames=864):
    _, _, total_frames, _ = mel_db.shape
    remainder = total_frames % segment_frames
    if remainder != 0:
        pad_width = segment_frames - remainder
        mel_db = np.pad(mel_db, ((0,0), (0,0), (0,pad_width), (0,0)), mode='constant')
    return mel_db

mel_db = pad_mel_to_fit(mel_db, segment_frames)

# แบ่ง segment
def split_mel_segments(mel_db, segment_frames=864):
    _, _, total_frames, _ = mel_db.shape
    segments = []
    for start in range(0, total_frames, segment_frames):
        segment = mel_db[:, :, start:start+segment_frames, :]
        segments.append(segment)
    return segments

mel_segments = split_mel_segments(mel_db)
print(f"✔️ Segments prepared: {len(mel_segments)}")

# โหลดโมเดล
model = tf.keras.models.load_model(MODEL_PATH)
print("✔️ Model loaded")

# แปลง mel -> wave
def mel_to_wave(mel_db_single):
    mel_db_single = np.maximum(mel_db_single, -80)
    S = librosa.db_to_power(mel_db_single)
    return librosa.feature.inverse.mel_to_audio(S, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=fmax)

# Normalize RMS
def match_volume(target_wave, reference_wave):
    target_rms = np.sqrt(np.mean(target_wave**2))
    reference_rms = np.sqrt(np.mean(reference_wave**2))
    if target_rms == 0:
        return target_wave
    gain = reference_rms / target_rms
    return target_wave * gain

# ทำนายทีละ segment แล้วรวม
chunks = {"vocals": [], "drums": [], "bass": [], "other": []}

for segment in mel_segments:
    prediction = model.predict(segment, verbose=0)[0]  # (128, 864, 4)
    chunks["vocals"].append(mel_to_wave(prediction[:, :, 0]))
    chunks["drums"].append(mel_to_wave(prediction[:, :, 1]))
    chunks["bass"].append(mel_to_wave(prediction[:, :, 2]))
    chunks["other"].append(mel_to_wave(prediction[:, :, 3]))

# รวมและ normalize แต่ละ track
pred_tracks = {}
for name in chunks:
    combined = np.concatenate(chunks[name])
    pred_tracks[name] = match_volume(combined, ref_tracks[name])

# บันทึกไฟล์
for name, audio in pred_tracks.items():
    sf.write(os.path.join(OUTPUT_FOLDER, f"{name}.wav"), audio, sr)
    print(f"✔️ Saved {name}.wav")

# เปรียบเทียบและวาดกราฟ
with open(os.path.join(OUTPUT_FOLDER, "similarity_results.txt"), "w") as f:
    for name in ["vocals", "drums", "bass", "other"]:
        ref = ref_tracks[name]
        pred = pred_tracks[name]
        min_len = min(len(ref), len(pred))
        ref = ref[:min_len]
        pred = pred[:min_len]

        # วาดกราฟ
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(ref, sr=sr)
        plt.title(f"Reference {name}")
        plt.subplot(1, 2, 2)
        librosa.display.waveshow(pred, sr=sr)
        plt.title(f"Predicted {name} (Normalized)")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_FOLDER, f"{name}_waveform_comparison.png"))
        plt.close()

        # คำนวณ similarity
        mse = mean_squared_error(ref, pred)
        sim_mse = (1 - mse) * 100
        cos_sim = 1 - cosine(ref, pred)
        sim_cos = cos_sim * 100

        # บันทึกค่า similarity
        f.write(f"{name.capitalize()} Similarity:\n")
        f.write(f"   MSE Similarity    : {sim_mse:.2f}%\n")
        f.write(f"   Cosine Similarity : {sim_cos:.2f}%\n\n")
        print(f"🔎 {name.capitalize()} saved and compared.")
