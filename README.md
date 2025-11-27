# Whisper Segment-Based Transcriber

This project provides a simple and efficient script for transcribing long audio files using **OpenAI Whisper** by splitting the audio into fixed-length segments and writing the transcription to a `.txt` file **as soon as each segment is processed**.

The script is designed for users who want real-time transcription output while handling long-duration audio files such as lectures, meetings, interviews, podcasts, and recordings.


## 🚀 Features

- **Segment-based transcription**  
  Splits the audio into equal-length segments (default: 30 seconds).

- **Real-time TXT writing**  
  Each segment is transcribed and appended to a text file immediately.

- **Automatic timestamp labels**  
  Outputs lines such as:  
```

[  0.0– 30.0s] Transcribed text...

````

- **Uses Whisper's official `load_audio()`**  
Ensures consistent loading and resampling to 16 kHz.

- **CPU-friendly**  
`fp16=False` is set by default to ensure compatibility without requiring GPU.

---

## 📦 Requirements

### FFmpeg  
The script depends on FFmpeg for decoding audio formats.

Check installation:
```bash
ffmpeg -version
````

### Python packages

Install dependencies using:

```bash
pip install openai-whisper torch
```

Or put them in `requirements.txt`:

```
openai-whisper
torch
```


## 📁 Example Python Script

This repository includes the following transcription script:

```python
import os
import math
import whisper
from whisper.audio import load_audio

# ========= Settings =========
wav_path = "files.wav"  # Input audio file
segment_sec = 30        # Split audio every N seconds

# ========= Load Whisper model =========
print("🔄 Loading Whisper model: small ...")
model = whisper.load_model("small")
print("✅ Model loaded\n")

# ========= Load audio using Whisper's loader =========
print(f"🎧 Reading audio file: {wav_path}")
audio = load_audio(wav_path)   # Returns 16 kHz float32 1D array
sr = 16000                     # Whisper uses 16k sample rate

total_samples = len(audio)
audio_duration = total_samples / sr
print(f"🎧 Total audio duration: {audio_duration:.1f} sec")

segment_samples = int(segment_sec * sr)
total_segments = math.ceil(total_samples / segment_samples)
print(f"🔪 Will be split into {total_segments} segments ({segment_sec} sec each)\n")

# ========= Prepare output file =========
out_path = os.path.splitext(wav_path)[0] + ".txt"
if os.path.exists(out_path):
    os.remove(out_path)

# ========= Segment transcription with streaming write =========
with open(out_path, "a", encoding="utf-8") as f:
    for i in range(total_segments):
        start = i * segment_samples
        end = min((i + 1) * segment_samples, total_samples)
        segment_audio = audio[start:end]

        # Skip extremely short tail segments (< 0.5 sec)
        if len(segment_audio) < sr * 0.5:
            continue

        start_time = start / sr
        end_time = end / sr

        print(f"⏳ Processing segment {i+1}/{total_segments}, time {start_time:.1f}–{end_time:.1f} sec")

        result = model.transcribe(
            segment_audio,
            fp16=False,      # CPU mode forces fp16 off
            language="zh",   # Set fixed language (Chinese)
            verbose=False
        )
        text = (result.get("text") or "").strip()

        print(f"📣 Segment {i+1} text: {text}\n")

        if text:
            f.write(f"[{start_time:6.1f}–{end_time:6.1f}s] {text}\n")

print("🎉 Finished processing")
print(f"📄 Output text file: {out_path}")
```

---

## ▶️ How to Use

Place your audio file as:

```
files.wav
```

Then run:

```bash
python transcribe_segments.py
```

(Or whatever the script filename is.)

A text file with the same base name will be generated:

```
files.txt
```

Example output:

```
[   0.0–  30.0s] Hello, this is the beginning of the audio...
[  30.0–  60.0s] Here the speaker continues talking...
```

## 🧠 Notes

* Whisper automatically converts audio to 16 kHz mono.
* Larger models (e.g., medium, large) improve accuracy but require more memory.
* For GPU acceleration, change:

  ```python
  fp16=True
  ```

  and ensure CUDA is installed.
* Extremely short tail fragments are skipped to prevent blank output.

## 📜 License

MIT License. Free to modify and use.

## ⭐ Acknowledgments

This project is powered by **OpenAI Whisper**
[https://github.com/openai/whisper](https://github.com/openai/whisper)


