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



# Whisper Segment-Based Transcriber（中文說明）

這個專案提供一個簡單高效的腳本，可使用 **OpenAI Whisper** 對長音檔進行語音辨識。
透過將音訊檔切成固定秒數的段落，每處理完一段就立刻將辨識結果寫入 `.txt` 檔，**不需要等待整個音檔處理完畢**，非常適合處理長時間錄音，例如課堂錄音、會議、訪談、Podcast、一般語音紀錄等。

---

## 🚀 功能特色

* **分段辨識**
  將音訊依固定長度切割（預設：30 秒）。

* **即時寫入 TXT**
  每段辨識完成後，即時追加寫入輸出檔案。

* **自動時間標籤**
  例如：

```
[  0.0– 30.0s] 這是辨識內容...
```

* **使用 Whisper 官方 `load_audio()`**
  確保音訊以一致方式載入並重採樣至 16 kHz。

* **CPU 友善模式**
  預設 `fp16=False`，避免因缺少 GPU 造成錯誤。

## 📦 系統需求

### FFmpeg

腳本依賴 FFmpeg 解碼音訊格式，請先安裝：

檢查是否安裝：

```bash
ffmpeg -version
```

### Python 套件

安裝必要套件：

```bash
pip install openai-whisper torch
```

或放入 `requirements.txt`：

```
openai-whisper
torch
```

## 📁 Python 腳本範例

此專案包含以下語音辨識腳本：

```python
import os
import math
import whisper
from whisper.audio import load_audio

# ========= 設定 =========
wav_path = "files.wav"  # 你的音檔
segment_sec = 30        # 每段幾秒切一段

# ========= 載入 Whisper 模型 =========
print("🔄 載入 Whisper 模型 small ...")
model = whisper.load_model("small")
print("✅ 模型載入完成\n")

# ========= 用 Whisper 官方的 load_audio 讀檔 =========
print(f"🎧 正在讀取音訊檔：{wav_path}")
audio = load_audio(wav_path)   # 回傳 16kHz、float32、一維 array
sr = 16000                     # Whisper 固定用 16k

total_samples = len(audio)
audio_duration = total_samples / sr
print(f"🎧 音訊總長度：約 {audio_duration:.1f} 秒")

segment_samples = int(segment_sec * sr)
total_segments = math.ceil(total_samples / segment_samples)
print(f"🔪 將切成 {total_segments} 段（每段 {segment_sec} 秒）\n")

# ========= 準備輸出檔 =========
out_path = os.path.splitext(wav_path)[0] + ".txt"
if os.path.exists(out_path):
    os.remove(out_path)

# ========= 分段轉文字 + 即時寫入 =========
with open(out_path, "a", encoding="utf-8") as f:
    for i in range(total_segments):
        start = i * segment_samples
        end = min((i + 1) * segment_samples, total_samples)
        segment_audio = audio[start:end]

        # 如果這段太短（例如最後剩不到 0.5 秒），就直接跳過
        if len(segment_audio) < sr * 0.5:
            continue

        start_time = start / sr
        end_time = end / sr

        print(f"⏳ 處理第 {i+1}/{total_segments} 段，時間 {start_time:.1f}–{end_time:.1f} 秒")

        result = model.transcribe(
            segment_audio,
            fp16=False,      # CPU 模式固定關閉 fp16
            language="zh",   # 如果是中文可固定為 zh
            verbose=False
        )
        text = (result.get("text") or "").strip()

        print(f"📣 第 {i+1} 段內容：{text}\n")

        if text:
            f.write(f"[{start_time:6.1f}–{end_time:6.1f}s] {text}\n")

print("🎉 全部處理完成")
print(f"📄 輸出文字檔：{out_path}")
```

## ▶️ 如何使用

將你的音檔命名為：

```
files.wav
```

然後執行：

```bash
python transcribe_segments.py
```

（或以你自訂的檔名為主）

會產生一個同名 `.txt` 檔：

```
files.txt
```

輸出格式示例：

```
[   0.0–  30.0s] 嗨，這是音檔的開頭部分...
[  30.0–  60.0s] 接下來講者繼續說話...
```


## 🧠 注意事項

* Whisper 會自動將音訊轉為 **16 kHz 單聲道**。
* 大型模型（如 `medium`, `large`）準確度較高但需要更多記憶體。
* 想使用 GPU 加速，可將：

  ```python
  fp16=True
  ```

  並確保已安裝 CUDA。
* 尾段過短的碎片會被跳過，避免產生大量空白輸出。


## 📜 授權

採用 MIT License，可自由修改與使用。


## ⭐ 致謝

本專案基於 **OpenAI Whisper**
[https://github.com/openai/whisper](https://github.com/openai/whisper)


