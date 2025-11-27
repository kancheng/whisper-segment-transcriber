import os
import math
import whisper
from whisper.audio import load_audio

# ========= 設定 =========
wav_path = "files.wav"  # 你的音檔
segment_sec = 30                  # 每段幾秒切一段

# ========= 載入模型 =========
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
            fp16=False,      # 你現在用 CPU，就固定關掉 fp16
            language="zh",   # 如果是中文可以寫死 zh
            verbose=False
        )
        text = (result.get("text") or "").strip()

        # 終端機即時顯示這一段辨識結果
        print(f"📣 第 {i+1} 段內容：{text}\n")

        # 只有有內容才寫入，避免一堆空行
        if text:
            # 也可以在前面加上 [00:00–00:30] 之類時間標籤
            f.write(f"[{start_time:6.1f}–{end_time:6.1f}s] {text}\n")

print("🎉 全部處理完成")
print(f"📄 輸出文字檔：{out_path}")
