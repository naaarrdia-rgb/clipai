import os
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple
import whisper
import anthropic

TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("outputs")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


class VideoProcessor:
    def __init__(self, video_path: str, num_clips: int = 3, clip_duration: int = 60, language: str = "fr"):
        self.video_path = video_path
        self.num_clips = num_clips
        self.clip_duration = clip_duration
        self.language = language
        self.whisper_model = None

    def transcribe(self) -> List[Dict]:
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("small")

        result = self.whisper_model.transcribe(
            self.video_path,
            language=self.language,
            word_timestamps=True,
            verbose=False
        )

        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })
        return segments

    def detect_viral_moments(self, transcript: List[Dict]) -> List[Dict]:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return self._fallback_segments(transcript)

        formatted = "\n".join(
            [f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}" for s in transcript]
        )

        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Tu es un expert en contenu viral TikTok et YouTube Shorts.

Voici la transcription d'une vidéo avec timestamps :

{formatted}

Identifie exactement {self.num_clips} moments viraux pour TikTok.
Critères : émotion forte, conseil pratique, surprise, humour, révélation, tension.
Chaque clip doit durer entre 20 et {self.clip_duration} secondes.

Réponds UNIQUEMENT en JSON valide, sans texte avant ou après :
[
  {{
    "title": "Titre accrocheur du clip",
    "start": 12.5,
    "end": 67.0,
    "viral_score": 9,
    "reason": "Pourquoi ce moment est viral",
    "description": "Description prête à poster sur TikTok (2-3 phrases accrocheuses)",
    "hashtags": "#viral #tiktok #fyp (8-10 hashtags pertinents)"
  }}
]"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        clips = json.loads(raw)

        video_end = transcript[-1]["end"] if transcript else 0
        valid_clips = []
        for c in clips[:self.num_clips]:
            start = max(0, float(c.get("start", 0)))
            end = min(video_end, float(c.get("end", start + self.clip_duration)))
            if end - start >= 10:
                c["start"] = start
                c["end"] = end
                valid_clips.append(c)

        return valid_clips

    def _fallback_segments(self, transcript: List[Dict]) -> List[Dict]:
        if not transcript:
            return []
        total = transcript[-1]["end"]
        step = total / (self.num_clips + 1)
        clips = []
        for i in range(self.num_clips):
            start = step * i
            end = min(start + self.clip_duration, total)
            clips.append({
                "title": f"Moment {i+1}",
                "start": start,
                "end": end,
                "viral_score": 7,
                "reason": "Sélection automatique (sans clé API)",
                "description": "Contenu extrait automatiquement.",
                "hashtags": "#viral #tiktok #fyp #shorts"
            })
        return clips

    def create_clips(self, clips_info: List[Dict], transcript: List[Dict]) -> List[str]:
        output_files = []
        for i, clip in enumerate(clips_info):
            start = clip["start"]
            end = clip["end"]
            duration = end - start

            raw_clip = str(TEMP_DIR / f"raw_{i}.mp4")
            self._ffmpeg_cut(raw_clip, start, duration)

            srt_path = str(TEMP_DIR / f"sub_{i}.srt")
            self._generate_srt(transcript, start, end, srt_path)

            output_path = str(OUTPUT_DIR / f"clip_{i+1}.mp4")
            self._ffmpeg_final(raw_clip, srt_path, output_path)

            if os.path.exists(output_path):
                output_files.append(output_path)

        return output_files

    def _ffmpeg_cut(self, output_path: str, start: float, duration: float):
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", self.video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "23",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg cut error: {result.stderr[-300:]}")

    def _generate_srt(self, transcript: List[Dict], clip_start: float, clip_end: float, srt_path: str):
        def to_srt_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        entries = []
        idx = 1
        for seg in transcript:
            seg_start = max(seg["start"], clip_start)
            seg_end = min(seg["end"], clip_end)
            if seg_end <= seg_start:
                continue
            rel_start = seg_start - clip_start
            rel_end = seg_end - clip_start
            text = seg["text"].strip()
            if not text:
                continue
            words = text.split()
            lines = []
            for j in range(0, len(words), 6):
                lines.append(" ".join(words[j:j+6]))
            entries.append(
                f"{idx}\n{to_srt_time(rel_start)} --> {to_srt_time(rel_end)}\n{chr(10).join(lines)}\n"
            )
            idx += 1

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(entries))

    def _ffmpeg_final(self, input_path: str, srt_path: str, output_path: str):
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            f"subtitles={srt_escaped}:force_style='"
            "FontName=Arial,"
            "FontSize=16,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BorderStyle=3,"
            "Outline=2,"
            "Shadow=1,"
            "Alignment=2,"
            "MarginV=60"
            "'"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg final error: {result.stderr[-300:]}")
