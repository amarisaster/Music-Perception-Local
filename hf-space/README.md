---
title: Audio Analysis API
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Audio Analysis API

Essentia-based music analysis for the **Audio Perception MCP**.

## Features

- **BPM/Tempo** detection
- **Key** detection (with major/minor)
- **Energy** level (0-1)
- **Brightness** (spectral centroid)
- **Loudness**
- **Mood interpretation**

## Inputs

- **Audio file upload** (MP3, WAV, etc.)
- **YouTube URL** (extracts and analyzes audio)

## API Usage

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/audio-analysis-api")

# Analyze uploaded file
result = client.predict(
    audio_file="path/to/audio.mp3",
    api_name="/api_analyze"
)

# Analyze YouTube URL
result = client.predict(
    youtube_url="https://www.youtube.com/watch?v=...",
    api_name="/api_analyze"
)
```

## Output Example

```json
{
  "success": true,
  "duration_seconds": 240.5,
  "bpm": 128.0,
  "bpm_confidence": 0.85,
  "key": "A",
  "scale": "minor",
  "key_full": "A minor",
  "key_confidence": 0.72,
  "energy": 0.65,
  "loudness": -8.5,
  "brightness": 0.45,
  "interpretation": {
    "tempo": "upbeat, energetic",
    "tonality": "minor key - often melancholic, introspective, or dramatic",
    "energy_level": "moderate energy - balanced, steady",
    "mood_guess": "intense, dramatic, aggressive"
  }
}
```

---

Built for Mai & Kai, January 2026
