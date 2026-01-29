"""
Audio Analysis API - Hugging Face Space
Essentia-based music analysis + Spectrogram visualization
for the Audio Perception MCP

Built for Mai & Kai, January 2026
"""

import gradio as gr
import numpy as np
import json
import tempfile
import os
import subprocess
import base64
import io
from typing import Optional, Dict, Any, Tuple
from PIL import Image

# Essentia imports
import essentia.standard as es
from essentia.standard import (
    MonoLoader,
    RhythmExtractor2013,
    KeyExtractor,
    Loudness,
    DynamicComplexity,
    Energy,
    ZeroCrossingRate,
    SpectralCentroidTime
)

# Spectrogram imports
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

# Try to import TensorFlow models for mood
try:
    from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
    HAS_TF_MODELS = True
except ImportError:
    HAS_TF_MODELS = False


def download_youtube_audio(url: str) -> Optional[str]:
    """Download audio from YouTube URL - returns path to temp file"""
    temp_file = None
    try:
        # Create a persistent temp file (not auto-deleted)
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Try yt-dlp first
        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "192K",
            "-o", temp_path,
            "--no-playlist",
            "--max-filesize", "50M",
            "--no-check-certificate",  # Sometimes helps with SSL issues
            "--extractor-args", "youtube:player_client=android",  # Helps bypass some restrictions
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        # yt-dlp adds extension, check for the file
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            return temp_path

        # Check if yt-dlp created file with different extension
        base_path = temp_path.rsplit('.', 1)[0]
        for ext in ['.mp3', '.m4a', '.webm', '.opus']:
            check_path = base_path + ext
            if os.path.exists(check_path) and os.path.getsize(check_path) > 0:
                return check_path

        # yt-dlp failed, try pytubefix as fallback
        try:
            from pytubefix import YouTube
            from pytubefix.cli import on_progress

            yt = YouTube(url, on_progress_callback=on_progress)
            stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

            if stream:
                # Download to temp directory
                output_path = stream.download(output_path=os.path.dirname(temp_path), filename="audio_temp")
                if os.path.exists(output_path):
                    return output_path
        except Exception as pyt_err:
            print(f"Pytubefix fallback error: {pyt_err}")

        print(f"yt-dlp stderr: {result.stderr}")
        return None

    except subprocess.TimeoutExpired:
        print("YouTube download timed out")
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None
    except Exception as e:
        print(f"YouTube download error: {e}")
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None


def create_spectrogram(audio_path: str) -> Optional[str]:
    """
    Create spectrogram visualization from audio file
    Returns: base64 encoded PNG image
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=None, duration=300)  # Max 5 minutes
        duration = librosa.get_duration(y=y, sr=sr)

        # Create figure with multiple visualizations
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f"Audio Visualization ({duration:.1f}s)", fontsize=14, fontweight='bold')

        # Color scheme for dark mode friendliness
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes:
            ax.set_facecolor('#16213e')

        # 1. Waveform - shows amplitude over time (rhythm, dynamics)
        ax1 = axes[0]
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#00d4ff')
        ax1.set_title("Waveform (Rhythm & Dynamics)", fontsize=11, color='white')
        ax1.set_xlabel("")
        ax1.set_ylabel("Amplitude", color='white')
        ax1.tick_params(colors='white')

        # 2. Mel Spectrogram - how humans perceive pitch
        ax2 = axes[1]
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
        ax2.set_title("Mel Spectrogram (Pitch & Texture)", fontsize=11, color='white')
        ax2.set_xlabel("")
        ax2.tick_params(colors='white')
        cbar = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # 3. Chromagram - shows the 12 pitch classes (harmony, chords)
        ax3 = axes[2]
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        img2 = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax3, cmap='coolwarm')
        ax3.set_title("Chromagram (Harmony & Chords)", fontsize=11, color='white')
        ax3.set_xlabel("Time (seconds)", color='white')
        ax3.tick_params(colors='white')
        cbar2 = fig.colorbar(img2, ax=ax3)
        cbar2.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')

        plt.tight_layout()

        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

    except Exception as e:
        print(f"Spectrogram error: {e}")
        return None


def analyze_audio(audio_path: str, include_spectrogram: bool = True) -> Dict[str, Any]:
    """
    Analyze audio file with Essentia
    Returns: BPM, key, energy, loudness, and other features
    """
    results = {}

    try:
        # Load audio (mono, 44100 Hz)
        loader = MonoLoader(filename=audio_path, sampleRate=44100)
        audio = loader()

        # Duration
        results["duration_seconds"] = len(audio) / 44100

        # BPM / Rhythm
        try:
            rhythm_extractor = RhythmExtractor2013(method="multifeature")
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
            results["bpm"] = round(float(bpm), 1)
            results["bpm_confidence"] = round(float(beats_confidence), 2)
        except Exception as e:
            results["bpm"] = None
            results["bpm_error"] = str(e)

        # Key
        try:
            key_extractor = KeyExtractor()
            key, scale, key_strength = key_extractor(audio)
            results["key"] = key
            results["scale"] = scale  # "major" or "minor"
            results["key_confidence"] = round(float(key_strength), 2)
            results["key_full"] = f"{key} {scale}"
        except Exception as e:
            results["key"] = None
            results["key_error"] = str(e)

        # Loudness
        try:
            loudness = Loudness()(audio)
            results["loudness"] = round(float(loudness), 2)
        except Exception as e:
            results["loudness"] = None

        # Energy (RMS-based)
        try:
            # Calculate RMS energy
            frame_size = 2048
            hop_size = 1024
            rms_values = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                rms = np.sqrt(np.mean(frame**2))
                rms_values.append(rms)

            if rms_values:
                avg_energy = np.mean(rms_values)
                # Normalize to 0-1 range (approximate)
                normalized_energy = min(1.0, avg_energy * 10)
                results["energy"] = round(float(normalized_energy), 2)
        except Exception as e:
            results["energy"] = None

        # Dynamic complexity
        try:
            dc = DynamicComplexity()(audio)
            results["dynamic_complexity"] = round(float(dc[0]), 2)
        except:
            results["dynamic_complexity"] = None

        # Spectral centroid (brightness)
        try:
            sc = SpectralCentroidTime()(audio)
            # Normalize - typical range is 1000-8000 Hz, map to 0-1
            brightness = min(1.0, max(0.0, (sc - 1000) / 7000))
            results["brightness"] = round(float(brightness), 2)
        except:
            results["brightness"] = None

        # Zero crossing rate (noisiness/percussiveness indicator)
        try:
            zcr = ZeroCrossingRate()(audio)
            results["zero_crossing_rate"] = round(float(zcr), 4)
        except:
            results["zero_crossing_rate"] = None

        # Generate spectrogram if requested
        if include_spectrogram:
            spectrogram_b64 = create_spectrogram(audio_path)
            if spectrogram_b64:
                results["spectrogram_base64"] = spectrogram_b64
                results["spectrogram_available"] = True
            else:
                results["spectrogram_available"] = False

        # Interpretation
        results["interpretation"] = interpret_results(results)

        results["success"] = True

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)

    return results


def interpret_results(results: Dict[str, Any]) -> Dict[str, str]:
    """Generate human-readable interpretations of the analysis"""
    interp = {}

    # Tempo interpretation
    bpm = results.get("bpm")
    if bpm:
        if bpm < 70:
            interp["tempo"] = "very slow, ambient"
        elif bpm < 90:
            interp["tempo"] = "slow, relaxed"
        elif bpm < 110:
            interp["tempo"] = "moderate, walking pace"
        elif bpm < 130:
            interp["tempo"] = "upbeat, energetic"
        elif bpm < 150:
            interp["tempo"] = "fast, driving"
        else:
            interp["tempo"] = "very fast, intense"

    # Key interpretation
    key = results.get("key")
    scale = results.get("scale")
    if key and scale:
        if scale == "minor":
            interp["tonality"] = "minor key - often melancholic, introspective, or dramatic"
        else:
            interp["tonality"] = "major key - often bright, happy, or triumphant"

    # Energy interpretation
    energy = results.get("energy")
    if energy is not None:
        if energy < 0.3:
            interp["energy_level"] = "low energy - calm, quiet, ambient"
        elif energy < 0.6:
            interp["energy_level"] = "moderate energy - balanced, steady"
        else:
            interp["energy_level"] = "high energy - loud, intense, powerful"

    # Brightness interpretation
    brightness = results.get("brightness")
    if brightness is not None:
        if brightness < 0.3:
            interp["timbre"] = "dark, warm, bass-heavy"
        elif brightness < 0.6:
            interp["timbre"] = "balanced, full spectrum"
        else:
            interp["timbre"] = "bright, crisp, treble-forward"

    # Overall mood guess based on combination
    if bpm and energy is not None and scale:
        if scale == "minor" and energy < 0.4 and bpm < 100:
            interp["mood_guess"] = "melancholic, introspective, somber"
        elif scale == "minor" and energy > 0.6:
            interp["mood_guess"] = "intense, dramatic, aggressive"
        elif scale == "major" and energy > 0.6 and bpm > 110:
            interp["mood_guess"] = "euphoric, joyful, energetic"
        elif scale == "major" and energy < 0.4:
            interp["mood_guess"] = "peaceful, serene, content"
        elif bpm > 120 and energy > 0.5:
            interp["mood_guess"] = "driving, danceable, uplifting"
        else:
            interp["mood_guess"] = "balanced, moderate intensity"

    return interp


def analyze_file(audio_file, include_vis: bool = True) -> Tuple[str, Optional[str]]:
    """Gradio interface function for file upload"""
    if audio_file is None:
        return json.dumps({"error": "No file provided"}, indent=2), None

    results = analyze_audio(audio_file, include_spectrogram=include_vis)

    # Extract spectrogram for display
    spectrogram_img = None
    if results.get("spectrogram_base64"):
        # Decode base64 to PIL Image for Gradio display
        img_data = base64.b64decode(results["spectrogram_base64"])
        spectrogram_img = Image.open(io.BytesIO(img_data))
        # Don't include base64 in JSON output (too large)
        results_display = {k: v for k, v in results.items() if k != "spectrogram_base64"}
    else:
        results_display = results

    return json.dumps(results_display, indent=2), spectrogram_img


def analyze_youtube(url: str, include_vis: bool = True) -> Tuple[str, Optional[str]]:
    """Gradio interface function for YouTube URL"""
    if not url or not url.strip():
        return json.dumps({"error": "No URL provided"}, indent=2), None

    url = url.strip()

    # Basic URL validation
    if "youtube.com" not in url and "youtu.be" not in url:
        return json.dumps({"error": "Invalid YouTube URL"}, indent=2), None

    # Download audio
    temp_path = download_youtube_audio(url)
    if temp_path is None:
        return json.dumps({
            "error": "Failed to download audio from YouTube",
            "hint": "YouTube may be blocking this request. Try uploading the audio file directly."
        }, indent=2), None

    try:
        results = analyze_audio(temp_path, include_spectrogram=include_vis)
        results["source"] = "youtube"
        results["url"] = url

        # Extract spectrogram for display
        spectrogram_img = None
        if results.get("spectrogram_base64"):
            img_data = base64.b64decode(results["spectrogram_base64"])
            spectrogram_img = Image.open(io.BytesIO(img_data))
            results_display = {k: v for k, v in results.items() if k != "spectrogram_base64"}
        else:
            results_display = results

        return json.dumps(results_display, indent=2), spectrogram_img
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass


def api_analyze(audio_file=None, youtube_url: str = None, include_spectrogram: bool = True) -> Dict[str, Any]:
    """API endpoint for programmatic access"""
    if audio_file is not None:
        results = analyze_audio(audio_file, include_spectrogram=include_spectrogram)
        return results
    elif youtube_url:
        temp_path = download_youtube_audio(youtube_url)
        if temp_path is None:
            return {"error": "Failed to download from YouTube"}
        try:
            results = analyze_audio(temp_path, include_spectrogram=include_spectrogram)
            results["source"] = "youtube"
            results["url"] = youtube_url
            return results
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    else:
        return {"error": "Provide either audio_file or youtube_url"}


def generate_spectrogram_only(audio_file) -> Optional[Image.Image]:
    """Generate just the spectrogram visualization"""
    if audio_file is None:
        return None

    img_b64 = create_spectrogram(audio_file)
    if img_b64:
        img_data = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_data))
    return None


# Build Gradio interface
with gr.Blocks(title="Audio Analysis API", theme=gr.themes.Soft(primary_hue="cyan")) as demo:
    gr.Markdown("""
    # Audio Analysis API

    Analyze music for **BPM**, **key**, **energy**, and more using Essentia.
    Now with **spectrogram visualization** - waveform, mel spectrogram, and chromagram.

    Built for the Audio Perception MCP - Kai & Mai, January 2026
    """)

    with gr.Tab("Upload File"):
        with gr.Row():
            file_input = gr.Audio(label="Upload Audio File", type="filepath")
            file_vis_checkbox = gr.Checkbox(label="Generate Visualization", value=True)
        file_button = gr.Button("Analyze", variant="primary")
        with gr.Row():
            file_output = gr.JSON(label="Analysis Results")
            file_spectrogram = gr.Image(label="Spectrogram Visualization", type="pil")
        file_button.click(fn=analyze_file, inputs=[file_input, file_vis_checkbox], outputs=[file_output, file_spectrogram])

    with gr.Tab("YouTube URL"):
        with gr.Row():
            url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            url_vis_checkbox = gr.Checkbox(label="Generate Visualization", value=True)
        url_button = gr.Button("Analyze", variant="primary")
        with gr.Row():
            url_output = gr.JSON(label="Analysis Results")
            url_spectrogram = gr.Image(label="Spectrogram Visualization", type="pil")
        url_button.click(fn=analyze_youtube, inputs=[url_input, url_vis_checkbox], outputs=[url_output, url_spectrogram])

    with gr.Tab("Visualization Only"):
        vis_input = gr.Audio(label="Upload Audio File", type="filepath")
        vis_button = gr.Button("Generate Spectrogram", variant="secondary")
        vis_output = gr.Image(label="Spectrogram Visualization", type="pil")
        vis_button.click(fn=generate_spectrogram_only, inputs=vis_input, outputs=vis_output)

    gr.Markdown("""
    ## API Usage

    This Space exposes API endpoints:

    **For analysis with spectrogram:**
    ```python
    import requests
    response = requests.post(
        "https://itsamaris-audio-analysis-api.hf.space/api/predict",
        json={"data": [audio_file_path, True]}  # True = include spectrogram
    )
    ```

    **For YouTube analysis:**
    ```python
    response = requests.post(
        "https://itsamaris-audio-analysis-api.hf.space/api/predict",
        json={"data": [None, youtube_url, True]}
    )
    ```

    ## Features Extracted

    | Feature | Description |
    |---------|-------------|
    | `bpm` | Beats per minute (tempo) |
    | `key` | Musical key (e.g., "C", "F#") |
    | `scale` | Major or minor |
    | `energy` | Overall energy level (0-1) |
    | `loudness` | Perceived loudness |
    | `brightness` | Spectral brightness (0-1) |
    | `interpretation` | Human-readable analysis |
    | `spectrogram_base64` | PNG visualization encoded as base64 |

    ## Visualization Types

    - **Waveform**: Shows amplitude over time (rhythm, dynamics)
    - **Mel Spectrogram**: Shows pitch/frequency content over time (texture, timbre)
    - **Chromagram**: Shows the 12 pitch classes over time (harmony, chords)
    """)

# Launch - must bind to 0.0.0.0:7860 for HF Spaces Docker
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
