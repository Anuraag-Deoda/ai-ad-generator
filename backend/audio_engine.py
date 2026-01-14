"""
Audio Engine Module
Handles music selection, beat detection, sound effects, and audio mixing.
"""

import os
import json
import random
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Try to import audio processing libraries
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - beat detection disabled")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available - audio mixing disabled")


@dataclass
class AudioSettings:
    """Audio configuration for video generation."""
    music_style: str = "energetic"       # energetic, professional, casual, luxury
    music_track: Optional[str] = None    # Specific track name, or None for auto-select
    sfx_enabled: bool = True
    beat_sync: bool = True
    music_volume: float = 0.7            # 0.0 - 1.0
    sfx_volume: float = 0.8              # 0.0 - 1.0
    fade_in: float = 1.0                 # Seconds
    fade_out: float = 2.0                # Seconds


@dataclass
class SFXEvent:
    """Sound effect event."""
    time: float          # Time in seconds
    sfx_type: str        # Type: transition, accent, impact
    sfx_name: str        # Specific sound name
    volume: float = 1.0  # Volume multiplier


class AudioEngine:
    """Audio processing engine for video generation."""

    def __init__(self, assets_path: str = None):
        if assets_path is None:
            assets_path = os.path.join(os.path.dirname(__file__), "assets", "audio")

        self.assets_path = Path(assets_path)
        self.metadata = self._load_metadata()
        self.music_tracks = {}
        self.sfx_library = {}
        self._index_assets()

    def _load_metadata(self) -> Dict:
        """Load audio assets metadata."""
        metadata_path = self.assets_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {"music": {}, "sfx": {}, "style_defaults": {}}

    def _index_assets(self):
        """Index available audio assets."""
        # Index music tracks
        music_path = self.assets_path / "music"
        if music_path.exists():
            for style_dir in music_path.iterdir():
                if style_dir.is_dir():
                    style = style_dir.name
                    self.music_tracks[style] = []
                    for audio_file in style_dir.glob("*.mp3"):
                        self.music_tracks[style].append(str(audio_file))
                    for audio_file in style_dir.glob("*.wav"):
                        self.music_tracks[style].append(str(audio_file))

        # Index SFX
        sfx_path = self.assets_path / "sfx"
        if sfx_path.exists():
            for sfx_file in sfx_path.glob("*.wav"):
                # Extract type from filename (e.g., whoosh_01.wav -> whoosh)
                sfx_name = sfx_file.stem.rsplit('_', 1)[0]
                if sfx_name not in self.sfx_library:
                    self.sfx_library[sfx_name] = []
                self.sfx_library[sfx_name].append(str(sfx_file))

    def get_available_tracks(self, style: str = None) -> List[Dict]:
        """Get list of available music tracks."""
        tracks = []

        if style and style in self.metadata.get("music", {}):
            tracks = self.metadata["music"][style]
        else:
            for style_tracks in self.metadata.get("music", {}).values():
                tracks.extend(style_tracks)

        return tracks

    def get_available_sfx(self) -> Dict[str, List[Dict]]:
        """Get available sound effects by category."""
        return self.metadata.get("sfx", {})

    def select_music_for_style(
        self,
        style: str,
        duration: int = 30,
        specific_track: str = None
    ) -> Optional[str]:
        """
        Select appropriate music track for video style.

        Args:
            style: Video style (energetic, professional, casual, luxury)
            duration: Video duration in seconds
            specific_track: Specific track filename to use

        Returns:
            Path to selected music file
        """
        if specific_track:
            # Look for specific track
            track_path = self.assets_path / "music" / style / specific_track
            if track_path.exists():
                return str(track_path)

        # Get tracks for style
        style_tracks = self.music_tracks.get(style, [])
        if not style_tracks:
            # Fallback to any available tracks
            for tracks in self.music_tracks.values():
                style_tracks.extend(tracks)

        if style_tracks:
            return random.choice(style_tracks)

        return None

    def analyze_beats(self, audio_path: str) -> List[float]:
        """
        Analyze audio file and return beat timestamps.

        Args:
            audio_path: Path to audio file

        Returns:
            List of beat times in seconds
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available for beat detection")
            return []

        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050)

            # Detect tempo and beats
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            logger.info(f"Detected {len(beat_times)} beats at {tempo:.1f} BPM")
            return beat_times.tolist()
        except Exception as e:
            logger.error(f"Beat detection failed: {e}")
            return []

    def get_beat_synced_events(
        self,
        beat_times: List[float],
        scene_boundaries: List[float],
        video_duration: float
    ) -> Dict[str, List[float]]:
        """
        Generate beat-synced event times for animations.

        Args:
            beat_times: List of beat timestamps
            scene_boundaries: List of scene change times
            video_duration: Total video duration

        Returns:
            Dict with event types and their timestamps
        """
        events = {
            "particle_bursts": [],
            "text_reveals": [],
            "transitions": [],
            "accents": []
        }

        if not beat_times:
            return events

        # Find beats near scene boundaries for transitions
        for boundary in scene_boundaries:
            closest_beat = min(beat_times, key=lambda x: abs(x - boundary))
            if abs(closest_beat - boundary) < 0.5:  # Within 0.5 seconds
                events["transitions"].append(closest_beat)

        # Select strong beats for particle bursts (every 4th beat typically)
        for i, beat in enumerate(beat_times):
            if i % 4 == 0 and beat < video_duration:
                events["particle_bursts"].append(beat)
            if i % 8 == 0 and beat < video_duration:
                events["accents"].append(beat)

        return events

    def select_sfx_for_scene(self, scene_type: str) -> Optional[str]:
        """Select appropriate SFX for scene type."""
        sfx_mapping = {
            "hook": ["impact", "whoosh"],
            "pitch": ["swoosh", "ding"],
            "features": ["click", "pop"],
            "cta": ["impact", "boom", "ding"]
        }

        sfx_types = sfx_mapping.get(scene_type, ["whoosh"])

        for sfx_type in sfx_types:
            if sfx_type in self.sfx_library and self.sfx_library[sfx_type]:
                return random.choice(self.sfx_library[sfx_type])

        return None

    def create_sfx_events(
        self,
        scene_times: Dict[str, Tuple[float, float]],
        beat_times: List[float] = None
    ) -> List[SFXEvent]:
        """
        Create SFX events for video scenes.

        Args:
            scene_times: Dict of scene_name -> (start_time, end_time)
            beat_times: Optional beat timestamps for syncing

        Returns:
            List of SFXEvent objects
        """
        events = []

        for scene_name, (start, end) in scene_times.items():
            # Add transition SFX at scene start
            sfx_path = self.select_sfx_for_scene(scene_name)
            if sfx_path:
                event_time = start
                # Sync to nearest beat if available
                if beat_times:
                    closest_beat = min(beat_times, key=lambda x: abs(x - start))
                    if abs(closest_beat - start) < 0.3:
                        event_time = closest_beat

                events.append(SFXEvent(
                    time=event_time,
                    sfx_type="transition",
                    sfx_name=os.path.basename(sfx_path),
                    volume=0.8
                ))

        return events

    def mix_audio(
        self,
        music_path: str,
        sfx_events: List[SFXEvent],
        output_path: str,
        duration: float,
        settings: AudioSettings
    ) -> Optional[str]:
        """
        Mix background music with sound effects.

        Args:
            music_path: Path to background music
            sfx_events: List of SFX events to overlay
            output_path: Output file path
            duration: Target duration in seconds
            settings: Audio settings

        Returns:
            Path to mixed audio file
        """
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available for audio mixing")
            return music_path

        try:
            # Load and prepare background music
            if music_path and os.path.exists(music_path):
                music = AudioSegment.from_file(music_path)

                # Adjust to target duration
                if len(music) < duration * 1000:
                    # Loop music if too short
                    loops_needed = int((duration * 1000) / len(music)) + 1
                    music = music * loops_needed

                music = music[:int(duration * 1000)]

                # Apply volume
                music = music - (20 * (1 - settings.music_volume))  # dB adjustment

                # Apply fade in/out
                if settings.fade_in > 0:
                    music = music.fade_in(int(settings.fade_in * 1000))
                if settings.fade_out > 0:
                    music = music.fade_out(int(settings.fade_out * 1000))
            else:
                # Create silent audio if no music
                music = AudioSegment.silent(duration=int(duration * 1000))

            # Overlay SFX events
            if settings.sfx_enabled:
                for event in sfx_events:
                    sfx_path = self.assets_path / "sfx" / event.sfx_name
                    if sfx_path.exists():
                        sfx = AudioSegment.from_file(str(sfx_path))
                        sfx = sfx - (20 * (1 - settings.sfx_volume * event.volume))

                        position = int(event.time * 1000)
                        if position < len(music):
                            music = music.overlay(sfx, position=position)

            # Export mixed audio
            music.export(output_path, format="mp3")
            logger.info(f"Audio mixed and saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            return music_path

    def apply_audio_to_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> Optional[str]:
        """
        Apply audio track to video using FFmpeg.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Output video path

        Returns:
            Path to output video with audio
        """
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                logger.info(f"Audio applied to video: {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return video_path

        except Exception as e:
            logger.error(f"Failed to apply audio to video: {e}")
            return video_path


def generate_video_with_audio(
    video_path: str,
    job_data: Dict[str, Any],
    audio_settings: AudioSettings = None
) -> str:
    """
    Add audio to generated video.

    Args:
        video_path: Path to video without audio
        job_data: Job data with style and duration info
        audio_settings: Audio configuration

    Returns:
        Path to video with audio
    """
    if audio_settings is None:
        audio_settings = AudioSettings()

    engine = AudioEngine()

    # Get video info
    style = job_data.get('style', 'energetic')
    duration = job_data.get('duration', 30)

    # Select music
    music_path = engine.select_music_for_style(
        style,
        duration,
        audio_settings.music_track
    )

    if not music_path:
        logger.warning("No music track available, returning video without audio")
        return video_path

    # Analyze beats for sync
    beat_times = []
    if audio_settings.beat_sync:
        beat_times = engine.analyze_beats(music_path)

    # Calculate scene times
    scene_times = {
        "hook": (0, duration * 0.2),
        "pitch": (duration * 0.2, duration * 0.5),
        "features": (duration * 0.5, duration * 0.8),
        "cta": (duration * 0.8, duration)
    }

    # Create SFX events
    sfx_events = engine.create_sfx_events(scene_times, beat_times)

    # Mix audio
    audio_output = video_path.replace('.mp4', '_audio.mp3')
    mixed_audio = engine.mix_audio(
        music_path,
        sfx_events,
        audio_output,
        duration,
        audio_settings
    )

    # Apply audio to video
    final_output = video_path.replace('.mp4', '_final.mp4')
    result = engine.apply_audio_to_video(video_path, mixed_audio, final_output)

    # Cleanup temp audio file
    if os.path.exists(audio_output):
        try:
            os.remove(audio_output)
        except:
            pass

    return result if result else video_path
