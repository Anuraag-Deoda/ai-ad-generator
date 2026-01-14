"""
Platform Variants Module
Handles multi-platform video generation with different aspect ratios and safe zones.
"""

import os
import uuid
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import json

logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """Configuration for a social media platform."""
    name: str
    display_name: str
    aspect_ratio: Tuple[int, int]       # width:height ratio
    resolution: Tuple[int, int]          # actual pixels (width, height)
    safe_zone: Dict[str, int]            # margins in pixels
    max_duration: Optional[int]          # max duration in seconds, None = unlimited
    recommended_duration: int            # recommended duration
    file_format: str = "mp4"
    codec_settings: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.codec_settings:
            self.codec_settings = {"preset": "medium", "crf": "23"}


# Platform configurations
PLATFORM_CONFIGS: Dict[str, PlatformConfig] = {
    "tiktok": PlatformConfig(
        name="tiktok",
        display_name="TikTok",
        aspect_ratio=(9, 16),
        resolution=(1080, 1920),
        safe_zone={"top": 150, "bottom": 280, "left": 40, "right": 40},
        max_duration=180,
        recommended_duration=30,
        codec_settings={"preset": "fast", "crf": "23"}
    ),
    "instagram_feed": PlatformConfig(
        name="instagram_feed",
        display_name="Instagram Feed",
        aspect_ratio=(1, 1),
        resolution=(1080, 1080),
        safe_zone={"top": 60, "bottom": 180, "left": 40, "right": 40},
        max_duration=60,
        recommended_duration=30,
        codec_settings={"preset": "medium", "crf": "23"}
    ),
    "instagram_story": PlatformConfig(
        name="instagram_story",
        display_name="Instagram Story",
        aspect_ratio=(9, 16),
        resolution=(1080, 1920),
        safe_zone={"top": 180, "bottom": 220, "left": 40, "right": 40},
        max_duration=60,
        recommended_duration=15,
        codec_settings={"preset": "fast", "crf": "23"}
    ),
    "instagram_reels": PlatformConfig(
        name="instagram_reels",
        display_name="Instagram Reels",
        aspect_ratio=(9, 16),
        resolution=(1080, 1920),
        safe_zone={"top": 150, "bottom": 300, "left": 40, "right": 40},
        max_duration=90,
        recommended_duration=30,
        codec_settings={"preset": "fast", "crf": "23"}
    ),
    "youtube_short": PlatformConfig(
        name="youtube_short",
        display_name="YouTube Short",
        aspect_ratio=(9, 16),
        resolution=(1080, 1920),
        safe_zone={"top": 100, "bottom": 180, "left": 40, "right": 40},
        max_duration=60,
        recommended_duration=30,
        codec_settings={"preset": "medium", "crf": "21"}
    ),
    "youtube_standard": PlatformConfig(
        name="youtube_standard",
        display_name="YouTube Standard",
        aspect_ratio=(16, 9),
        resolution=(1920, 1080),
        safe_zone={"top": 80, "bottom": 120, "left": 100, "right": 100},
        max_duration=None,
        recommended_duration=30,
        codec_settings={"preset": "medium", "crf": "20"}
    ),
    "facebook_feed": PlatformConfig(
        name="facebook_feed",
        display_name="Facebook Feed",
        aspect_ratio=(4, 5),
        resolution=(1080, 1350),
        safe_zone={"top": 60, "bottom": 150, "left": 40, "right": 40},
        max_duration=240,
        recommended_duration=30,
        codec_settings={"preset": "medium", "crf": "23"}
    ),
    "facebook_story": PlatformConfig(
        name="facebook_story",
        display_name="Facebook Story",
        aspect_ratio=(9, 16),
        resolution=(1080, 1920),
        safe_zone={"top": 150, "bottom": 200, "left": 40, "right": 40},
        max_duration=20,
        recommended_duration=15,
        codec_settings={"preset": "fast", "crf": "23"}
    ),
    "linkedin": PlatformConfig(
        name="linkedin",
        display_name="LinkedIn",
        aspect_ratio=(1, 1),
        resolution=(1080, 1080),
        safe_zone={"top": 40, "bottom": 100, "left": 40, "right": 40},
        max_duration=600,
        recommended_duration=30,
        codec_settings={"preset": "medium", "crf": "23"}
    ),
    "twitter": PlatformConfig(
        name="twitter",
        display_name="Twitter/X",
        aspect_ratio=(16, 9),
        resolution=(1280, 720),
        safe_zone={"top": 40, "bottom": 80, "left": 60, "right": 60},
        max_duration=140,
        recommended_duration=30,
        codec_settings={"preset": "fast", "crf": "24"}
    )
}


def get_platform_config(platform: str) -> Optional[PlatformConfig]:
    """Get configuration for a platform."""
    return PLATFORM_CONFIGS.get(platform)


def get_all_platforms() -> List[Dict[str, Any]]:
    """Get list of all supported platforms with their info."""
    return [
        {
            "id": config.name,
            "name": config.display_name,
            "aspect_ratio": f"{config.aspect_ratio[0]}:{config.aspect_ratio[1]}",
            "resolution": f"{config.resolution[0]}x{config.resolution[1]}",
            "max_duration": config.max_duration,
            "recommended_duration": config.recommended_duration
        }
        for config in PLATFORM_CONFIGS.values()
    ]


class VariantGenerator:
    """Generates video variants for different platforms."""

    def __init__(self, base_video_path: str, output_dir: str = None):
        """
        Initialize variant generator.

        Args:
            base_video_path: Path to the base video (typically 9:16 portrait)
            output_dir: Directory for output files
        """
        self.base_video_path = base_video_path
        self.output_dir = output_dir or os.path.dirname(base_video_path)

        # Get base video info
        self.base_info = self._get_video_info(base_video_path)

    def _get_video_info(self, video_path: str) -> Dict:
        """Get video metadata using FFprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        return {
                            'width': int(stream.get('width', 1080)),
                            'height': int(stream.get('height', 1920)),
                            'duration': float(stream.get('duration', 30))
                        }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")

        return {'width': 1080, 'height': 1920, 'duration': 30}

    def _calculate_crop_params(
        self,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop parameters to achieve target aspect ratio.

        Returns:
            (crop_width, crop_height, x_offset, y_offset)
        """
        source_ratio = source_width / source_height
        target_ratio = target_width / target_height

        if source_ratio > target_ratio:
            # Source is wider, crop sides
            crop_height = source_height
            crop_width = int(source_height * target_ratio)
            x_offset = (source_width - crop_width) // 2
            y_offset = 0
        else:
            # Source is taller, crop top/bottom
            crop_width = source_width
            crop_height = int(source_width / target_ratio)
            x_offset = 0
            y_offset = (source_height - crop_height) // 2

        return crop_width, crop_height, x_offset, y_offset

    def generate_variant(
        self,
        platform: str,
        output_filename: str = None
    ) -> Optional[str]:
        """
        Generate a video variant for a specific platform.

        Args:
            platform: Platform ID (e.g., "tiktok", "instagram_feed")
            output_filename: Optional output filename

        Returns:
            Path to generated variant or None if failed
        """
        config = PLATFORM_CONFIGS.get(platform)
        if not config:
            logger.error(f"Unknown platform: {platform}")
            return None

        try:
            # Generate output path
            if output_filename is None:
                base_name = os.path.splitext(os.path.basename(self.base_video_path))[0]
                output_filename = f"{base_name}_{platform}.mp4"

            output_path = os.path.join(self.output_dir, output_filename)

            # Calculate transformation
            source_w = self.base_info['width']
            source_h = self.base_info['height']
            target_w, target_h = config.resolution

            # Calculate crop to match target aspect ratio
            crop_w, crop_h, crop_x, crop_y = self._calculate_crop_params(
                source_w, source_h, target_w, target_h
            )

            # Build FFmpeg command
            filter_complex = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_w}:{target_h}"

            cmd = [
                'ffmpeg', '-y',
                '-i', self.base_video_path,
                '-vf', filter_complex,
                '-c:v', 'libx264',
                '-preset', config.codec_settings.get('preset', 'medium'),
                '-crf', config.codec_settings.get('crf', '23'),
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            logger.info(f"Generating {platform} variant: {target_w}x{target_h}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info(f"Generated variant: {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg error for {platform}: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate {platform} variant: {e}")
            return None

    def batch_generate_variants(
        self,
        platforms: List[str],
        max_workers: int = 3
    ) -> Dict[str, Optional[str]]:
        """
        Generate variants for multiple platforms in parallel.

        Args:
            platforms: List of platform IDs
            max_workers: Max parallel workers

        Returns:
            Dict of platform -> output_path (or None if failed)
        """
        results = {}

        # Generate sequentially to avoid overwhelming system
        # (video encoding is CPU-intensive)
        for platform in platforms:
            results[platform] = self.generate_variant(platform)

        return results


@dataclass
class VariantBatch:
    """Tracks a batch of variant generations."""
    batch_id: str
    job_id: str
    platforms: List[str]
    status: str = "pending"  # pending, processing, completed, failed
    results: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "job_id": self.job_id,
            "platforms": self.platforms,
            "status": self.status,
            "results": self.results,
            "errors": self.errors
        }


def generate_platform_variants(
    base_video_path: str,
    job_id: str,
    platforms: List[str],
    output_dir: str = None
) -> VariantBatch:
    """
    Generate video variants for multiple platforms.

    Args:
        base_video_path: Path to base video
        job_id: Job identifier
        platforms: List of platform IDs to generate
        output_dir: Output directory

    Returns:
        VariantBatch with results
    """
    batch = VariantBatch(
        batch_id=f"batch_{uuid.uuid4().hex[:8]}",
        job_id=job_id,
        platforms=platforms,
        status="processing"
    )

    try:
        generator = VariantGenerator(base_video_path, output_dir)
        results = generator.batch_generate_variants(platforms)

        for platform, path in results.items():
            if path:
                batch.results[platform] = path
            else:
                batch.errors[platform] = "Generation failed"

        batch.status = "completed" if batch.results else "failed"

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        batch.status = "failed"
        batch.errors["general"] = str(e)

    return batch


def get_safe_text_area(
    platform: str,
    frame_width: int,
    frame_height: int
) -> Dict[str, int]:
    """
    Get safe area for text placement on a platform.

    Args:
        platform: Platform ID
        frame_width: Frame width
        frame_height: Frame height

    Returns:
        Dict with x, y, width, height of safe area
    """
    config = PLATFORM_CONFIGS.get(platform)
    if not config:
        return {"x": 40, "y": 40, "width": frame_width - 80, "height": frame_height - 80}

    safe = config.safe_zone
    return {
        "x": safe["left"],
        "y": safe["top"],
        "width": frame_width - safe["left"] - safe["right"],
        "height": frame_height - safe["top"] - safe["bottom"]
    }
