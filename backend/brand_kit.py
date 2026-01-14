"""
Brand Kit Module
Handles brand customization including logos, colors, and watermarks.
"""

import io
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
import requests
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BrandKit:
    """Brand customization settings."""
    logo_url: Optional[str] = None
    primary_color: str = "#0066FF"      # Main brand color
    secondary_color: str = "#1A1A1A"    # Secondary color
    accent_color: str = "#FFD700"       # Accent/highlight color
    logo_position: str = "bottom_right"  # top_left, top_right, bottom_left, bottom_right
    logo_size: str = "medium"           # small, medium, large
    watermark_opacity: float = 0.8      # 0.0 - 1.0
    apply_to_gradients: bool = True
    apply_to_text: bool = True

    def __post_init__(self):
        # Validate hex colors
        for color_name in ['primary_color', 'secondary_color', 'accent_color']:
            color = getattr(self, color_name)
            if not BrandKitManager.validate_hex_color(color):
                logger.warning(f"Invalid {color_name}: {color}, using default")


class BrandKitManager:
    """Manages brand kit operations."""

    # Logo size presets (relative to frame)
    LOGO_SIZES = {
        "small": 0.08,    # 8% of frame width
        "medium": 0.12,   # 12% of frame width
        "large": 0.18     # 18% of frame width
    }

    # Position offsets (as percentage of frame)
    POSITION_OFFSETS = {
        "top_left": (0.03, 0.03),
        "top_right": (0.97, 0.03),
        "bottom_left": (0.03, 0.97),
        "bottom_right": (0.97, 0.97),
        "center": (0.5, 0.5)
    }

    @staticmethod
    def validate_hex_color(color: str) -> bool:
        """Validate hex color format."""
        if not color:
            return False
        pattern = r'^#?([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
        return bool(re.match(pattern, color))

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
        """Convert hex color to RGBA tuple."""
        rgb = BrandKitManager.hex_to_rgb(hex_color)
        return (*rgb, alpha)

    @staticmethod
    def load_logo(url: str, timeout: int = 15) -> Optional[Image.Image]:
        """Load logo from URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            logo = Image.open(io.BytesIO(response.content))

            # Convert to RGBA for transparency support
            if logo.mode != 'RGBA':
                logo = logo.convert('RGBA')

            return logo
        except Exception as e:
            logger.error(f"Failed to load logo from {url}: {e}")
            return None

    @staticmethod
    def resize_logo(
        logo: Image.Image,
        frame_width: int,
        size: str = "medium"
    ) -> Image.Image:
        """Resize logo based on frame size and size preset."""
        scale = BrandKitManager.LOGO_SIZES.get(size, 0.12)
        target_width = int(frame_width * scale)

        # Maintain aspect ratio
        aspect = logo.height / logo.width
        target_height = int(target_width * aspect)

        return logo.resize((target_width, target_height), Image.Resampling.LANCZOS)

    @staticmethod
    def position_logo(
        frame: Image.Image,
        logo: Image.Image,
        position: str = "bottom_right",
        opacity: float = 0.8
    ) -> Image.Image:
        """Position logo on frame with opacity."""
        try:
            # Get position offset
            offset = BrandKitManager.POSITION_OFFSETS.get(position, (0.97, 0.97))

            # Calculate position
            x = int(frame.width * offset[0])
            y = int(frame.height * offset[1])

            # Adjust for logo size based on position
            if "right" in position:
                x -= logo.width
            elif "center" in position.lower() or offset[0] == 0.5:
                x -= logo.width // 2

            if "bottom" in position:
                y -= logo.height
            elif "center" in position.lower() or offset[1] == 0.5:
                y -= logo.height // 2

            # Apply opacity to logo
            if opacity < 1.0:
                logo = logo.copy()
                alpha = logo.split()[3]
                alpha = alpha.point(lambda p: int(p * opacity))
                logo.putalpha(alpha)

            # Ensure frame is RGBA
            if frame.mode != 'RGBA':
                frame = frame.convert('RGBA')

            # Create a copy and paste logo
            result = frame.copy()
            result.paste(logo, (x, y), logo)

            return result
        except Exception as e:
            logger.error(f"Logo positioning failed: {e}")
            return frame

    @staticmethod
    def create_branded_gradient(
        width: int,
        height: int,
        primary_color: str,
        secondary_color: str,
        direction: str = "diagonal"
    ) -> Image.Image:
        """
        Create gradient background using brand colors.

        Args:
            width: Image width
            height: Image height
            primary_color: Starting hex color
            secondary_color: Ending hex color
            direction: "horizontal", "vertical", "diagonal", "radial"

        Returns:
            Gradient PIL Image
        """
        try:
            primary_rgb = BrandKitManager.hex_to_rgb(primary_color)
            secondary_rgb = BrandKitManager.hex_to_rgb(secondary_color)

            # Create gradient array
            gradient = np.zeros((height, width, 3), dtype=np.uint8)

            if direction == "horizontal":
                for x in range(width):
                    ratio = x / width
                    gradient[:, x] = [
                        int(primary_rgb[i] + (secondary_rgb[i] - primary_rgb[i]) * ratio)
                        for i in range(3)
                    ]

            elif direction == "vertical":
                for y in range(height):
                    ratio = y / height
                    gradient[y, :] = [
                        int(primary_rgb[i] + (secondary_rgb[i] - primary_rgb[i]) * ratio)
                        for i in range(3)
                    ]

            elif direction == "diagonal":
                for y in range(height):
                    for x in range(width):
                        ratio = (x + y) / (width + height)
                        gradient[y, x] = [
                            int(primary_rgb[i] + (secondary_rgb[i] - primary_rgb[i]) * ratio)
                            for i in range(3)
                        ]

            elif direction == "radial":
                center_x, center_y = width // 2, height // 2
                max_dist = np.sqrt(center_x**2 + center_y**2)

                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        ratio = min(1.0, dist / max_dist)
                        gradient[y, x] = [
                            int(primary_rgb[i] + (secondary_rgb[i] - primary_rgb[i]) * ratio)
                            for i in range(3)
                        ]

            return Image.fromarray(gradient, 'RGB')
        except Exception as e:
            logger.error(f"Gradient creation failed: {e}")
            return Image.new('RGB', (width, height), primary_rgb)

    @staticmethod
    def apply_brand_colors_to_accents(
        frame: Image.Image,
        accent_color: str,
        accent_areas: List[Tuple[int, int, int, int]] = None
    ) -> Image.Image:
        """
        Apply brand accent color to specified areas or auto-detected highlights.

        Args:
            frame: Input frame
            accent_color: Brand accent hex color
            accent_areas: List of (x, y, width, height) areas to colorize

        Returns:
            Frame with branded accents
        """
        # This is a placeholder for more sophisticated accent application
        # In production, this could detect UI elements and apply brand colors
        return frame

    @staticmethod
    def create_watermark_text(
        text: str,
        width: int,
        font_size: int = 24,
        color: str = "#FFFFFF",
        opacity: float = 0.3
    ) -> Image.Image:
        """Create text watermark."""
        try:
            # Create transparent image
            height = font_size + 20
            watermark = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark)

            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Get text size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]

            # Calculate position (centered)
            x = (width - text_width) // 2
            y = 10

            # Draw text with opacity
            rgba_color = BrandKitManager.hex_to_rgba(color, int(255 * opacity))
            draw.text((x, y), text, font=font, fill=rgba_color)

            return watermark
        except Exception as e:
            logger.error(f"Watermark creation failed: {e}")
            return Image.new('RGBA', (width, 50), (0, 0, 0, 0))


def apply_brand_kit_to_frame(
    frame: Image.Image,
    brand_kit: BrandKit,
    logo_cache: dict = None
) -> Image.Image:
    """
    Apply brand kit to a video frame.

    Args:
        frame: Input frame
        brand_kit: Brand kit settings
        logo_cache: Optional cache dict for loaded logo

    Returns:
        Branded frame
    """
    manager = BrandKitManager()

    # Apply logo watermark if URL provided
    if brand_kit.logo_url:
        # Check cache first
        logo = None
        if logo_cache is not None and 'logo' in logo_cache:
            logo = logo_cache['logo']
        else:
            logo = manager.load_logo(brand_kit.logo_url)
            if logo_cache is not None:
                logo_cache['logo'] = logo

        if logo:
            # Resize logo
            logo = manager.resize_logo(logo, frame.width, brand_kit.logo_size)

            # Position and apply logo
            frame = manager.position_logo(
                frame,
                logo,
                brand_kit.logo_position,
                brand_kit.watermark_opacity
            )

    return frame


def get_brand_gradient_colors(brand_kit: BrandKit) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Get gradient colors from brand kit for use in video generator.

    Returns:
        Tuple of (start_color, end_color) as RGB tuples
    """
    manager = BrandKitManager()
    start_color = manager.hex_to_rgb(brand_kit.primary_color)
    end_color = manager.hex_to_rgb(brand_kit.secondary_color)
    return start_color, end_color


def get_brand_accent_color(brand_kit: BrandKit) -> Tuple[int, int, int]:
    """Get accent color from brand kit as RGB tuple."""
    return BrandKitManager.hex_to_rgb(brand_kit.accent_color)
