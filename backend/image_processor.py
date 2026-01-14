"""
Advanced Image Processing Module
Handles background removal, auto-enhancement, and smart cropping for product images.
"""

import io
import logging
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import requests
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Try to import rembg, but make it optional
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg not available - background removal disabled")


class ImageProcessor:
    """Advanced image processing for product images."""

    @staticmethod
    def load_image_from_url(url: str, timeout: int = 15) -> Optional[Image.Image]:
        """Load an image from URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGBA")
        except Exception as e:
            logger.error(f"Failed to load image from {url}: {e}")
            return None

    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        """
        Remove background from product image using rembg.
        Returns image with transparent background.
        """
        if not REMBG_AVAILABLE:
            logger.warning("Background removal skipped - rembg not installed")
            return image

        try:
            # Convert to bytes for rembg
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Remove background
            result = rembg_remove(img_byte_arr.getvalue())
            return Image.open(io.BytesIO(result)).convert("RGBA")
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            return image

    @staticmethod
    def auto_enhance(
        image: Image.Image,
        contrast: float = 1.15,
        sharpness: float = 1.2,
        brightness: float = 1.05,
        saturation: float = 1.1
    ) -> Image.Image:
        """
        Auto-enhance image with adjustable parameters.

        Args:
            image: Input PIL Image
            contrast: Contrast factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)
            brightness: Brightness factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)

        Returns:
            Enhanced PIL Image
        """
        try:
            # Work with RGB for enhancement, preserve alpha
            has_alpha = image.mode == 'RGBA'
            if has_alpha:
                alpha = image.split()[3]
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image.convert('RGB')

            # Apply enhancements
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(rgb_image)
                rgb_image = enhancer.enhance(contrast)

            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(rgb_image)
                rgb_image = enhancer.enhance(brightness)

            if saturation != 1.0:
                enhancer = ImageEnhance.Color(rgb_image)
                rgb_image = enhancer.enhance(saturation)

            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(rgb_image)
                rgb_image = enhancer.enhance(sharpness)

            # Restore alpha channel if present
            if has_alpha:
                rgb_image = rgb_image.convert('RGBA')
                rgb_image.putalpha(alpha)

            return rgb_image
        except Exception as e:
            logger.error(f"Auto-enhance failed: {e}")
            return image

    @staticmethod
    def smart_crop(
        image: Image.Image,
        target_ratio: float = 1.0,
        padding: float = 0.1
    ) -> Image.Image:
        """
        Smart crop image to target aspect ratio, centering on content.

        Args:
            image: Input PIL Image
            target_ratio: Target width/height ratio (1.0 = square)
            padding: Padding around content (0.1 = 10%)

        Returns:
            Cropped PIL Image
        """
        try:
            # Find bounding box of non-transparent content
            if image.mode == 'RGBA':
                # Get alpha channel
                alpha = np.array(image.split()[3])
                # Find non-transparent pixels
                rows = np.any(alpha > 10, axis=1)
                cols = np.any(alpha > 10, axis=0)

                if not np.any(rows) or not np.any(cols):
                    return image

                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
            else:
                # For non-RGBA, use the full image
                x_min, y_min = 0, 0
                x_max, y_max = image.width - 1, image.height - 1

            # Calculate content dimensions with padding
            content_width = x_max - x_min
            content_height = y_max - y_min

            pad_x = int(content_width * padding)
            pad_y = int(content_height * padding)

            # Expand bounds with padding
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(image.width, x_max + pad_x)
            y_max = min(image.height, y_max + pad_y)

            # Calculate center
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Calculate crop size to match target ratio
            current_width = x_max - x_min
            current_height = y_max - y_min
            current_ratio = current_width / current_height

            if current_ratio > target_ratio:
                # Too wide, adjust height
                new_height = int(current_width / target_ratio)
                new_width = current_width
            else:
                # Too tall, adjust width
                new_width = int(current_height * target_ratio)
                new_height = current_height

            # Calculate crop box centered on content
            left = max(0, center_x - new_width // 2)
            top = max(0, center_y - new_height // 2)
            right = min(image.width, left + new_width)
            bottom = min(image.height, top + new_height)

            # Adjust if crop exceeds image bounds
            if right > image.width:
                left = max(0, image.width - new_width)
                right = image.width
            if bottom > image.height:
                top = max(0, image.height - new_height)
                bottom = image.height

            return image.crop((left, top, right, bottom))
        except Exception as e:
            logger.error(f"Smart crop failed: {e}")
            return image

    @staticmethod
    def add_drop_shadow(
        image: Image.Image,
        offset: Tuple[int, int] = (15, 15),
        blur_radius: int = 20,
        shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 100)
    ) -> Image.Image:
        """
        Add drop shadow to image with transparency.

        Args:
            image: Input PIL Image (RGBA)
            offset: Shadow offset (x, y)
            blur_radius: Shadow blur amount
            shadow_color: Shadow color with alpha

        Returns:
            Image with drop shadow on larger canvas
        """
        try:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            # Create larger canvas for shadow
            padding = blur_radius * 2 + max(abs(offset[0]), abs(offset[1]))
            new_width = image.width + padding * 2
            new_height = image.height + padding * 2

            # Create shadow layer
            shadow = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))

            # Get alpha mask and create shadow
            alpha = image.split()[3]
            shadow_mask = Image.new('RGBA', image.size, shadow_color)
            shadow_mask.putalpha(alpha)

            # Position shadow
            shadow_x = padding + offset[0]
            shadow_y = padding + offset[1]
            shadow.paste(shadow_mask, (shadow_x, shadow_y))

            # Blur shadow
            shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))

            # Paste original image on top
            image_x = padding
            image_y = padding
            shadow.paste(image, (image_x, image_y), image)

            return shadow
        except Exception as e:
            logger.error(f"Drop shadow failed: {e}")
            return image

    @staticmethod
    def resize_for_video(
        image: Image.Image,
        target_size: Tuple[int, int] = (800, 800),
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Resize image for video use while maintaining quality.

        Args:
            image: Input PIL Image
            target_size: Target (width, height)
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Resized PIL Image
        """
        try:
            if maintain_aspect:
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
                return image
            else:
                return image.resize(target_size, Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Resize failed: {e}")
            return image

    @staticmethod
    def create_product_collage(
        images: list,
        layout: str = "grid",
        canvas_size: Tuple[int, int] = (1080, 1080),
        padding: int = 20,
        background_color: Tuple[int, int, int, int] = (255, 255, 255, 0)
    ) -> Image.Image:
        """
        Create a collage of multiple product images.

        Args:
            images: List of PIL Images
            layout: "grid", "horizontal", "vertical", "featured"
            canvas_size: Output canvas size
            padding: Padding between images
            background_color: Background RGBA color

        Returns:
            Collage PIL Image
        """
        try:
            canvas = Image.new('RGBA', canvas_size, background_color)

            if not images:
                return canvas

            num_images = len(images)

            if layout == "horizontal":
                # All images in a row
                img_width = (canvas_size[0] - padding * (num_images + 1)) // num_images
                img_height = canvas_size[1] - padding * 2

                for i, img in enumerate(images):
                    resized = ImageProcessor.resize_for_video(img.copy(), (img_width, img_height))
                    x = padding + i * (img_width + padding)
                    y = (canvas_size[1] - resized.height) // 2
                    canvas.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)

            elif layout == "vertical":
                # All images in a column
                img_width = canvas_size[0] - padding * 2
                img_height = (canvas_size[1] - padding * (num_images + 1)) // num_images

                for i, img in enumerate(images):
                    resized = ImageProcessor.resize_for_video(img.copy(), (img_width, img_height))
                    x = (canvas_size[0] - resized.width) // 2
                    y = padding + i * (img_height + padding)
                    canvas.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)

            elif layout == "featured":
                # One large image with smaller ones below
                if num_images >= 1:
                    # Main image takes 60% height
                    main_height = int((canvas_size[1] - padding * 3) * 0.6)
                    main_width = canvas_size[0] - padding * 2

                    main_img = ImageProcessor.resize_for_video(images[0].copy(), (main_width, main_height))
                    x = (canvas_size[0] - main_img.width) // 2
                    canvas.paste(main_img, (x, padding), main_img if main_img.mode == 'RGBA' else None)

                    # Smaller images below
                    if num_images > 1:
                        remaining = images[1:4]  # Max 3 small images
                        small_height = canvas_size[1] - main_height - padding * 3
                        small_width = (canvas_size[0] - padding * (len(remaining) + 1)) // len(remaining)

                        for i, img in enumerate(remaining):
                            resized = ImageProcessor.resize_for_video(img.copy(), (small_width, small_height))
                            x = padding + i * (small_width + padding)
                            y = main_height + padding * 2
                            canvas.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)

            else:  # grid layout
                # Calculate grid dimensions
                cols = min(2, num_images)
                rows = (num_images + cols - 1) // cols

                img_width = (canvas_size[0] - padding * (cols + 1)) // cols
                img_height = (canvas_size[1] - padding * (rows + 1)) // rows

                for i, img in enumerate(images[:4]):  # Max 4 images in grid
                    row = i // cols
                    col = i % cols

                    resized = ImageProcessor.resize_for_video(img.copy(), (img_width, img_height))
                    x = padding + col * (img_width + padding)
                    y = padding + row * (img_height + padding)
                    canvas.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)

            return canvas
        except Exception as e:
            logger.error(f"Collage creation failed: {e}")
            return Image.new('RGBA', canvas_size, background_color)


def process_product_image(
    image_url: str,
    operations: Dict[str, Any] = None
) -> Optional[Image.Image]:
    """
    Process a product image with specified operations.

    Args:
        image_url: URL of the product image
        operations: Dict of operations to apply:
            - remove_background: bool
            - auto_enhance: bool or dict with params
            - smart_crop: bool or dict with target_ratio
            - add_shadow: bool
            - resize: tuple (width, height)

    Returns:
        Processed PIL Image
    """
    if operations is None:
        operations = {}

    processor = ImageProcessor()

    # Load image
    image = processor.load_image_from_url(image_url)
    if image is None:
        return None

    # Apply operations in order
    if operations.get('remove_background', False):
        image = processor.remove_background(image)

    if operations.get('auto_enhance', False):
        enhance_params = operations.get('auto_enhance')
        if isinstance(enhance_params, dict):
            image = processor.auto_enhance(image, **enhance_params)
        else:
            image = processor.auto_enhance(image)

    if operations.get('smart_crop', False):
        crop_params = operations.get('smart_crop')
        if isinstance(crop_params, dict):
            image = processor.smart_crop(image, **crop_params)
        else:
            image = processor.smart_crop(image)

    if operations.get('add_shadow', False):
        image = processor.add_drop_shadow(image)

    if operations.get('resize'):
        image = processor.resize_for_video(image, operations['resize'])

    return image
