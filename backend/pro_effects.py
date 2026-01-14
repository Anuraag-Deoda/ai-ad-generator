"""
PRO Video Effects - Advanced Cinematic Effects
Next-level animations, 3D transforms, and professional motion graphics
"""

import math
import random
import colorsys
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageChops, ImageOps


# ============================================================================
# 3D PERSPECTIVE TRANSFORMS
# ============================================================================

class Transform3D:
    """3D perspective transformations for images"""

    @staticmethod
    def perspective_transform(image: Image.Image,
                             coefficients: Tuple[float, ...]) -> Image.Image:
        """Apply perspective transform using coefficients"""
        return image.transform(image.size, Image.Transform.PERSPECTIVE,
                              coefficients, Image.Resampling.BICUBIC)

    @staticmethod
    def rotate_y(image: Image.Image, angle: float,
                 target_size: Tuple[int, int] = None) -> Image.Image:
        """Simulate Y-axis rotation (left-right tilt)"""
        width, height = image.size
        target_size = target_size or (width, height)

        # Convert angle to radians
        rad = math.radians(angle)

        # Calculate perspective coefficients
        # This simulates rotating around the Y axis
        factor = math.cos(rad)
        skew = math.sin(rad) * 0.5

        if abs(angle) < 1:
            return image.resize(target_size, Image.Resampling.LANCZOS)

        # Calculate corner positions after rotation
        # Original corners: (0,0), (w,0), (w,h), (0,h)
        if angle > 0:
            # Rotate right - left side comes forward
            coeffs = Transform3D._find_coeffs(
                [(0, 0), (width, 0), (width, height), (0, height)],
                [(width * skew, height * 0.1),
                 (width * (1 - skew * 0.5), 0),
                 (width * (1 - skew * 0.5), height),
                 (width * skew, height * 0.9)]
            )
        else:
            # Rotate left - right side comes forward
            coeffs = Transform3D._find_coeffs(
                [(0, 0), (width, 0), (width, height), (0, height)],
                [(width * abs(skew) * 0.5, 0),
                 (width * (1 - abs(skew)), height * 0.1),
                 (width * (1 - abs(skew)), height * 0.9),
                 (width * abs(skew) * 0.5, height)]
            )

        result = image.transform(image.size, Image.Transform.PERSPECTIVE,
                                coeffs, Image.Resampling.BICUBIC)
        return result.resize(target_size, Image.Resampling.LANCZOS)

    @staticmethod
    def rotate_x(image: Image.Image, angle: float) -> Image.Image:
        """Simulate X-axis rotation (top-bottom tilt)"""
        width, height = image.size
        rad = math.radians(angle)
        factor = abs(math.sin(rad)) * 0.3

        if abs(angle) < 1:
            return image

        if angle > 0:
            # Tilt back - top goes away
            coeffs = Transform3D._find_coeffs(
                [(0, 0), (width, 0), (width, height), (0, height)],
                [(width * factor, height * factor),
                 (width * (1 - factor), height * factor),
                 (width, height),
                 (0, height)]
            )
        else:
            # Tilt forward - bottom goes away
            coeffs = Transform3D._find_coeffs(
                [(0, 0), (width, 0), (width, height), (0, height)],
                [(0, 0),
                 (width, 0),
                 (width * (1 - factor), height * (1 - factor)),
                 (width * factor, height * (1 - factor))]
            )

        return image.transform(image.size, Image.Transform.PERSPECTIVE,
                              coeffs, Image.Resampling.BICUBIC)

    @staticmethod
    def card_flip(image: Image.Image, progress: float,
                  direction: str = 'horizontal') -> Image.Image:
        """3D card flip animation"""
        # Progress 0-0.5: front face rotating away
        # Progress 0.5-1: back face rotating in

        if progress < 0.5:
            angle = progress * 180
        else:
            angle = (1 - progress) * 180

        if direction == 'horizontal':
            return Transform3D.rotate_y(image, angle)
        else:
            return Transform3D.rotate_x(image, angle)

    @staticmethod
    def _find_coeffs(source_coords: List[Tuple], target_coords: List[Tuple]) -> Tuple:
        """Find perspective transform coefficients"""
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])

        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(s for pair in zip(source_coords, source_coords) for s in pair).reshape(8)

        try:
            res = np.linalg.solve(A, B)
            return tuple(np.array(res).flatten())
        except:
            return (1, 0, 0, 0, 1, 0, 0, 0)


# ============================================================================
# ADVANCED TEXT EFFECTS
# ============================================================================

class AdvancedTextEffects:
    """Professional text effects like neon, metallic, gradient"""

    @staticmethod
    def neon_glow(draw: ImageDraw.Draw, text: str, position: Tuple[int, int],
                  font, color: Tuple[int, int, int],
                  glow_color: Tuple[int, int, int] = None,
                  intensity: float = 1.0) -> Image.Image:
        """Create neon glow text effect"""
        x, y = position
        glow_color = glow_color or color

        # Multiple glow layers
        glow_sizes = [20, 15, 10, 5, 3]

        for i, size in enumerate(glow_sizes):
            alpha = int(50 * intensity * (1 - i / len(glow_sizes)))
            glow_col = (*glow_color, alpha)

            # Draw offset copies for glow
            for dx in range(-size, size + 1, 2):
                for dy in range(-size, size + 1, 2):
                    if dx*dx + dy*dy <= size*size:
                        draw.text((x + dx, y + dy), text, font=font, fill=glow_col)

        # Core bright text
        draw.text((x, y), text, font=font, fill=(*color, 255))

        # Inner bright core
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 200))

    @staticmethod
    def create_gradient_text(text: str, size: Tuple[int, int], font,
                            colors: List[Tuple[int, int, int]],
                            direction: str = 'vertical') -> Image.Image:
        """Create text with gradient fill"""
        # Create text mask
        mask = Image.new('L', size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # Center text
        bbox = mask_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        mask_draw.text((x, y), text, font=font, fill=255)

        # Create gradient
        gradient = Image.new('RGB', size)
        gradient_draw = ImageDraw.Draw(gradient)

        if direction == 'vertical':
            for i in range(size[1]):
                ratio = i / size[1]
                # Interpolate between colors
                if len(colors) == 2:
                    r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                    g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                    b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                else:
                    # Multi-color gradient
                    segment = ratio * (len(colors) - 1)
                    idx = int(segment)
                    local_ratio = segment - idx
                    if idx >= len(colors) - 1:
                        idx = len(colors) - 2
                    r = int(colors[idx][0] * (1 - local_ratio) + colors[idx+1][0] * local_ratio)
                    g = int(colors[idx][1] * (1 - local_ratio) + colors[idx+1][1] * local_ratio)
                    b = int(colors[idx][2] * (1 - local_ratio) + colors[idx+1][2] * local_ratio)

                gradient_draw.line([(0, i), (size[0], i)], fill=(r, g, b))
        else:
            for i in range(size[0]):
                ratio = i / size[0]
                if len(colors) == 2:
                    r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                    g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                    b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
                else:
                    segment = ratio * (len(colors) - 1)
                    idx = int(segment)
                    local_ratio = segment - idx
                    if idx >= len(colors) - 1:
                        idx = len(colors) - 2
                    r = int(colors[idx][0] * (1 - local_ratio) + colors[idx+1][0] * local_ratio)
                    g = int(colors[idx][1] * (1 - local_ratio) + colors[idx+1][1] * local_ratio)
                    b = int(colors[idx][2] * (1 - local_ratio) + colors[idx+1][2] * local_ratio)

                gradient_draw.line([(i, 0), (i, size[1])], fill=(r, g, b))

        # Apply mask
        result = Image.new('RGBA', size, (0, 0, 0, 0))
        gradient_rgba = gradient.convert('RGBA')
        result.paste(gradient_rgba, mask=mask)

        return result

    @staticmethod
    def metallic_text(text: str, size: Tuple[int, int], font,
                     metal_type: str = 'gold') -> Image.Image:
        """Create metallic text effect"""
        metal_gradients = {
            'gold': [(255, 215, 0), (255, 245, 180), (180, 140, 0), (255, 223, 100)],
            'silver': [(192, 192, 192), (255, 255, 255), (128, 128, 128), (220, 220, 220)],
            'bronze': [(205, 127, 50), (255, 190, 120), (139, 90, 43), (220, 160, 80)],
            'chrome': [(200, 200, 210), (255, 255, 255), (100, 100, 120), (180, 180, 200)],
            'rose_gold': [(255, 190, 180), (255, 220, 210), (200, 140, 130), (255, 200, 190)]
        }

        colors = metal_gradients.get(metal_type, metal_gradients['gold'])
        return AdvancedTextEffects.create_gradient_text(text, size, font, colors, 'vertical')

    @staticmethod
    def outline_text(draw: ImageDraw.Draw, text: str, position: Tuple[int, int],
                    font, fill_color: Tuple[int, int, int],
                    outline_color: Tuple[int, int, int],
                    outline_width: int = 3) -> None:
        """Draw text with outline"""
        x, y = position

        # Draw outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx*dx + dy*dy <= outline_width*outline_width:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

        # Draw fill
        draw.text((x, y), text, font=font, fill=fill_color)

    @staticmethod
    def shadow_text(draw: ImageDraw.Draw, text: str, position: Tuple[int, int],
                   font, color: Tuple[int, int, int],
                   shadow_color: Tuple[int, int, int] = (0, 0, 0),
                   shadow_offset: Tuple[int, int] = (4, 4),
                   shadow_blur: int = 5) -> None:
        """Draw text with drop shadow"""
        x, y = position
        sx, sy = shadow_offset

        # Draw shadow layers
        for i in range(shadow_blur, 0, -1):
            alpha = int(150 / i)
            for dx in range(-i, i + 1):
                for dy in range(-i, i + 1):
                    draw.text((x + sx + dx, y + sy + dy), text, font=font,
                             fill=(*shadow_color, alpha))

        # Draw main text
        draw.text((x, y), text, font=font, fill=(*color, 255))


# ============================================================================
# COLOR GRADING / LUT SYSTEM
# ============================================================================

class ColorGrading:
    """Professional color grading and LUT-like effects"""

    @staticmethod
    def apply_lut(image: Image.Image, lut_name: str) -> Image.Image:
        """Apply cinematic color grading"""
        luts = {
            'cinematic_teal_orange': ColorGrading._teal_orange,
            'vintage_film': ColorGrading._vintage_film,
            'high_contrast': ColorGrading._high_contrast,
            'muted_pastel': ColorGrading._muted_pastel,
            'cyberpunk': ColorGrading._cyberpunk,
            'golden_hour': ColorGrading._golden_hour,
            'cool_blue': ColorGrading._cool_blue,
            'warm_sunset': ColorGrading._warm_sunset,
            'noir': ColorGrading._noir,
            'vibrant_pop': ColorGrading._vibrant_pop
        }

        func = luts.get(lut_name)
        if func:
            return func(image)
        return image

    @staticmethod
    def _teal_orange(image: Image.Image) -> Image.Image:
        """Cinematic teal and orange color grade"""
        # Split into channels
        r, g, b = image.split()

        # Boost orange in highlights (red channel)
        r = r.point(lambda x: min(255, int(x * 1.1)))

        # Add teal to shadows (boost blue, reduce red in darks)
        b = b.point(lambda x: min(255, int(x * 1.05 + 10)))

        # Reduce green slightly
        g = g.point(lambda x: int(x * 0.95))

        result = Image.merge('RGB', (r, g, b))

        # Boost contrast slightly
        enhancer = ImageEnhance.Contrast(result)
        return enhancer.enhance(1.1)

    @staticmethod
    def _vintage_film(image: Image.Image) -> Image.Image:
        """Vintage film look with faded blacks"""
        # Reduce contrast
        enhancer = ImageEnhance.Contrast(image)
        result = enhancer.enhance(0.85)

        # Fade blacks (lift shadows)
        r, g, b = result.split()
        r = r.point(lambda x: int(x * 0.9 + 25))
        g = g.point(lambda x: int(x * 0.85 + 20))
        b = b.point(lambda x: int(x * 0.8 + 30))

        result = Image.merge('RGB', (r, g, b))

        # Add warmth
        enhancer = ImageEnhance.Color(result)
        return enhancer.enhance(0.9)

    @staticmethod
    def _high_contrast(image: Image.Image) -> Image.Image:
        """High contrast dramatic look"""
        # Boost contrast
        enhancer = ImageEnhance.Contrast(image)
        result = enhancer.enhance(1.4)

        # Boost saturation
        enhancer = ImageEnhance.Color(result)
        return enhancer.enhance(1.2)

    @staticmethod
    def _muted_pastel(image: Image.Image) -> Image.Image:
        """Soft muted pastel colors"""
        # Reduce saturation
        enhancer = ImageEnhance.Color(image)
        result = enhancer.enhance(0.6)

        # Lift shadows, reduce contrast
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(0.8)

        # Add brightness
        enhancer = ImageEnhance.Brightness(result)
        return enhancer.enhance(1.1)

    @staticmethod
    def _cyberpunk(image: Image.Image) -> Image.Image:
        """Cyberpunk neon color grade"""
        r, g, b = image.split()

        # Push magenta and cyan
        r = r.point(lambda x: min(255, int(x * 1.1 + 20)))
        b = b.point(lambda x: min(255, int(x * 1.2 + 30)))
        g = g.point(lambda x: int(x * 0.8))

        result = Image.merge('RGB', (r, g, b))

        # High contrast
        enhancer = ImageEnhance.Contrast(result)
        return enhancer.enhance(1.3)

    @staticmethod
    def _golden_hour(image: Image.Image) -> Image.Image:
        """Warm golden hour look"""
        r, g, b = image.split()

        # Warm up
        r = r.point(lambda x: min(255, int(x * 1.15)))
        g = g.point(lambda x: min(255, int(x * 1.05)))
        b = b.point(lambda x: int(x * 0.85))

        result = Image.merge('RGB', (r, g, b))

        # Add warmth
        enhancer = ImageEnhance.Color(result)
        return enhancer.enhance(1.15)

    @staticmethod
    def _cool_blue(image: Image.Image) -> Image.Image:
        """Cool blue tint"""
        r, g, b = image.split()

        r = r.point(lambda x: int(x * 0.9))
        b = b.point(lambda x: min(255, int(x * 1.15 + 10)))

        result = Image.merge('RGB', (r, g, b))

        enhancer = ImageEnhance.Contrast(result)
        return enhancer.enhance(1.05)

    @staticmethod
    def _warm_sunset(image: Image.Image) -> Image.Image:
        """Warm sunset colors"""
        r, g, b = image.split()

        r = r.point(lambda x: min(255, int(x * 1.2 + 20)))
        g = g.point(lambda x: min(255, int(x * 1.05)))
        b = b.point(lambda x: int(x * 0.75))

        return Image.merge('RGB', (r, g, b))

    @staticmethod
    def _noir(image: Image.Image) -> Image.Image:
        """Film noir black and white with high contrast"""
        # Convert to grayscale
        gray = image.convert('L')

        # High contrast
        enhancer = ImageEnhance.Contrast(gray)
        result = enhancer.enhance(1.5)

        # Convert back to RGB
        return result.convert('RGB')

    @staticmethod
    def _vibrant_pop(image: Image.Image) -> Image.Image:
        """Vibrant saturated pop look"""
        # Boost saturation
        enhancer = ImageEnhance.Color(image)
        result = enhancer.enhance(1.5)

        # Boost contrast
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.2)

        # Boost brightness slightly
        enhancer = ImageEnhance.Brightness(result)
        return enhancer.enhance(1.05)


# ============================================================================
# ANIMATED UI ELEMENTS
# ============================================================================

class AnimatedUIElements:
    """Animated UI components for ads"""

    @staticmethod
    def animated_price_tag(draw: ImageDraw.Draw, price: str,
                          position: Tuple[int, int], progress: float,
                          style: str = 'modern',
                          color: Tuple[int, int, int] = (255, 59, 48)) -> None:
        """Draw animated price tag"""
        x, y = position

        # Animation
        scale = 0.5 + 0.5 * min(1, progress * 2) if progress < 0.5 else 1.0
        if progress > 0.8:
            # Pulse effect
            pulse = math.sin((progress - 0.8) * 50) * 0.05
            scale = 1.0 + pulse

        tag_width = int(250 * scale)
        tag_height = int(80 * scale)

        if style == 'modern':
            # Rounded rectangle
            draw.rounded_rectangle(
                [x - tag_width//2, y - tag_height//2,
                 x + tag_width//2, y + tag_height//2],
                radius=tag_height//2,
                fill=(*color, int(255 * min(1, progress * 3)))
            )
        elif style == 'badge':
            # Circle badge
            radius = tag_height // 2
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(*color, int(255 * min(1, progress * 3)))
            )
        elif style == 'ribbon':
            # Ribbon style
            points = [
                (x - tag_width//2, y - tag_height//2),
                (x + tag_width//2, y - tag_height//2),
                (x + tag_width//2 + 20, y),
                (x + tag_width//2, y + tag_height//2),
                (x - tag_width//2, y + tag_height//2),
                (x - tag_width//2 - 20, y)
            ]
            draw.polygon(points, fill=(*color, int(255 * min(1, progress * 3))))

    @staticmethod
    def countdown_timer(draw: ImageDraw.Draw, remaining: int,
                       position: Tuple[int, int], size: int = 100,
                       color: Tuple[int, int, int] = (255, 59, 48),
                       font = None) -> None:
        """Draw countdown timer"""
        x, y = position

        # Background circle
        draw.ellipse(
            [x - size, y - size, x + size, y + size],
            fill=(30, 30, 30, 200),
            outline=(*color, 255),
            width=4
        )

        # Time text
        hours = remaining // 3600
        minutes = (remaining % 3600) // 60
        seconds = remaining % 60

        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        if font:
            bbox = draw.textbbox((0, 0), time_str, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw//2, y - th//2), time_str, font=font,
                     fill=(255, 255, 255, 255))

    @staticmethod
    def progress_ring(draw: ImageDraw.Draw, progress: float,
                     center: Tuple[int, int], radius: int,
                     color: Tuple[int, int, int] = (0, 200, 100),
                     bg_color: Tuple[int, int, int] = (60, 60, 60),
                     width: int = 10) -> None:
        """Draw circular progress ring"""
        x, y = center

        # Background ring
        draw.arc(
            [x - radius, y - radius, x + radius, y + radius],
            0, 360,
            fill=(*bg_color, 200),
            width=width
        )

        # Progress arc
        end_angle = -90 + (360 * progress)
        draw.arc(
            [x - radius, y - radius, x + radius, y + radius],
            -90, end_angle,
            fill=(*color, 255),
            width=width
        )

    @staticmethod
    def star_rating(draw: ImageDraw.Draw, rating: float,
                   position: Tuple[int, int], star_size: int = 30,
                   color: Tuple[int, int, int] = (255, 193, 7)) -> None:
        """Draw star rating"""
        x, y = position
        full_stars = int(rating)
        partial = rating - full_stars

        for i in range(5):
            star_x = x + i * (star_size + 5)

            if i < full_stars:
                # Full star
                AnimatedUIElements._draw_star(draw, (star_x, y), star_size,
                                             (*color, 255))
            elif i == full_stars and partial > 0:
                # Partial star
                AnimatedUIElements._draw_star(draw, (star_x, y), star_size,
                                             (*color, int(255 * partial)))
            else:
                # Empty star
                AnimatedUIElements._draw_star(draw, (star_x, y), star_size,
                                             (100, 100, 100, 150))

    @staticmethod
    def _draw_star(draw: ImageDraw.Draw, center: Tuple[int, int],
                  size: int, color: Tuple[int, int, int, int]) -> None:
        """Draw a 5-pointed star"""
        x, y = center
        points = []

        for i in range(10):
            angle = math.pi / 2 + (i * math.pi / 5)
            r = size if i % 2 == 0 else size * 0.5
            px = x + r * math.cos(angle)
            py = y - r * math.sin(angle)
            points.append((px, py))

        draw.polygon(points, fill=color)

    @staticmethod
    def animated_badge(draw: ImageDraw.Draw, text: str,
                      position: Tuple[int, int], progress: float,
                      badge_type: str = 'sale',
                      font = None) -> None:
        """Draw animated badge (SALE, NEW, HOT, etc.)"""
        x, y = position

        badge_configs = {
            'sale': {'color': (255, 59, 48), 'bg': (255, 59, 48)},
            'new': {'color': (0, 200, 83), 'bg': (0, 200, 83)},
            'hot': {'color': (255, 149, 0), 'bg': (255, 149, 0)},
            'limited': {'color': (175, 82, 222), 'bg': (175, 82, 222)},
            'bestseller': {'color': (255, 215, 0), 'bg': (255, 215, 0)}
        }

        config = badge_configs.get(badge_type, badge_configs['sale'])

        # Animation
        if progress < 0.3:
            scale = progress / 0.3
            # Bounce in
            scale = 1 + 0.3 * math.sin(scale * math.pi)
        else:
            scale = 1.0
            # Subtle pulse
            scale += 0.05 * math.sin((progress - 0.3) * 10)

        # Badge shape
        badge_width = int(120 * scale)
        badge_height = int(40 * scale)

        # Draw rotated badge
        angle = -15

        # Background
        draw.rounded_rectangle(
            [x - badge_width//2, y - badge_height//2,
             x + badge_width//2, y + badge_height//2],
            radius=5,
            fill=(*config['bg'], int(255 * min(1, progress * 4)))
        )


# ============================================================================
# ADVANCED PARTICLES
# ============================================================================

@dataclass
class AdvancedParticle:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    rotation: float
    rotation_speed: float
    alpha: float
    color: Tuple[int, int, int]
    life: float
    max_life: float
    particle_type: str
    trail: List[Tuple[float, float]]


class AdvancedParticleSystem:
    """Advanced particle effects with trails and physics"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.particles: List[AdvancedParticle] = []
        self.gravity = 0.2
        self.wind = 0

    def emit_fire(self, x: int, y: int, count: int = 10):
        """Emit fire particles"""
        for _ in range(count):
            self.particles.append(AdvancedParticle(
                x=x + random.uniform(-20, 20),
                y=y,
                vx=random.uniform(-1, 1),
                vy=random.uniform(-8, -4),
                size=random.uniform(10, 25),
                rotation=0,
                rotation_speed=0,
                alpha=255,
                color=random.choice([
                    (255, 100, 0), (255, 150, 0), (255, 200, 50), (255, 80, 0)
                ]),
                life=1.0,
                max_life=1.0,
                particle_type='fire',
                trail=[]
            ))

    def emit_smoke(self, x: int, y: int, count: int = 5):
        """Emit smoke particles"""
        for _ in range(count):
            self.particles.append(AdvancedParticle(
                x=x + random.uniform(-10, 10),
                y=y,
                vx=random.uniform(-0.5, 0.5),
                vy=random.uniform(-3, -1),
                size=random.uniform(30, 60),
                rotation=random.uniform(0, 360),
                rotation_speed=random.uniform(-2, 2),
                alpha=150,
                color=(100, 100, 100),
                life=1.0,
                max_life=1.0,
                particle_type='smoke',
                trail=[]
            ))

    def emit_energy(self, x: int, y: int, count: int = 15,
                   color: Tuple[int, int, int] = (0, 200, 255)):
        """Emit energy/electric particles"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(5, 15)

            self.particles.append(AdvancedParticle(
                x=x,
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                size=random.uniform(3, 8),
                rotation=0,
                rotation_speed=0,
                alpha=255,
                color=color,
                life=1.0,
                max_life=1.0,
                particle_type='energy',
                trail=[(x, y)] * 5
            ))

    def emit_starburst(self, x: int, y: int, count: int = 20):
        """Emit starburst explosion"""
        for i in range(count):
            angle = (i / count) * 2 * math.pi
            speed = random.uniform(8, 15)

            self.particles.append(AdvancedParticle(
                x=x,
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                size=random.uniform(5, 12),
                rotation=random.uniform(0, 360),
                rotation_speed=random.uniform(-10, 10),
                alpha=255,
                color=random.choice([
                    (255, 215, 0), (255, 255, 200), (255, 180, 50)
                ]),
                life=1.0,
                max_life=1.0,
                particle_type='star',
                trail=[]
            ))

    def emit_hearts(self, x: int, y: int, count: int = 10):
        """Emit heart particles"""
        for _ in range(count):
            self.particles.append(AdvancedParticle(
                x=x + random.uniform(-50, 50),
                y=y,
                vx=random.uniform(-1, 1),
                vy=random.uniform(-5, -2),
                size=random.uniform(15, 30),
                rotation=random.uniform(-20, 20),
                rotation_speed=random.uniform(-3, 3),
                alpha=255,
                color=random.choice([
                    (255, 100, 150), (255, 50, 100), (255, 150, 180)
                ]),
                life=1.0,
                max_life=1.0,
                particle_type='heart',
                trail=[]
            ))

    def emit_money(self, x: int, y: int, count: int = 8):
        """Emit money/coin particles"""
        for _ in range(count):
            self.particles.append(AdvancedParticle(
                x=x + random.uniform(-100, 100),
                y=y,
                vx=random.uniform(-2, 2),
                vy=random.uniform(2, 5),
                size=random.uniform(25, 40),
                rotation=random.uniform(0, 360),
                rotation_speed=random.uniform(-5, 5),
                alpha=255,
                color=(255, 215, 0),  # Gold
                life=1.0,
                max_life=1.0,
                particle_type='coin',
                trail=[]
            ))

    def update(self, dt: float = 1/30):
        """Update all particles"""
        for p in self.particles:
            # Store trail position
            if p.particle_type == 'energy':
                p.trail.append((p.x, p.y))
                if len(p.trail) > 10:
                    p.trail.pop(0)

            # Update position
            p.x += p.vx
            p.y += p.vy

            # Apply physics
            if p.particle_type in ['fire', 'smoke']:
                p.vy -= 0.1  # Rise
                p.vx += self.wind
            elif p.particle_type in ['coin', 'confetti']:
                p.vy += self.gravity  # Fall
            elif p.particle_type == 'energy':
                p.vx *= 0.95  # Friction
                p.vy *= 0.95

            # Update rotation
            p.rotation += p.rotation_speed

            # Update life
            p.life -= dt * 0.5

            # Update alpha based on life
            if p.particle_type == 'fire':
                p.alpha = int(255 * (p.life / p.max_life))
                p.size *= 0.98  # Shrink
            elif p.particle_type == 'smoke':
                p.alpha = int(100 * (p.life / p.max_life))
                p.size *= 1.02  # Grow
            else:
                p.alpha = int(255 * (p.life / p.max_life))

        # Remove dead particles
        self.particles = [p for p in self.particles if p.life > 0]

    def render(self, frame: Image.Image) -> Image.Image:
        """Render particles onto frame"""
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        overlay = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for p in self.particles:
            if p.alpha <= 0:
                continue

            color = (*p.color, int(p.alpha))

            if p.particle_type == 'fire':
                # Gradient fire particle
                for i in range(3):
                    size = p.size * (1 - i * 0.3)
                    alpha = int(p.alpha * (1 - i * 0.3))
                    draw.ellipse([
                        p.x - size, p.y - size,
                        p.x + size, p.y + size
                    ], fill=(*p.color, alpha))

            elif p.particle_type == 'smoke':
                # Soft smoke
                for i in range(2):
                    size = p.size * (1 + i * 0.2)
                    alpha = int(p.alpha / (i + 1))
                    draw.ellipse([
                        p.x - size, p.y - size,
                        p.x + size, p.y + size
                    ], fill=(100, 100, 100, alpha))

            elif p.particle_type == 'energy':
                # Draw trail
                if len(p.trail) > 1:
                    for i in range(len(p.trail) - 1):
                        t_alpha = int(p.alpha * (i / len(p.trail)))
                        draw.line([p.trail[i], p.trail[i + 1]],
                                 fill=(*p.color, t_alpha), width=int(p.size))

                # Draw core
                draw.ellipse([
                    p.x - p.size, p.y - p.size,
                    p.x + p.size, p.y + p.size
                ], fill=color)

            elif p.particle_type == 'star':
                # 4-pointed star
                self._draw_4star(draw, (p.x, p.y), p.size, p.rotation, color)

            elif p.particle_type == 'heart':
                self._draw_heart(draw, (p.x, p.y), p.size, color)

            elif p.particle_type == 'coin':
                # Simple coin (circle with shine)
                draw.ellipse([
                    p.x - p.size, p.y - p.size * 0.8,
                    p.x + p.size, p.y + p.size * 0.8
                ], fill=color, outline=(255, 255, 200, int(p.alpha)))

        return Image.alpha_composite(frame, overlay)

    def _draw_4star(self, draw: ImageDraw.Draw, center: Tuple[float, float],
                   size: float, rotation: float, color: Tuple[int, int, int, int]):
        """Draw 4-pointed star"""
        x, y = center
        points = []

        for i in range(8):
            angle = math.radians(rotation) + (i * math.pi / 4)
            r = size if i % 2 == 0 else size * 0.3
            px = x + r * math.cos(angle)
            py = y + r * math.sin(angle)
            points.append((px, py))

        draw.polygon(points, fill=color)

    def _draw_heart(self, draw: ImageDraw.Draw, center: Tuple[float, float],
                   size: float, color: Tuple[int, int, int, int]):
        """Draw heart shape"""
        x, y = center

        # Simplified heart using circles and triangle
        r = size * 0.4
        # Left bump
        draw.ellipse([x - r * 1.5, y - r, x - r * 0.5, y + r * 0.5], fill=color)
        # Right bump
        draw.ellipse([x + r * 0.5, y - r, x + r * 1.5, y + r * 0.5], fill=color)
        # Bottom triangle
        draw.polygon([
            (x - r * 1.5, y),
            (x + r * 1.5, y),
            (x, y + size)
        ], fill=color)


# ============================================================================
# CAMERA EFFECTS
# ============================================================================

class CameraEffects:
    """Camera shake, zoom, and movement effects"""

    @staticmethod
    def shake(image: Image.Image, intensity: float = 10,
             frame_num: int = 0) -> Image.Image:
        """Apply camera shake effect"""
        # Random but consistent shake based on frame
        random.seed(frame_num)
        offset_x = int(random.uniform(-intensity, intensity))
        offset_y = int(random.uniform(-intensity, intensity))
        random.seed()

        # Create slightly larger canvas
        width, height = image.size
        canvas = Image.new('RGB', (width, height), (0, 0, 0))

        # Paste with offset
        canvas.paste(image, (offset_x, offset_y))

        return canvas

    @staticmethod
    def zoom_pulse(image: Image.Image, progress: float,
                  intensity: float = 0.1) -> Image.Image:
        """Pulsing zoom effect"""
        pulse = math.sin(progress * math.pi * 2) * intensity
        scale = 1 + pulse

        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Scale image
        scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - width) // 2
        top = (new_height - height) // 2

        return scaled.crop((left, top, left + width, top + height))

    @staticmethod
    def dolly_zoom(image: Image.Image, progress: float,
                  intensity: float = 0.3) -> Image.Image:
        """Dolly zoom (Vertigo) effect"""
        # Zoom in while moving camera back (or vice versa)
        zoom = 1 + intensity * progress

        width, height = image.size

        # Apply zoom
        new_width = int(width * zoom)
        new_height = int(height * zoom)
        scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - width) // 2
        top = (new_height - height) // 2

        return scaled.crop((left, top, left + width, top + height))

    @staticmethod
    def rack_focus(image: Image.Image, progress: float,
                  focus_point: str = 'center') -> Image.Image:
        """Rack focus effect - shift focus point"""
        width, height = image.size

        # Create blur mask based on focus point
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        focus_points = {
            'center': (width // 2, height // 2),
            'top': (width // 2, height // 4),
            'bottom': (width // 2, height * 3 // 4),
            'left': (width // 4, height // 2),
            'right': (width * 3 // 4, height // 2)
        }

        cx, cy = focus_points.get(focus_point, focus_points['center'])

        # Draw radial gradient for focus
        max_radius = int(max(width, height) * 0.8)
        for r in range(max_radius, 0, -5):
            alpha = int(255 * (1 - r / max_radius))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=alpha)

        # Create blurred version
        blur_amount = int(10 * (1 - progress))
        if blur_amount > 0:
            blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        else:
            blurred = image

        # Composite
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        if blurred.mode != 'RGBA':
            blurred = blurred.convert('RGBA')

        result = Image.composite(image, blurred, mask)
        return result


# ============================================================================
# GEOMETRIC PATTERNS
# ============================================================================

class GeometricPatterns:
    """Animated geometric patterns and shapes"""

    @staticmethod
    def rotating_circles(size: Tuple[int, int], progress: float,
                        count: int = 8, color: Tuple[int, int, int] = (255, 255, 255),
                        alpha: int = 100) -> Image.Image:
        """Create rotating circle pattern"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx, cy = width // 2, height // 2
        radius = min(width, height) // 3

        for i in range(count):
            angle = (i / count) * 2 * math.pi + (progress * 2 * math.pi)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)

            circle_r = 20 + 10 * math.sin(progress * 4 + i)
            draw.ellipse([x - circle_r, y - circle_r, x + circle_r, y + circle_r],
                        fill=(*color, alpha))

        return img

    @staticmethod
    def hexagon_grid(size: Tuple[int, int], progress: float,
                    hex_size: int = 50,
                    color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Create animated hexagon grid"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        hex_height = hex_size * 2
        hex_width = hex_size * math.sqrt(3)

        rows = int(height / (hex_height * 0.75)) + 2
        cols = int(width / hex_width) + 2

        for row in range(rows):
            for col in range(cols):
                x = col * hex_width + (row % 2) * (hex_width / 2)
                y = row * hex_height * 0.75

                # Animate each hexagon
                delay = (row + col) * 0.1
                local_progress = max(0, min(1, progress * 3 - delay))

                if local_progress > 0:
                    alpha = int(80 * local_progress)
                    points = GeometricPatterns._hexagon_points((x, y),
                                                               hex_size * local_progress)
                    draw.polygon(points, outline=(*color, alpha), width=2)

        return img

    @staticmethod
    def _hexagon_points(center: Tuple[float, float], size: float) -> List[Tuple[float, float]]:
        """Get hexagon corner points"""
        cx, cy = center
        points = []
        for i in range(6):
            angle = i * math.pi / 3 - math.pi / 6
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            points.append((x, y))
        return points

    @staticmethod
    def wave_lines(size: Tuple[int, int], progress: float,
                  line_count: int = 10,
                  color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Create animated wave lines"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        for i in range(line_count):
            y_base = (i + 1) * height // (line_count + 1)

            points = []
            for x in range(0, width, 5):
                wave = math.sin((x / 50) + progress * 4 + i * 0.5) * 30
                y = y_base + wave
                points.append((x, y))

            if len(points) > 1:
                alpha = int(150 * (1 - i / line_count))
                draw.line(points, fill=(*color, alpha), width=2)

        return img

    @staticmethod
    def particle_trail(size: Tuple[int, int], progress: float,
                      color: Tuple[int, int, int] = (255, 215, 0)) -> Image.Image:
        """Create particle trail animation"""
        width, height = size
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Trail path (sine wave)
        trail_length = 20

        for i in range(trail_length):
            t = progress - i * 0.02
            if t < 0:
                continue

            x = width * t
            y = height // 2 + math.sin(t * 6) * 100

            size_factor = 1 - i / trail_length
            particle_size = int(15 * size_factor)
            alpha = int(255 * size_factor)

            draw.ellipse([
                x - particle_size, y - particle_size,
                x + particle_size, y + particle_size
            ], fill=(*color, alpha))

        return img
