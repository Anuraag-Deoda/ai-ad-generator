"""
Advanced Video Effects Module for AI Ad Generator
Professional-grade animations, transitions, and visual effects
"""

import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# EASING FUNCTIONS - Smooth, professional animation curves
# ============================================================================

class Easing:
    """Professional easing functions for smooth animations"""

    @staticmethod
    def linear(t: float) -> float:
        return t

    @staticmethod
    def ease_in_quad(t: float) -> float:
        return t * t

    @staticmethod
    def ease_out_quad(t: float) -> float:
        return 1 - (1 - t) * (1 - t)

    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2

    @staticmethod
    def ease_in_cubic(t: float) -> float:
        return t * t * t

    @staticmethod
    def ease_out_cubic(t: float) -> float:
        return 1 - pow(1 - t, 3)

    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

    @staticmethod
    def ease_in_quart(t: float) -> float:
        return t * t * t * t

    @staticmethod
    def ease_out_quart(t: float) -> float:
        return 1 - pow(1 - t, 4)

    @staticmethod
    def ease_in_out_quart(t: float) -> float:
        return 8 * t * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 4) / 2

    @staticmethod
    def ease_in_expo(t: float) -> float:
        return 0 if t == 0 else pow(2, 10 * t - 10)

    @staticmethod
    def ease_out_expo(t: float) -> float:
        return 1 if t == 1 else 1 - pow(2, -10 * t)

    @staticmethod
    def ease_in_out_expo(t: float) -> float:
        if t == 0:
            return 0
        if t == 1:
            return 1
        if t < 0.5:
            return pow(2, 20 * t - 10) / 2
        return (2 - pow(2, -20 * t + 10)) / 2

    @staticmethod
    def ease_out_back(t: float) -> float:
        """Overshoot effect - great for bouncy entrances"""
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)

    @staticmethod
    def ease_out_elastic(t: float) -> float:
        """Elastic bounce effect"""
        if t == 0:
            return 0
        if t == 1:
            return 1
        c4 = (2 * math.pi) / 3
        return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1

    @staticmethod
    def ease_out_bounce(t: float) -> float:
        """Bouncing effect"""
        n1 = 7.5625
        d1 = 2.75

        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375


# ============================================================================
# TRANSITION TYPES
# ============================================================================

class TransitionType(Enum):
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"
    WIPE_DOWN = "wipe_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    BLUR_TRANSITION = "blur_transition"
    FLASH = "flash"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    CIRCLE_REVEAL = "circle_reveal"
    DIAGONAL_WIPE = "diagonal_wipe"


class TextAnimation(Enum):
    TYPEWRITER = "typewriter"
    FADE_IN = "fade_in"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    BOUNCE_IN = "bounce_in"
    WAVE = "wave"
    GLITCH = "glitch"
    SCALE_UP = "scale_up"
    BLUR_IN = "blur_in"
    LETTER_BY_LETTER = "letter_by_letter"
    WORD_BY_WORD = "word_by_word"


# ============================================================================
# SCENE TRANSITIONS
# ============================================================================

class SceneTransitions:
    """Professional scene-to-scene transitions"""

    @staticmethod
    def fade_transition(frame1: Image.Image, frame2: Image.Image, progress: float) -> Image.Image:
        """Smooth crossfade between two frames"""
        progress = Easing.ease_in_out_cubic(progress)
        return Image.blend(frame1, frame2, progress)

    @staticmethod
    def dissolve_transition(frame1: Image.Image, frame2: Image.Image, progress: float) -> Image.Image:
        """Dissolve with noise pattern"""
        progress = Easing.ease_in_out_quad(progress)

        width, height = frame1.size
        result = frame1.copy()

        # Create noise-based dissolve mask
        noise = np.random.random((height, width))
        mask = (noise < progress).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode='L')

        # Apply slight blur to mask for smoother transition
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))

        result.paste(frame2, mask=mask_img)
        return result

    @staticmethod
    def wipe_transition(frame1: Image.Image, frame2: Image.Image, progress: float,
                       direction: str = "left") -> Image.Image:
        """Wipe transition in specified direction"""
        progress = Easing.ease_in_out_cubic(progress)

        width, height = frame1.size
        result = frame1.copy()

        if direction == "left":
            wipe_x = int(width * progress)
            if wipe_x > 0:
                crop = frame2.crop((0, 0, wipe_x, height))
                result.paste(crop, (0, 0))
        elif direction == "right":
            wipe_x = int(width * (1 - progress))
            if wipe_x < width:
                crop = frame2.crop((wipe_x, 0, width, height))
                result.paste(crop, (wipe_x, 0))
        elif direction == "up":
            wipe_y = int(height * progress)
            if wipe_y > 0:
                crop = frame2.crop((0, 0, width, wipe_y))
                result.paste(crop, (0, 0))
        elif direction == "down":
            wipe_y = int(height * (1 - progress))
            if wipe_y < height:
                crop = frame2.crop((0, wipe_y, width, height))
                result.paste(crop, (0, wipe_y))

        return result

    @staticmethod
    def zoom_blur_transition(frame1: Image.Image, frame2: Image.Image, progress: float,
                            zoom_in: bool = True) -> Image.Image:
        """Zoom with motion blur transition"""
        width, height = frame1.size

        if progress < 0.5:
            # First half: zoom and blur out frame1
            local_progress = progress * 2
            local_progress = Easing.ease_in_expo(local_progress)

            if zoom_in:
                scale = 1 + local_progress * 0.3
            else:
                scale = 1 - local_progress * 0.2

            # Scale frame
            new_size = (int(width * scale), int(height * scale))
            scaled = frame1.resize(new_size, Image.Resampling.LANCZOS)

            # Center crop
            left = (new_size[0] - width) // 2
            top = (new_size[1] - height) // 2
            result = scaled.crop((left, top, left + width, top + height))

            # Add motion blur
            blur_amount = local_progress * 15
            if blur_amount > 0:
                result = result.filter(ImageFilter.GaussianBlur(radius=blur_amount))

            # Fade to white/black at peak
            overlay = Image.new('RGB', (width, height), (255, 255, 255))
            result = Image.blend(result, overlay, local_progress * 0.7)

        else:
            # Second half: zoom and blur in frame2
            local_progress = (progress - 0.5) * 2
            local_progress = Easing.ease_out_expo(local_progress)

            if zoom_in:
                scale = 1.3 - local_progress * 0.3
            else:
                scale = 0.8 + local_progress * 0.2

            # Scale frame
            new_size = (int(width * scale), int(height * scale))
            scaled = frame2.resize(new_size, Image.Resampling.LANCZOS)

            # Center crop
            left = (new_size[0] - width) // 2
            top = (new_size[1] - height) // 2
            right = left + width
            bottom = top + height

            # Handle edge cases
            if left < 0:
                left = 0
                right = width
            if top < 0:
                top = 0
                bottom = height

            result = scaled.crop((left, top, right, bottom))
            if result.size != (width, height):
                result = result.resize((width, height), Image.Resampling.LANCZOS)

            # Reduce blur
            blur_amount = (1 - local_progress) * 15
            if blur_amount > 0:
                result = result.filter(ImageFilter.GaussianBlur(radius=blur_amount))

            # Fade from white
            overlay = Image.new('RGB', (width, height), (255, 255, 255))
            result = Image.blend(overlay, result, local_progress)

        return result

    @staticmethod
    def flash_transition(frame1: Image.Image, frame2: Image.Image, progress: float) -> Image.Image:
        """Quick flash/strobe transition"""
        width, height = frame1.size

        if progress < 0.3:
            # Build up brightness on frame1
            local_progress = progress / 0.3
            enhancer = ImageEnhance.Brightness(frame1)
            return enhancer.enhance(1 + local_progress * 2)
        elif progress < 0.5:
            # Flash white
            return Image.new('RGB', (width, height), (255, 255, 255))
        else:
            # Fade from white to frame2
            local_progress = (progress - 0.5) / 0.5
            local_progress = Easing.ease_out_expo(local_progress)
            white = Image.new('RGB', (width, height), (255, 255, 255))
            return Image.blend(white, frame2, local_progress)

    @staticmethod
    def circle_reveal_transition(frame1: Image.Image, frame2: Image.Image, progress: float) -> Image.Image:
        """Circular reveal from center"""
        progress = Easing.ease_out_cubic(progress)

        width, height = frame1.size
        result = frame1.copy()

        # Create circular mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Calculate circle radius
        max_radius = math.sqrt(width**2 + height**2) / 2
        radius = int(max_radius * progress)

        center_x, center_y = width // 2, height // 2

        draw.ellipse([
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        ], fill=255)

        # Feather the edge
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))

        result.paste(frame2, mask=mask)
        return result

    @staticmethod
    def diagonal_wipe_transition(frame1: Image.Image, frame2: Image.Image, progress: float) -> Image.Image:
        """Diagonal wipe from corner"""
        progress = Easing.ease_in_out_cubic(progress)

        width, height = frame1.size

        # Create diagonal mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Calculate diagonal line position
        offset = int((width + height) * progress)

        # Draw polygon for the revealed area
        points = [
            (0, 0),
            (offset, 0),
            (0, offset)
        ]

        if offset > width:
            points = [
                (0, 0),
                (width, 0),
                (width, offset - width),
                (0, offset)
            ]

        if offset > height:
            extra = offset - height
            points = [
                (0, 0),
                (width, 0),
                (width, height),
                (extra, height),
                (0, offset) if offset <= height else (0, height)
            ]

        if offset > width + height:
            points = [(0, 0), (width, 0), (width, height), (0, height)]

        draw.polygon(points, fill=255)

        # Feather edge
        mask = mask.filter(ImageFilter.GaussianBlur(radius=8))

        result = frame1.copy()
        result.paste(frame2, mask=mask)
        return result

    @staticmethod
    def slide_transition(frame1: Image.Image, frame2: Image.Image, progress: float,
                        direction: str = "left") -> Image.Image:
        """Slide transition with both frames visible"""
        progress = Easing.ease_out_cubic(progress)

        width, height = frame1.size
        result = Image.new('RGB', (width, height))

        if direction == "left":
            offset = int(width * progress)
            # Frame1 slides out to left
            result.paste(frame1, (-offset, 0))
            # Frame2 slides in from right
            result.paste(frame2, (width - offset, 0))
        elif direction == "right":
            offset = int(width * progress)
            result.paste(frame1, (offset, 0))
            result.paste(frame2, (-width + offset, 0))

        return result


# ============================================================================
# KINETIC TYPOGRAPHY
# ============================================================================

class KineticTypography:
    """Advanced text animation effects"""

    @staticmethod
    def typewriter_effect(text: str, progress: float,
                         include_cursor: bool = True) -> Tuple[str, bool]:
        """Typewriter effect - reveals text character by character"""
        progress = Easing.ease_out_quad(progress)

        total_chars = len(text)
        visible_chars = int(total_chars * progress)

        result = text[:visible_chars]

        # Blinking cursor
        show_cursor = include_cursor and (int(progress * 20) % 2 == 0)
        if show_cursor and progress < 1:
            result += "│"

        return result, progress >= 1

    @staticmethod
    def get_letter_positions(text: str, progress: float, animation: TextAnimation,
                            base_x: int, base_y: int, font_size: int) -> List[Dict]:
        """Get individual letter positions and properties for animation"""
        letters = []
        char_width = font_size * 0.6  # Approximate character width

        for i, char in enumerate(text):
            char_progress = max(0, min(1, (progress * len(text) - i) / 3))

            x = base_x + i * char_width
            y = base_y
            alpha = 255
            scale = 1.0
            rotation = 0

            if animation == TextAnimation.WAVE:
                wave_offset = math.sin((progress * 10) + i * 0.5) * 20
                y += wave_offset
                alpha = int(255 * Easing.ease_out_cubic(char_progress))

            elif animation == TextAnimation.BOUNCE_IN:
                if char_progress < 1:
                    bounce = Easing.ease_out_bounce(char_progress)
                    y = base_y + (1 - bounce) * -100
                    scale = 0.5 + bounce * 0.5
                alpha = int(255 * min(1, char_progress * 2))

            elif animation == TextAnimation.LETTER_BY_LETTER:
                if char_progress < 1:
                    alpha = int(255 * Easing.ease_out_cubic(char_progress))
                    scale = 0.5 + Easing.ease_out_back(char_progress) * 0.5
                    y = base_y + (1 - char_progress) * 50

            elif animation == TextAnimation.GLITCH:
                if random.random() < 0.1 * (1 - progress):
                    x += random.randint(-10, 10)
                    y += random.randint(-5, 5)
                alpha = int(255 * Easing.ease_out_expo(char_progress))

            elif animation == TextAnimation.SCALE_UP:
                scale = Easing.ease_out_back(char_progress)
                alpha = int(255 * Easing.ease_out_cubic(char_progress))

            letters.append({
                'char': char,
                'x': x,
                'y': y,
                'alpha': alpha,
                'scale': scale,
                'rotation': rotation
            })

        return letters

    @staticmethod
    def word_by_word_reveal(text: str, progress: float) -> List[Dict]:
        """Reveal text word by word with animation"""
        words = text.split()
        total_words = len(words)

        result = []
        for i, word in enumerate(words):
            word_progress = max(0, min(1, (progress * total_words - i)))

            if word_progress > 0:
                result.append({
                    'word': word,
                    'progress': Easing.ease_out_cubic(word_progress),
                    'alpha': int(255 * Easing.ease_out_cubic(word_progress)),
                    'scale': 0.8 + 0.2 * Easing.ease_out_back(word_progress),
                    'y_offset': (1 - Easing.ease_out_cubic(word_progress)) * 30
                })

        return result


# ============================================================================
# PARTICLE EFFECTS
# ============================================================================

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    alpha: float
    color: Tuple[int, int, int]
    life: float
    max_life: float


class ParticleSystem:
    """Dynamic particle effects system"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.particles: List[Particle] = []

    def emit_sparkles(self, count: int = 20, center_x: int = None, center_y: int = None):
        """Emit sparkle particles"""
        cx = center_x or self.width // 2
        cy = center_y or self.height // 2

        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)

            self.particles.append(Particle(
                x=cx + random.uniform(-50, 50),
                y=cy + random.uniform(-50, 50),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                size=random.uniform(2, 6),
                alpha=255,
                color=(255, 255, random.randint(200, 255)),  # Golden sparkles
                life=1.0,
                max_life=1.0
            ))

    def emit_confetti(self, count: int = 30):
        """Emit confetti particles from top"""
        colors = [
            (255, 107, 107),  # Red
            (78, 205, 196),   # Teal
            (255, 230, 109),  # Yellow
            (170, 111, 255),  # Purple
            (255, 154, 162),  # Pink
        ]

        for _ in range(count):
            self.particles.append(Particle(
                x=random.uniform(0, self.width),
                y=random.uniform(-50, 0),
                vx=random.uniform(-2, 2),
                vy=random.uniform(3, 7),
                size=random.uniform(8, 15),
                alpha=255,
                color=random.choice(colors),
                life=1.0,
                max_life=1.0
            ))

    def emit_floating_particles(self, count: int = 50):
        """Emit slow floating ambient particles"""
        for _ in range(count):
            self.particles.append(Particle(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                vx=random.uniform(-0.5, 0.5),
                vy=random.uniform(-1, -0.3),
                size=random.uniform(2, 5),
                alpha=random.randint(50, 150),
                color=(255, 255, 255),
                life=random.uniform(0.5, 1.0),
                max_life=1.0
            ))

    def emit_light_rays(self, count: int = 5, source_x: int = None, source_y: int = None):
        """Emit light ray particles"""
        sx = source_x or self.width // 2
        sy = source_y or 0

        for _ in range(count):
            angle = random.uniform(-0.5, 0.5) + math.pi / 2  # Downward rays

            self.particles.append(Particle(
                x=sx,
                y=sy,
                vx=math.cos(angle) * 3,
                vy=math.sin(angle) * 5,
                size=random.uniform(100, 300),  # Long rays
                alpha=random.randint(20, 50),
                color=(255, 250, 240),
                life=1.0,
                max_life=1.0
            ))

    def update(self, dt: float = 1/30):
        """Update particle positions and lifetimes"""
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= dt * 0.5
            p.alpha = int(255 * (p.life / p.max_life))

            # Gravity for confetti
            if p.size > 7:
                p.vy += 0.1

        # Remove dead particles
        self.particles = [p for p in self.particles if p.life > 0]

    def render(self, frame: Image.Image) -> Image.Image:
        """Render particles onto frame"""
        overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for p in self.particles:
            if p.alpha <= 0:
                continue

            color = (*p.color, int(p.alpha))

            if p.size > 50:  # Light rays
                # Draw as elongated ellipse
                draw.ellipse([
                    p.x - 5, p.y - p.size // 2,
                    p.x + 5, p.y + p.size // 2
                ], fill=color)
            else:
                # Draw as circle with glow
                for i in range(3):
                    glow_size = p.size * (1 + i * 0.5)
                    glow_alpha = int(p.alpha / (i + 1))
                    glow_color = (*p.color, glow_alpha)

                    draw.ellipse([
                        p.x - glow_size, p.y - glow_size,
                        p.x + glow_size, p.y + glow_size
                    ], fill=glow_color)

        # Composite onto frame
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        return Image.alpha_composite(frame, overlay)


# ============================================================================
# KEN BURNS EFFECT
# ============================================================================

class KenBurnsEffect:
    """Cinematic Ken Burns (pan and zoom) effect for images"""

    @staticmethod
    def apply(image: Image.Image, progress: float,
              start_zoom: float = 1.0, end_zoom: float = 1.2,
              start_pos: Tuple[float, float] = (0.5, 0.5),
              end_pos: Tuple[float, float] = (0.5, 0.5),
              target_size: Tuple[int, int] = None) -> Image.Image:
        """
        Apply Ken Burns effect to an image.

        Args:
            image: Source image
            progress: Animation progress (0.0 to 1.0)
            start_zoom: Starting zoom level (1.0 = 100%)
            end_zoom: Ending zoom level
            start_pos: Starting position (0.0-1.0 for x and y, 0.5 = center)
            end_pos: Ending position
            target_size: Output size (width, height)
        """
        progress = Easing.ease_in_out_cubic(progress)

        orig_width, orig_height = image.size
        target_size = target_size or (orig_width, orig_height)

        # Interpolate zoom and position
        current_zoom = start_zoom + (end_zoom - start_zoom) * progress
        current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

        # Calculate crop region
        crop_width = int(orig_width / current_zoom)
        crop_height = int(orig_height / current_zoom)

        # Calculate crop position (centered on current_x, current_y)
        max_x = orig_width - crop_width
        max_y = orig_height - crop_height

        crop_x = int(current_x * max_x)
        crop_y = int(current_y * max_y)

        # Ensure bounds
        crop_x = max(0, min(crop_x, max_x))
        crop_y = max(0, min(crop_y, max_y))

        # Crop and resize
        cropped = image.crop((
            crop_x, crop_y,
            crop_x + crop_width, crop_y + crop_height
        ))

        return cropped.resize(target_size, Image.Resampling.LANCZOS)

    @staticmethod
    def get_preset(preset_name: str) -> Dict:
        """Get preset Ken Burns configurations"""
        presets = {
            'zoom_in_center': {
                'start_zoom': 1.0, 'end_zoom': 1.3,
                'start_pos': (0.5, 0.5), 'end_pos': (0.5, 0.5)
            },
            'zoom_out_center': {
                'start_zoom': 1.3, 'end_zoom': 1.0,
                'start_pos': (0.5, 0.5), 'end_pos': (0.5, 0.5)
            },
            'pan_left_to_right': {
                'start_zoom': 1.2, 'end_zoom': 1.2,
                'start_pos': (0.0, 0.5), 'end_pos': (1.0, 0.5)
            },
            'pan_right_to_left': {
                'start_zoom': 1.2, 'end_zoom': 1.2,
                'start_pos': (1.0, 0.5), 'end_pos': (0.0, 0.5)
            },
            'zoom_in_top_left': {
                'start_zoom': 1.0, 'end_zoom': 1.4,
                'start_pos': (0.5, 0.5), 'end_pos': (0.2, 0.2)
            },
            'zoom_in_bottom_right': {
                'start_zoom': 1.0, 'end_zoom': 1.4,
                'start_pos': (0.5, 0.5), 'end_pos': (0.8, 0.8)
            },
            'dramatic_push': {
                'start_zoom': 1.0, 'end_zoom': 1.5,
                'start_pos': (0.5, 0.3), 'end_pos': (0.5, 0.5)
            },
            'pull_back_reveal': {
                'start_zoom': 1.5, 'end_zoom': 1.0,
                'start_pos': (0.5, 0.5), 'end_pos': (0.5, 0.5)
            }
        }
        return presets.get(preset_name, presets['zoom_in_center'])


# ============================================================================
# GLOW AND BLUR EFFECTS
# ============================================================================

class VisualEffects:
    """Advanced visual effects for professional look"""

    @staticmethod
    def add_glow(image: Image.Image, intensity: float = 1.0,
                color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Add glow effect to bright areas"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Extract bright areas
        enhancer = ImageEnhance.Brightness(image)
        bright = enhancer.enhance(1.5)

        # Blur for glow
        glow = bright.filter(ImageFilter.GaussianBlur(radius=20 * intensity))

        # Tint the glow
        glow_array = np.array(glow)
        for i, c in enumerate(color):
            glow_array[:, :, i] = np.clip(glow_array[:, :, i] * (c / 255), 0, 255)
        glow = Image.fromarray(glow_array.astype(np.uint8))

        # Blend
        return Image.blend(image, glow, 0.3 * intensity)

    @staticmethod
    def add_vignette(image: Image.Image, intensity: float = 0.5) -> Image.Image:
        """Add vignette (darkened corners) effect"""
        width, height = image.size

        # Create radial gradient mask
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)

        # Calculate gradient
        center_x, center_y = width // 2, height // 2
        max_dist = math.sqrt(center_x**2 + center_y**2)

        for y in range(height):
            for x in range(width):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                value = int(255 * (1 - (dist / max_dist) ** 2 * intensity))
                mask.putpixel((x, y), max(0, value))

        # Apply slight blur to mask
        mask = mask.filter(ImageFilter.GaussianBlur(radius=50))

        # Apply vignette
        darkened = ImageEnhance.Brightness(image).enhance(0.3)

        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        if darkened.mode != 'RGBA':
            darkened = darkened.convert('RGBA')

        result = Image.composite(image, darkened, mask)
        return result

    @staticmethod
    def add_film_grain(image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Add subtle film grain effect"""
        width, height = image.size

        # Generate noise
        noise = np.random.normal(0, 255 * intensity, (height, width, 3))
        noise = np.clip(noise, -50, 50).astype(np.int16)

        # Apply noise
        img_array = np.array(image).astype(np.int16)
        result = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

    @staticmethod
    def add_chromatic_aberration(image: Image.Image, intensity: float = 3) -> Image.Image:
        """Add chromatic aberration (color fringing) effect"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        r, g, b = image.split()

        # Offset red and blue channels slightly
        offset = int(intensity)

        # Shift red channel
        r_shifted = Image.new('L', image.size, 0)
        r_shifted.paste(r, (offset, 0))

        # Shift blue channel opposite direction
        b_shifted = Image.new('L', image.size, 0)
        b_shifted.paste(b, (-offset, 0))

        return Image.merge('RGB', (r_shifted, g, b_shifted))

    @staticmethod
    def add_light_leak(image: Image.Image, position: str = 'top_right',
                      color: Tuple[int, int, int] = (255, 150, 100),
                      intensity: float = 0.3) -> Image.Image:
        """Add light leak effect"""
        width, height = image.size

        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Position mapping
        positions = {
            'top_right': (width, 0),
            'top_left': (0, 0),
            'bottom_right': (width, height),
            'bottom_left': (0, height),
            'center': (width // 2, height // 2)
        }

        cx, cy = positions.get(position, positions['top_right'])

        # Draw radial gradient
        max_radius = int(max(width, height) * 0.8)

        for r in range(max_radius, 0, -5):
            alpha = int(255 * intensity * (1 - r / max_radius) ** 2)
            fill_color = (*color, alpha)

            draw.ellipse([
                cx - r, cy - r,
                cx + r, cy + r
            ], fill=fill_color)

        # Apply blur
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=50))

        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        return Image.alpha_composite(image, overlay)

    @staticmethod
    def add_motion_blur(image: Image.Image, angle: float = 0,
                       intensity: int = 20) -> Image.Image:
        """Add directional motion blur"""
        # Create motion blur kernel
        size = intensity
        kernel = np.zeros((size, size))

        # Calculate line based on angle
        center = size // 2
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))

        for i in range(size):
            offset = i - center
            x = int(center + offset * cos_a)
            y = int(center + offset * sin_a)
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1

        kernel = kernel / kernel.sum()

        # Apply kernel using PIL (approximation with box blur + rotation)
        rotated = image.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)
        blurred = rotated.filter(ImageFilter.BoxBlur(intensity // 2))
        result = blurred.rotate(angle, expand=False, resample=Image.Resampling.BILINEAR)

        return result


# ============================================================================
# GRADIENT BACKGROUNDS
# ============================================================================

class GradientBackgrounds:
    """Professional gradient background generator"""

    @staticmethod
    def create_linear_gradient(width: int, height: int,
                               color1: Tuple[int, int, int],
                               color2: Tuple[int, int, int],
                               angle: float = 0) -> Image.Image:
        """Create linear gradient at any angle"""
        base = Image.new('RGB', (width, height), color1)
        draw = ImageDraw.Draw(base)

        if angle == 0:
            # Vertical gradient (fastest)
            for y in range(height):
                ratio = y / height
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
        elif angle == 90:
            # Horizontal gradient
            for x in range(width):
                ratio = x / width
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                draw.line([(x, 0), (x, height)], fill=(r, g, b))
        else:
            # Angled gradient
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Calculate max distance for normalization
            max_dist = abs(width * cos_a) + abs(height * sin_a)

            for y in range(height):
                for x in range(width):
                    # Project point onto gradient line
                    dist = (x * cos_a + y * sin_a) / max_dist
                    dist = max(0, min(1, dist))

                    r = int(color1[0] * (1 - dist) + color2[0] * dist)
                    g = int(color1[1] * (1 - dist) + color2[1] * dist)
                    b = int(color1[2] * (1 - dist) + color2[2] * dist)

                    base.putpixel((x, y), (r, g, b))

        return base

    @staticmethod
    def create_radial_gradient(width: int, height: int,
                               inner_color: Tuple[int, int, int],
                               outer_color: Tuple[int, int, int],
                               center: Tuple[float, float] = (0.5, 0.5)) -> Image.Image:
        """Create radial gradient"""
        base = Image.new('RGB', (width, height))

        cx = int(width * center[0])
        cy = int(height * center[1])
        max_dist = math.sqrt(max(cx, width - cx)**2 + max(cy, height - cy)**2)

        for y in range(height):
            for x in range(width):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                ratio = min(1, dist / max_dist)

                r = int(inner_color[0] * (1 - ratio) + outer_color[0] * ratio)
                g = int(inner_color[1] * (1 - ratio) + outer_color[1] * ratio)
                b = int(inner_color[2] * (1 - ratio) + outer_color[2] * ratio)

                base.putpixel((x, y), (r, g, b))

        return base

    @staticmethod
    def create_multi_stop_gradient(width: int, height: int,
                                   stops: List[Tuple[float, Tuple[int, int, int]]]) -> Image.Image:
        """Create gradient with multiple color stops"""
        base = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(base)

        # Sort stops by position
        stops = sorted(stops, key=lambda x: x[0])

        for y in range(height):
            ratio = y / height

            # Find surrounding stops
            color = stops[-1][1]  # Default to last color

            for i in range(len(stops) - 1):
                if stops[i][0] <= ratio <= stops[i + 1][0]:
                    # Interpolate between these stops
                    local_ratio = (ratio - stops[i][0]) / (stops[i + 1][0] - stops[i][0])
                    c1 = stops[i][1]
                    c2 = stops[i + 1][1]

                    color = (
                        int(c1[0] * (1 - local_ratio) + c2[0] * local_ratio),
                        int(c1[1] * (1 - local_ratio) + c2[1] * local_ratio),
                        int(c1[2] * (1 - local_ratio) + c2[2] * local_ratio)
                    )
                    break

            draw.line([(0, y), (width, y)], fill=color)

        return base

    @staticmethod
    def get_preset(preset_name: str, width: int, height: int) -> Image.Image:
        """Get preset gradient backgrounds"""
        presets = {
            'sunset': [(0.0, (255, 94, 77)), (0.5, (255, 154, 86)), (1.0, (45, 13, 71))],
            'ocean': [(0.0, (0, 50, 80)), (0.5, (0, 100, 130)), (1.0, (0, 150, 180))],
            'midnight': [(0.0, (10, 10, 30)), (0.5, (30, 30, 60)), (1.0, (50, 50, 90))],
            'luxury_gold': [(0.0, (20, 15, 10)), (0.5, (60, 50, 30)), (1.0, (30, 25, 15))],
            'fresh_mint': [(0.0, (20, 60, 50)), (0.5, (40, 100, 80)), (1.0, (60, 140, 110))],
            'royal_purple': [(0.0, (30, 10, 50)), (0.5, (70, 30, 100)), (1.0, (40, 20, 70))],
            'fire': [(0.0, (80, 20, 10)), (0.4, (200, 80, 20)), (0.7, (255, 150, 50)), (1.0, (60, 15, 5))],
            'arctic': [(0.0, (200, 220, 240)), (0.5, (150, 180, 210)), (1.0, (100, 140, 180))],
            'neon_night': [(0.0, (10, 5, 20)), (0.3, (40, 10, 60)), (0.7, (20, 5, 40)), (1.0, (5, 2, 15))],
            'warm_earth': [(0.0, (60, 40, 30)), (0.5, (100, 70, 50)), (1.0, (50, 35, 25))],
        }

        stops = presets.get(preset_name, presets['midnight'])
        return GradientBackgrounds.create_multi_stop_gradient(width, height, stops)


# ============================================================================
# SHAPE ANIMATIONS
# ============================================================================

class ShapeAnimations:
    """Animated shapes and decorative elements"""

    @staticmethod
    def draw_animated_circle(draw: ImageDraw.Draw, center: Tuple[int, int],
                            radius: float, progress: float,
                            color: Tuple[int, int, int, int],
                            style: str = 'fill'):
        """Draw animated circle with various styles"""
        x, y = center
        r = radius * Easing.ease_out_elastic(progress)

        if style == 'fill':
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        elif style == 'stroke':
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        elif style == 'pulse':
            # Multiple rings expanding
            for i in range(3):
                ring_progress = (progress + i * 0.2) % 1.0
                ring_r = radius * ring_progress
                ring_alpha = int(color[3] * (1 - ring_progress))
                ring_color = (*color[:3], ring_alpha)
                draw.ellipse([x - ring_r, y - ring_r, x + ring_r, y + ring_r],
                           outline=ring_color, width=2)

    @staticmethod
    def draw_animated_line(draw: ImageDraw.Draw, start: Tuple[int, int],
                          end: Tuple[int, int], progress: float,
                          color: Tuple[int, int, int, int], width: int = 3):
        """Draw line that animates from start to end"""
        progress = Easing.ease_out_cubic(progress)

        current_x = start[0] + (end[0] - start[0]) * progress
        current_y = start[1] + (end[1] - start[1]) * progress

        draw.line([start, (current_x, current_y)], fill=color, width=width)

    @staticmethod
    def draw_progress_bar(draw: ImageDraw.Draw, position: Tuple[int, int],
                         size: Tuple[int, int], progress: float,
                         bg_color: Tuple[int, int, int, int],
                         fill_color: Tuple[int, int, int, int],
                         rounded: bool = True):
        """Draw animated progress bar"""
        x, y = position
        w, h = size

        # Background
        if rounded:
            draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg_color)
        else:
            draw.rectangle([x, y, x + w, y + h], fill=bg_color)

        # Fill
        fill_width = int(w * Easing.ease_out_cubic(progress))
        if fill_width > 0:
            if rounded:
                draw.rounded_rectangle([x, y, x + fill_width, y + h],
                                      radius=h // 2, fill=fill_color)
            else:
                draw.rectangle([x, y, x + fill_width, y + h], fill=fill_color)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def blend_images_with_mask(base: Image.Image, overlay: Image.Image,
                          mask: Image.Image) -> Image.Image:
    """Blend two images using a mask"""
    if base.mode != 'RGBA':
        base = base.convert('RGBA')
    if overlay.mode != 'RGBA':
        overlay = overlay.convert('RGBA')

    return Image.composite(overlay, base, mask)


def create_rounded_rectangle_mask(size: Tuple[int, int], radius: int) -> Image.Image:
    """Create a rounded rectangle mask"""
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size[0], size[1]], radius=radius, fill=255)
    return mask


def apply_shadow_to_image(image: Image.Image, offset: Tuple[int, int] = (10, 10),
                         blur_radius: int = 15, shadow_color: Tuple[int, int, int] = (0, 0, 0),
                         opacity: float = 0.5) -> Image.Image:
    """Apply drop shadow to an image"""
    # Create shadow
    shadow = Image.new('RGBA', image.size, (*shadow_color, int(255 * opacity)))

    # Use image alpha as shadow shape
    if image.mode == 'RGBA':
        shadow.putalpha(image.split()[3])

    # Blur shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Create output with extra space for shadow
    output_size = (
        image.size[0] + abs(offset[0]) + blur_radius * 2,
        image.size[1] + abs(offset[1]) + blur_radius * 2
    )
    output = Image.new('RGBA', output_size, (0, 0, 0, 0))

    # Paste shadow
    shadow_pos = (
        blur_radius + max(0, offset[0]),
        blur_radius + max(0, offset[1])
    )
    output.paste(shadow, shadow_pos, shadow)

    # Paste original image
    image_pos = (
        blur_radius + max(0, -offset[0]),
        blur_radius + max(0, -offset[1])
    )
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    output.paste(image, image_pos, image)

    return output
