"""
Professional Video Templates - Layouts, Compositions, and Scene Templates
Pre-designed professional ad layouts with advanced effects
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from video_effects import (
    Easing, SceneTransitions, KineticTypography, ParticleSystem,
    KenBurnsEffect, VisualEffects, GradientBackgrounds
)
from pro_effects import (
    Transform3D, AdvancedTextEffects, ColorGrading, AnimatedUIElements,
    AdvancedParticleSystem, CameraEffects, GeometricPatterns
)


# ============================================================================
# LAYOUT TYPES
# ============================================================================

class LayoutType(Enum):
    FULLSCREEN = "fullscreen"           # Product fills most of screen
    SPLIT_HORIZONTAL = "split_h"        # Top/bottom split
    SPLIT_VERTICAL = "split_v"          # Left/right split
    THIRDS = "thirds"                   # Rule of thirds
    CENTERED = "centered"               # Centered with borders
    DIAGONAL = "diagonal"               # Diagonal split
    PICTURE_IN_PICTURE = "pip"          # Small overlay
    COLLAGE = "collage"                 # Multiple images
    CINEMATIC = "cinematic"             # 21:9 letterbox style
    PHONE_MOCKUP = "phone"              # Phone frame overlay
    FLOATING_CARDS = "cards"            # Floating card elements


class TemplateStyle(Enum):
    MINIMAL = "minimal"
    BOLD = "bold"
    ELEGANT = "elegant"
    PLAYFUL = "playful"
    TECH = "tech"
    ORGANIC = "organic"
    RETRO = "retro"
    NEON = "neon"
    CORPORATE = "corporate"
    INFLUENCER = "influencer"


# ============================================================================
# TEMPLATE CONFIGURATIONS
# ============================================================================

TEMPLATE_CONFIGS = {
    'minimal': {
        'layouts': [LayoutType.CENTERED, LayoutType.FULLSCREEN],
        'color_grade': 'muted_pastel',
        'text_style': 'clean',
        'particles': None,
        'borders': False,
        'vignette': 0.2,
        'accent_opacity': 0.8
    },
    'bold': {
        'layouts': [LayoutType.SPLIT_VERTICAL, LayoutType.DIAGONAL],
        'color_grade': 'high_contrast',
        'text_style': 'impact',
        'particles': 'starburst',
        'borders': True,
        'vignette': 0.4,
        'accent_opacity': 1.0
    },
    'elegant': {
        'layouts': [LayoutType.CENTERED, LayoutType.CINEMATIC],
        'color_grade': 'golden_hour',
        'text_style': 'serif',
        'particles': 'floating',
        'borders': False,
        'vignette': 0.5,
        'accent_opacity': 0.9
    },
    'playful': {
        'layouts': [LayoutType.COLLAGE, LayoutType.FLOATING_CARDS],
        'color_grade': 'vibrant_pop',
        'text_style': 'bouncy',
        'particles': 'confetti',
        'borders': True,
        'vignette': 0.2,
        'accent_opacity': 1.0
    },
    'tech': {
        'layouts': [LayoutType.SPLIT_HORIZONTAL, LayoutType.PICTURE_IN_PICTURE],
        'color_grade': 'cyberpunk',
        'text_style': 'mono',
        'particles': 'energy',
        'borders': True,
        'vignette': 0.3,
        'accent_opacity': 0.95
    },
    'neon': {
        'layouts': [LayoutType.FULLSCREEN, LayoutType.DIAGONAL],
        'color_grade': 'cyberpunk',
        'text_style': 'neon',
        'particles': 'energy',
        'borders': True,
        'vignette': 0.6,
        'accent_opacity': 1.0
    },
    'retro': {
        'layouts': [LayoutType.CENTERED, LayoutType.SPLIT_HORIZONTAL],
        'color_grade': 'vintage_film',
        'text_style': 'retro',
        'particles': None,
        'borders': True,
        'vignette': 0.4,
        'accent_opacity': 0.85
    },
    'influencer': {
        'layouts': [LayoutType.FULLSCREEN, LayoutType.PICTURE_IN_PICTURE],
        'color_grade': 'warm_sunset',
        'text_style': 'modern',
        'particles': 'hearts',
        'borders': False,
        'vignette': 0.3,
        'accent_opacity': 0.95
    }
}


# ============================================================================
# LAYOUT RENDERER
# ============================================================================

class LayoutRenderer:
    """Renders different layout compositions"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def render_layout(self, layout: LayoutType, product_image: Image.Image,
                     background: Image.Image, progress: float,
                     frame_num: int) -> Image.Image:
        """Render specified layout with product image"""

        if layout == LayoutType.FULLSCREEN:
            return self._render_fullscreen(product_image, background, progress, frame_num)
        elif layout == LayoutType.SPLIT_HORIZONTAL:
            return self._render_split_h(product_image, background, progress, frame_num)
        elif layout == LayoutType.SPLIT_VERTICAL:
            return self._render_split_v(product_image, background, progress, frame_num)
        elif layout == LayoutType.CENTERED:
            return self._render_centered(product_image, background, progress, frame_num)
        elif layout == LayoutType.DIAGONAL:
            return self._render_diagonal(product_image, background, progress, frame_num)
        elif layout == LayoutType.CINEMATIC:
            return self._render_cinematic(product_image, background, progress, frame_num)
        elif layout == LayoutType.FLOATING_CARDS:
            return self._render_floating_cards(product_image, background, progress, frame_num)
        elif layout == LayoutType.PICTURE_IN_PICTURE:
            return self._render_pip(product_image, background, progress, frame_num)
        else:
            return self._render_centered(product_image, background, progress, frame_num)

    def _render_fullscreen(self, product: Image.Image, bg: Image.Image,
                          progress: float, frame_num: int) -> Image.Image:
        """Fullscreen product with subtle background"""
        result = bg.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        # Scale product to fill most of screen
        target_size = int(min(self.width, self.height) * 0.85)

        # Ken Burns effect
        kb_progress = Easing.ease_in_out_cubic(progress)
        animated = KenBurnsEffect.apply(
            product, kb_progress,
            start_zoom=1.0, end_zoom=1.15,
            start_pos=(0.5, 0.5), end_pos=(0.5, 0.4),
            target_size=(target_size, target_size)
        )

        # Center position with float
        float_y = math.sin(frame_num * 0.05) * 20
        x = (self.width - target_size) // 2
        y = (self.height - target_size) // 2 - 100 + int(float_y)

        # Entrance animation
        if progress < 0.15:
            entrance = Easing.ease_out_back(progress / 0.15)
            scale = 0.8 + 0.2 * entrance
            new_size = int(target_size * scale)
            animated = animated.resize((new_size, new_size), Image.Resampling.LANCZOS)
            x = (self.width - new_size) // 2
            y = (self.height - new_size) // 2 - 100 + int((1 - entrance) * 200)

        if animated.mode == 'RGBA':
            result.paste(animated, (x, y), animated)
        else:
            result.paste(animated, (x, y))

        return result

    def _render_split_h(self, product: Image.Image, bg: Image.Image,
                       progress: float, frame_num: int) -> Image.Image:
        """Horizontal split - product top, text bottom"""
        result = bg.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        # Product in top half
        product_height = self.height // 2 - 50
        product_size = min(self.width - 100, product_height - 50)

        # Animated product
        animated = product.resize((product_size, product_size), Image.Resampling.LANCZOS)

        # Slide in from top
        if progress < 0.2:
            slide = Easing.ease_out_cubic(progress / 0.2)
            y_offset = int((1 - slide) * -300)
        else:
            y_offset = int(math.sin(frame_num * 0.03) * 10)

        x = (self.width - product_size) // 2
        y = 80 + y_offset

        if animated.mode == 'RGBA':
            result.paste(animated, (x, y), animated)
        else:
            result.paste(animated, (x, y))

        # Add divider line
        divider_y = self.height // 2
        draw = ImageDraw.Draw(result)

        line_progress = min(1, progress * 2)
        line_width = int(self.width * 0.8 * Easing.ease_out_cubic(line_progress))
        line_x = (self.width - line_width) // 2

        draw.line([(line_x, divider_y), (line_x + line_width, divider_y)],
                 fill=(255, 255, 255, 100), width=2)

        return result

    def _render_split_v(self, product: Image.Image, bg: Image.Image,
                       progress: float, frame_num: int) -> Image.Image:
        """Vertical split - product on one side"""
        result = bg.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        # Product on left half
        product_width = self.width // 2 - 50
        product_size = min(product_width - 50, self.height - 400)

        animated = product.resize((product_size, product_size), Image.Resampling.LANCZOS)

        # 3D tilt effect
        if progress > 0.1:
            tilt_progress = min(1, (progress - 0.1) / 0.3)
            tilt_angle = 15 * math.sin(tilt_progress * math.pi)
            animated = Transform3D.rotate_y(animated, tilt_angle)

        # Slide in from left
        if progress < 0.2:
            slide = Easing.ease_out_back(progress / 0.2)
            x_offset = int((1 - slide) * -400)
        else:
            x_offset = 0

        x = 50 + x_offset
        y = (self.height - product_size) // 2 - 50

        if animated.mode == 'RGBA':
            result.paste(animated, (x, y), animated)
        else:
            result.paste(animated, (x, y))

        # Vertical divider
        draw = ImageDraw.Draw(result)
        div_x = self.width // 2

        if progress > 0.1:
            line_progress = min(1, (progress - 0.1) * 3)
            line_height = int(self.height * 0.7 * Easing.ease_out_cubic(line_progress))
            line_y = (self.height - line_height) // 2

            draw.line([(div_x, line_y), (div_x, line_y + line_height)],
                     fill=(255, 255, 255, 80), width=2)

        return result

    def _render_centered(self, product: Image.Image, bg: Image.Image,
                        progress: float, frame_num: int) -> Image.Image:
        """Centered product with decorative frame"""
        result = bg.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        # Product size
        product_size = int(min(self.width, self.height) * 0.55)

        # Pulse effect
        pulse = 1 + 0.03 * math.sin(frame_num * 0.1)
        current_size = int(product_size * pulse)

        animated = product.resize((current_size, current_size), Image.Resampling.LANCZOS)

        # Center position
        x = (self.width - current_size) // 2
        y = (self.height - current_size) // 2 - 150

        # Entrance with scale
        if progress < 0.2:
            entrance = Easing.ease_out_elastic(progress / 0.2)
            scale = entrance
            new_size = int(current_size * scale)
            if new_size > 0:
                animated = animated.resize((new_size, new_size), Image.Resampling.LANCZOS)
                x = (self.width - new_size) // 2
                y = (self.height - new_size) // 2 - 150

        # Draw decorative frame
        draw = ImageDraw.Draw(result)
        frame_padding = 30
        frame_x1 = x - frame_padding
        frame_y1 = y - frame_padding
        frame_x2 = x + current_size + frame_padding
        frame_y2 = y + current_size + frame_padding

        # Animated frame
        if progress > 0.15:
            frame_progress = min(1, (progress - 0.15) * 4)
            alpha = int(150 * frame_progress)

            # Corner brackets
            bracket_len = 50
            draw.line([(frame_x1, frame_y1), (frame_x1 + bracket_len, frame_y1)],
                     fill=(255, 255, 255, alpha), width=3)
            draw.line([(frame_x1, frame_y1), (frame_x1, frame_y1 + bracket_len)],
                     fill=(255, 255, 255, alpha), width=3)

            draw.line([(frame_x2, frame_y1), (frame_x2 - bracket_len, frame_y1)],
                     fill=(255, 255, 255, alpha), width=3)
            draw.line([(frame_x2, frame_y1), (frame_x2, frame_y1 + bracket_len)],
                     fill=(255, 255, 255, alpha), width=3)

            draw.line([(frame_x1, frame_y2), (frame_x1 + bracket_len, frame_y2)],
                     fill=(255, 255, 255, alpha), width=3)
            draw.line([(frame_x1, frame_y2), (frame_x1, frame_y2 - bracket_len)],
                     fill=(255, 255, 255, alpha), width=3)

            draw.line([(frame_x2, frame_y2), (frame_x2 - bracket_len, frame_y2)],
                     fill=(255, 255, 255, alpha), width=3)
            draw.line([(frame_x2, frame_y2), (frame_x2, frame_y2 - bracket_len)],
                     fill=(255, 255, 255, alpha), width=3)

        if animated.mode == 'RGBA':
            result.paste(animated, (x, y), animated)
        else:
            result.paste(animated, (x, y))

        return result

    def _render_diagonal(self, product: Image.Image, bg: Image.Image,
                        progress: float, frame_num: int) -> Image.Image:
        """Diagonal split layout"""
        result = bg.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        # Create diagonal overlay
        overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Animated diagonal
        if progress > 0.05:
            diag_progress = min(1, (progress - 0.05) * 3)
            offset = int(self.width * 0.3 * Easing.ease_out_cubic(diag_progress))

            # Dark triangle on one side
            draw.polygon([
                (0, 0),
                (self.width + offset, 0),
                (offset, self.height),
                (0, self.height)
            ], fill=(0, 0, 0, 120))

        result = Image.alpha_composite(result, overlay)

        # Product on the lighter side
        product_size = int(min(self.width, self.height) * 0.5)
        animated = product.resize((product_size, product_size), Image.Resampling.LANCZOS)

        # Position in upper right
        x = self.width - product_size - 80
        y = 200 + int(math.sin(frame_num * 0.04) * 15)

        # Rotate slightly
        rotation = 5 * math.sin(frame_num * 0.02)
        animated = animated.rotate(rotation, expand=False, resample=Image.Resampling.BILINEAR)

        # Entrance
        if progress < 0.25:
            entrance = Easing.ease_out_back(progress / 0.25)
            x = int(self.width + (x - self.width) * entrance)

        if animated.mode == 'RGBA':
            result.paste(animated, (x, y), animated)
        else:
            result.paste(animated, (x, y))

        return result

    def _render_cinematic(self, product: Image.Image, bg: Image.Image,
                         progress: float, frame_num: int) -> Image.Image:
        """Cinematic letterbox style"""
        result = bg.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        # Add letterbox bars
        bar_height = int(self.height * 0.15)
        draw = ImageDraw.Draw(result)

        # Animate bars in
        if progress < 0.1:
            bar_progress = progress / 0.1
            current_bar = int(bar_height * Easing.ease_out_cubic(bar_progress))
        else:
            current_bar = bar_height

        draw.rectangle([0, 0, self.width, current_bar], fill=(0, 0, 0, 255))
        draw.rectangle([0, self.height - current_bar, self.width, self.height],
                      fill=(0, 0, 0, 255))

        # Product in center (within letterbox area)
        available_height = self.height - 2 * bar_height - 400
        product_size = min(self.width - 200, available_height)

        # Ken Burns with cinematic feel
        kb_progress = (progress * 0.5) % 1.0
        animated = KenBurnsEffect.apply(
            product, kb_progress,
            start_zoom=1.1, end_zoom=1.0,
            start_pos=(0.3, 0.5), end_pos=(0.7, 0.5),
            target_size=(product_size, product_size)
        )

        x = (self.width - product_size) // 2
        y = bar_height + 50

        if animated.mode == 'RGBA':
            result.paste(animated, (x, y), animated)
        else:
            result.paste(animated, (x, y))

        return result

    def _render_floating_cards(self, product: Image.Image, bg: Image.Image,
                              progress: float, frame_num: int) -> Image.Image:
        """Floating card elements layout"""
        result = bg.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        # Main product card
        card_width = int(self.width * 0.7)
        card_height = int(card_width * 1.2)

        # Create card background
        card = Image.new('RGBA', (card_width, card_height), (255, 255, 255, 240))
        card_draw = ImageDraw.Draw(card)

        # Round corners (simulated)
        # Add product to card
        product_size = card_width - 60
        prod_resized = product.resize((product_size, product_size), Image.Resampling.LANCZOS)

        if prod_resized.mode == 'RGBA':
            card.paste(prod_resized, (30, 30), prod_resized)
        else:
            card.paste(prod_resized, (30, 30))

        # Card shadow
        shadow = Image.new('RGBA', (card_width + 40, card_height + 40), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rectangle([20, 20, card_width + 20, card_height + 20],
                             fill=(0, 0, 0, 60))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))

        # Position with float animation
        float_y = math.sin(frame_num * 0.04) * 20
        float_x = math.cos(frame_num * 0.03) * 10

        x = (self.width - card_width) // 2 + int(float_x)
        y = 150 + int(float_y)

        # Entrance animation
        if progress < 0.2:
            entrance = Easing.ease_out_back(progress / 0.2)
            y = int(self.height + (y - self.height) * entrance)
            rotation = (1 - entrance) * 15
            card = card.rotate(rotation, expand=True, resample=Image.Resampling.BILINEAR)
            shadow = shadow.rotate(rotation, expand=True, resample=Image.Resampling.BILINEAR)

        # Paste shadow then card
        result.paste(shadow, (x - 20, y - 20 + 30), shadow)
        result.paste(card, (x, y), card)

        return result

    def _render_pip(self, product: Image.Image, bg: Image.Image,
                   progress: float, frame_num: int) -> Image.Image:
        """Picture in picture layout"""
        result = bg.copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        # Main product (large)
        main_size = int(min(self.width, self.height) * 0.65)
        main_product = product.resize((main_size, main_size), Image.Resampling.LANCZOS)

        main_x = (self.width - main_size) // 2
        main_y = 200

        if main_product.mode == 'RGBA':
            result.paste(main_product, (main_x, main_y), main_product)
        else:
            result.paste(main_product, (main_x, main_y))

        # Small PIP overlay (could be different angle or detail)
        if progress > 0.3:
            pip_progress = min(1, (progress - 0.3) * 3)

            pip_size = int(main_size * 0.35)
            pip_product = product.resize((pip_size, pip_size), Image.Resampling.LANCZOS)

            # Position in corner with bounce in
            pip_x = self.width - pip_size - 50
            pip_y = int(main_y + main_size - pip_size + 50 + (1 - Easing.ease_out_back(pip_progress)) * 200)

            # Add border to PIP
            bordered = Image.new('RGBA', (pip_size + 8, pip_size + 8), (255, 255, 255, 255))
            if pip_product.mode == 'RGBA':
                bordered.paste(pip_product, (4, 4), pip_product)
            else:
                bordered.paste(pip_product, (4, 4))

            result.paste(bordered, (pip_x - 4, pip_y - 4), bordered)

        return result


# ============================================================================
# SCENE TEMPLATES
# ============================================================================

class SceneTemplates:
    """Pre-designed scene compositions"""

    @staticmethod
    def hook_scene_explosive(width: int, height: int, product: Image.Image,
                            text: str, progress: float, frame_num: int,
                            font) -> Image.Image:
        """Explosive hook scene with particles and bold text"""
        # Dark dramatic background
        bg = GradientBackgrounds.get_preset('fire', width, height)
        bg = ColorGrading.apply_lut(bg, 'high_contrast')

        if bg.mode != 'RGBA':
            bg = bg.convert('RGBA')

        # Add geometric patterns
        pattern = GeometricPatterns.rotating_circles(
            (width, height), progress, count=12,
            color=(255, 100, 50), alpha=50
        )
        bg = Image.alpha_composite(bg, pattern)

        # Product with zoom burst
        if progress < 0.3:
            zoom = 0.5 + Easing.ease_out_elastic(progress / 0.3) * 0.5
        else:
            zoom = 1.0 + 0.05 * math.sin(frame_num * 0.15)

        prod_size = int(min(width, height) * 0.5 * zoom)
        prod_resized = product.resize((prod_size, prod_size), Image.Resampling.LANCZOS)

        prod_x = (width - prod_size) // 2
        prod_y = 200 + int(math.sin(frame_num * 0.08) * 20)

        if prod_resized.mode == 'RGBA':
            bg.paste(prod_resized, (prod_x, prod_y), prod_resized)
        else:
            bg.paste(prod_resized, (prod_x, prod_y))

        # Bold text with glow
        draw = ImageDraw.Draw(bg)
        if font and text:
            # Neon glow effect
            AdvancedTextEffects.neon_glow(
                draw, text[:30], (width // 2 - 300, height - 500),
                font, (255, 255, 255), (255, 100, 50), intensity=1.2
            )

        # Camera shake for impact
        if progress < 0.15:
            bg = CameraEffects.shake(bg, intensity=15, frame_num=frame_num)

        return bg

    @staticmethod
    def hook_scene_elegant(width: int, height: int, product: Image.Image,
                          text: str, progress: float, frame_num: int,
                          font) -> Image.Image:
        """Elegant hook with smooth animations"""
        # Soft gradient
        bg = GradientBackgrounds.get_preset('midnight', width, height)
        bg = ColorGrading.apply_lut(bg, 'golden_hour')

        if bg.mode != 'RGBA':
            bg = bg.convert('RGBA')

        # Floating particles
        pattern = GeometricPatterns.particle_trail(
            (width, height), progress, color=(255, 215, 0)
        )
        bg = Image.alpha_composite(bg, pattern)

        # Product with gentle Ken Burns
        kb = KenBurnsEffect.apply(
            product, progress,
            start_zoom=1.2, end_zoom=1.0,
            start_pos=(0.5, 0.3), end_pos=(0.5, 0.5),
            target_size=(600, 600)
        )

        prod_x = (width - 600) // 2
        prod_y = 250

        # Fade in
        if progress < 0.2:
            # Create faded version
            fade_alpha = int(255 * (progress / 0.2))
            if kb.mode != 'RGBA':
                kb = kb.convert('RGBA')
            kb.putalpha(fade_alpha)

        if kb.mode == 'RGBA':
            bg.paste(kb, (prod_x, prod_y), kb)
        else:
            bg.paste(kb, (prod_x, prod_y))

        # Elegant text
        draw = ImageDraw.Draw(bg)
        if font and text:
            text_alpha = int(255 * min(1, progress * 3))
            AdvancedTextEffects.shadow_text(
                draw, text[:35],
                (width // 2 - 350, height - 450),
                font, (255, 255, 255),
                shadow_offset=(5, 5), shadow_blur=8
            )

        # Vignette
        bg = VisualEffects.add_vignette(bg, 0.5)

        return bg

    @staticmethod
    def cta_scene_urgent(width: int, height: int, product: Image.Image,
                        text: str, price: str, progress: float, frame_num: int,
                        font) -> Image.Image:
        """Urgent CTA with pulsing elements"""
        # Red gradient for urgency
        bg = GradientBackgrounds.get_preset('fire', width, height)

        if bg.mode != 'RGBA':
            bg = bg.convert('RGBA')

        # Pulsing background intensity
        pulse = 1 + 0.1 * math.sin(frame_num * 0.2)
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(pulse)

        if bg.mode != 'RGBA':
            bg = bg.convert('RGBA')

        # Product
        prod_size = int(min(width, height) * 0.45)
        prod_resized = product.resize((prod_size, prod_size), Image.Resampling.LANCZOS)

        # Pulse product size
        pulse_size = int(prod_size * (1 + 0.05 * math.sin(frame_num * 0.15)))
        if pulse_size != prod_size:
            prod_resized = prod_resized.resize((pulse_size, pulse_size),
                                               Image.Resampling.LANCZOS)

        prod_x = (width - pulse_size) // 2
        prod_y = 200

        if prod_resized.mode == 'RGBA':
            bg.paste(prod_resized, (prod_x, prod_y), prod_resized)
        else:
            bg.paste(prod_resized, (prod_x, prod_y))

        # Price tag
        draw = ImageDraw.Draw(bg)
        if price:
            AnimatedUIElements.animated_price_tag(
                draw, price,
                (width // 2, prod_y + pulse_size + 100),
                progress, style='ribbon',
                color=(255, 59, 48)
            )

        # CTA text with neon
        if font and text:
            AdvancedTextEffects.neon_glow(
                draw, text[:25],
                (width // 2 - 280, height - 400),
                font, (255, 255, 255), (255, 50, 50), intensity=1.5
            )

        # Corner accents (pulsing)
        corner_alpha = int(150 + 100 * math.sin(frame_num * 0.2))
        corner_size = int(100 + 20 * math.sin(frame_num * 0.15))

        draw.polygon([(0, 0), (corner_size, 0), (0, corner_size)],
                    fill=(255, 255, 255, corner_alpha))
        draw.polygon([(width, 0), (width - corner_size, 0), (width, corner_size)],
                    fill=(255, 255, 255, corner_alpha))
        draw.polygon([(0, height), (corner_size, height), (0, height - corner_size)],
                    fill=(255, 255, 255, corner_alpha))
        draw.polygon([(width, height), (width - corner_size, height),
                     (width, height - corner_size)],
                    fill=(255, 255, 255, corner_alpha))

        return bg

    @staticmethod
    def features_scene_carousel(width: int, height: int, product: Image.Image,
                               features: List[str], progress: float,
                               frame_num: int, font) -> Image.Image:
        """Features scene with carousel effect"""
        bg = GradientBackgrounds.get_preset('ocean', width, height)
        bg = ColorGrading.apply_lut(bg, 'cool_blue')

        if bg.mode != 'RGBA':
            bg = bg.convert('RGBA')

        # Product on left side
        prod_size = int(min(width, height) * 0.4)
        prod_resized = product.resize((prod_size, prod_size), Image.Resampling.LANCZOS)

        # Slight 3D rotation
        rotation = 10 * math.sin(frame_num * 0.02)
        prod_resized = Transform3D.rotate_y(prod_resized, rotation)

        prod_x = 80
        prod_y = (height - prod_size) // 2 - 100

        if prod_resized.mode == 'RGBA':
            bg.paste(prod_resized, (prod_x, prod_y), prod_resized)
        else:
            bg.paste(prod_resized, (prod_x, prod_y))

        # Features list on right with staggered animation
        draw = ImageDraw.Draw(bg)
        feature_x = width // 2 + 50
        feature_y = 400

        if font and features:
            for i, feature in enumerate(features[:4]):
                # Staggered entrance
                feature_progress = max(0, min(1, (progress * 4 - i * 0.5)))

                if feature_progress > 0:
                    alpha = int(255 * Easing.ease_out_cubic(feature_progress))
                    x_offset = int((1 - Easing.ease_out_cubic(feature_progress)) * 100)

                    # Bullet point
                    bullet_y = feature_y + i * 120
                    draw.ellipse([
                        feature_x - 30 + x_offset, bullet_y + 15,
                        feature_x - 10 + x_offset, bullet_y + 35
                    ], fill=(100, 200, 255, alpha))

                    # Feature text
                    draw.text(
                        (feature_x + x_offset, bullet_y),
                        feature[:40],
                        font=font,
                        fill=(255, 255, 255, alpha)
                    )

        return bg


# ============================================================================
# TEMPLATE MANAGER
# ============================================================================

class TemplateManager:
    """Manages and applies video templates"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.layout_renderer = LayoutRenderer(width, height)
        self.particle_system = AdvancedParticleSystem(width, height)

    def get_template_config(self, style: str) -> Dict:
        """Get configuration for a template style"""
        return TEMPLATE_CONFIGS.get(style, TEMPLATE_CONFIGS['minimal'])

    def apply_template(self, template_style: str, scene_type: str,
                      product: Image.Image, text: str, progress: float,
                      frame_num: int, font, **kwargs) -> Image.Image:
        """Apply a complete template to generate a frame"""
        config = self.get_template_config(template_style)

        # Select layout based on config
        layout = random.Random(hash(scene_type)).choice(config['layouts'])

        # Create background
        gradient_presets = ['midnight', 'ocean', 'sunset', 'luxury_gold', 'neon_night']
        gradient = random.Random(hash(scene_type + template_style)).choice(gradient_presets)
        bg = GradientBackgrounds.get_preset(gradient, self.width, self.height)

        # Apply color grading
        if config['color_grade']:
            bg = ColorGrading.apply_lut(bg, config['color_grade'])

        # Render layout
        frame = self.layout_renderer.render_layout(
            layout, product, bg, progress, frame_num
        )

        # Add particles
        if config['particles']:
            if frame_num % 15 == 0:
                self._emit_particles(config['particles'])

            self.particle_system.update()
            frame = self.particle_system.render(frame)

        # Add vignette
        if config['vignette'] > 0:
            frame = VisualEffects.add_vignette(frame, config['vignette'])

        return frame

    def _emit_particles(self, particle_type: str):
        """Emit particles based on type"""
        cx, cy = self.width // 2, self.height // 2

        if particle_type == 'starburst':
            self.particle_system.emit_starburst(cx, cy, count=10)
        elif particle_type == 'confetti':
            self.particle_system.emit_confetti(cx, 0, count=5)
        elif particle_type == 'floating':
            self.particle_system.emit_smoke(cx, self.height, count=3)
        elif particle_type == 'energy':
            self.particle_system.emit_energy(cx, cy, count=8)
        elif particle_type == 'hearts':
            self.particle_system.emit_hearts(cx, self.height - 200, count=3)
        elif particle_type == 'fire':
            self.particle_system.emit_fire(cx, self.height - 100, count=5)
