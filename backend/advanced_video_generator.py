"""
Advanced Video Generator with Professional Effects
Integrates all video_effects modules for cinematic ad generation
"""

import math
import random
import logging
import os
import subprocess
import requests
import textwrap
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from video_effects import (
    Easing,
    TransitionType,
    TextAnimation,
    SceneTransitions,
    KineticTypography,
    ParticleSystem,
    KenBurnsEffect,
    VisualEffects,
    GradientBackgrounds,
    ShapeAnimations,
    apply_shadow_to_image
)

# Import PRO effects
from pro_effects import (
    Transform3D,
    AdvancedTextEffects,
    ColorGrading,
    AnimatedUIElements,
    AdvancedParticleSystem,
    CameraEffects,
    GeometricPatterns
)

# Import templates
from video_templates import (
    LayoutType,
    TemplateStyle,
    LayoutRenderer,
    SceneTemplates,
    TemplateManager,
    TEMPLATE_CONFIGS
)

# Import dynamic content
from dynamic_content import (
    DynamicContentRenderer,
    PriceDisplay,
    CountdownTimer,
    StarRating,
    ReviewQuote,
    CTAButton,
    render_price,
    render_countdown,
    render_rating,
    render_quote,
    render_cta
)

# Import industry templates
from industry_templates import (
    IndustryTemplateRenderer,
    TemplateConfig,
    SceneConfig,
    INDUSTRY_TEMPLATES,
    COLOR_SCHEMES,
    get_template,
    list_templates,
    list_industries,
    apply_template
)

# Import new effects
from video_effects import LensFlare, GlitchEffect
from pro_effects import KineticTypography as KineticTypographyPro

# Import advanced modules
from scene_analyzer import (
    SceneAnalyzer,
    SmartSceneComposer,
    analyze_product_images,
    get_smart_scene_composition,
    extract_color_theme
)
from motion_graphics import (
    Easing as AdvancedEasing,
    EasingType,
    Transform3DEngine,
    MorphEngine,
    LiquidEffect,
    PathAnimation,
    ProceduralMotion,
    apply_3d_transform,
    transition_morph,
    apply_liquid_effect
)
from color_grading import (
    ColorGrader,
    ColorGradePreset,
    ColorGradeSettings,
    HDRToneMapper,
    apply_color_grade,
    get_available_presets,
    create_custom_grade
)
from particle_system import (
    ParticleSystem as AdvancedParticleEngine,
    ParticlePresets,
    ParticleConfig,
    EmitterConfig,
    ParticleShape,
    BlendMode,
    create_particle_effect,
    render_particles
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DynamicContentConfig:
    """Dynamic content configuration for video"""
    show_price: bool = False
    original_price: Optional[str] = None
    sale_price: Optional[str] = None
    price_animation: str = "drop"  # drop, slide, flash, bounce

    show_countdown: bool = False
    countdown_seconds: int = 86400  # 24 hours default
    countdown_style: str = "flip"  # flip, digital, minimal, urgent

    show_rating: bool = False
    rating: float = 4.8
    review_count: Optional[int] = None
    rating_animation: str = "fill"  # fill, pop, glow

    show_review: bool = False
    review_quote: Optional[str] = None
    review_author: Optional[str] = None

    cta_text: str = "Shop Now"
    cta_style: str = "pulse"  # pulse, shake, glow, swipe_up, bounce
    cta_color: Tuple[int, int, int] = (255, 87, 51)


@dataclass
class VideoConfig:
    """Video generation configuration"""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    transition_duration: float = 0.5  # seconds
    enable_particles: bool = True
    enable_ken_burns: bool = True
    enable_vignette: bool = True
    enable_film_grain: bool = False
    enable_color_grading: bool = True
    enable_3d_effects: bool = True
    enable_camera_effects: bool = True
    enable_advanced_text: bool = True
    enable_geometric_patterns: bool = True
    text_animation: TextAnimation = TextAnimation.SLIDE_UP
    quality_preset: str = "high"  # low, medium, high
    template_style: str = "bold"  # minimal, bold, elegant, playful, tech, neon

    # New dynamic content settings
    dynamic_content: Optional[DynamicContentConfig] = None
    industry_template: Optional[str] = None  # e.g., "ecommerce/flash_sale"

    # New effects settings
    enable_lens_flare: bool = False
    enable_glitch_effects: bool = False
    glitch_intensity: float = 0.5

    # Advanced module settings
    enable_smart_composition: bool = True  # Use AI scene analysis
    enable_motion_graphics: bool = True  # Advanced transitions
    enable_pro_color_grading: bool = True  # Professional color grading
    enable_advanced_particles: bool = True  # Advanced particle system
    color_grade_preset: str = "cinematic_teal_orange"  # Color grade preset name
    color_grade_intensity: float = 0.7  # 0-1 intensity
    transition_style: str = "dissolve"  # dissolve, wipe, iris, morph
    liquid_effects: bool = False  # Enable liquid/wave distortions
    enable_3d_transforms: bool = False  # 3D perspective effects
    particle_preset: str = "sparkles"  # Particle effect preset


class SceneType(Enum):
    HOOK = "hook"
    PITCH = "pitch"
    FEATURES = "features"
    CTA = "cta"


@dataclass
class Scene:
    """Individual scene configuration"""
    type: SceneType
    text: str
    duration: float  # in seconds
    start_frame: int
    end_frame: int
    transition_in: TransitionType
    transition_out: TransitionType
    text_animation: TextAnimation
    ken_burns_preset: str
    gradient_preset: str
    particle_effect: str  # sparkles, confetti, floating, none
    accent_color: Tuple[int, int, int]


# ============================================================================
# SCENE CONFIGURATIONS BY STYLE
# ============================================================================

STYLE_CONFIGS = {
    'energetic': {
        'gradients': ['fire', 'neon_night', 'sunset', 'royal_purple'],
        'transitions': [TransitionType.FLASH, TransitionType.ZOOM_IN, TransitionType.SLIDE_LEFT],
        'text_animations': [TextAnimation.BOUNCE_IN, TextAnimation.SCALE_UP, TextAnimation.GLITCH],
        'ken_burns': ['zoom_in_center', 'dramatic_push', 'pan_left_to_right'],
        'particles': ['sparkles', 'confetti', 'sparkles', 'confetti'],
        'advanced_particles': ['starburst', 'energy', 'fire', 'confetti'],
        'accent_colors': [(255, 107, 107), (255, 230, 109), (78, 205, 196), (255, 154, 162)],
        'vignette_intensity': 0.3,
        'color_grade': 'vibrant_pop',
        'text_style': 'neon',
        'camera_effects': ['shake', 'zoom_pulse'],
        'layouts': ['fullscreen', 'diagonal', 'split_v'],
        'geometric_patterns': ['rotating_circles', 'starburst'],
    },
    'professional': {
        'gradients': ['midnight', 'ocean', 'arctic', 'warm_earth'],
        'transitions': [TransitionType.FADE, TransitionType.DISSOLVE, TransitionType.WIPE_LEFT],
        'text_animations': [TextAnimation.FADE_IN, TextAnimation.SLIDE_UP, TextAnimation.WORD_BY_WORD],
        'ken_burns': ['zoom_in_center', 'pull_back_reveal', 'pan_right_to_left'],
        'particles': ['floating', 'none', 'floating', 'floating'],
        'advanced_particles': ['floating', 'none', 'floating', 'floating'],
        'accent_colors': [(100, 150, 200), (200, 200, 200), (150, 180, 200), (100, 180, 150)],
        'vignette_intensity': 0.4,
        'color_grade': 'cinematic_teal_orange',
        'text_style': 'clean',
        'camera_effects': ['rack_focus'],
        'layouts': ['centered', 'cinematic', 'split_h'],
        'geometric_patterns': ['wave_lines'],
    },
    'casual': {
        'gradients': ['fresh_mint', 'sunset', 'ocean', 'warm_earth'],
        'transitions': [TransitionType.SLIDE_LEFT, TransitionType.CIRCLE_REVEAL, TransitionType.FADE],
        'text_animations': [TextAnimation.WAVE, TextAnimation.TYPEWRITER, TextAnimation.SLIDE_UP],
        'ken_burns': ['pan_left_to_right', 'zoom_out_center', 'pan_right_to_left'],
        'particles': ['floating', 'sparkles', 'floating', 'confetti'],
        'advanced_particles': ['hearts', 'confetti', 'floating', 'confetti'],
        'accent_colors': [(255, 200, 150), (150, 220, 180), (200, 180, 220), (255, 180, 180)],
        'vignette_intensity': 0.25,
        'color_grade': 'warm_sunset',
        'text_style': 'playful',
        'camera_effects': ['zoom_pulse'],
        'layouts': ['floating_cards', 'fullscreen', 'pip'],
        'geometric_patterns': ['particle_trail', 'wave_lines'],
    },
    'luxury': {
        'gradients': ['luxury_gold', 'midnight', 'royal_purple', 'luxury_gold'],
        'transitions': [TransitionType.DISSOLVE, TransitionType.FADE, TransitionType.BLUR_TRANSITION],
        'text_animations': [TextAnimation.FADE_IN, TextAnimation.LETTER_BY_LETTER, TextAnimation.SCALE_UP],
        'ken_burns': ['zoom_in_center', 'dramatic_push', 'pull_back_reveal'],
        'particles': ['floating', 'floating', 'sparkles', 'sparkles'],
        'advanced_particles': ['floating', 'sparkles', 'floating', 'money'],
        'accent_colors': [(212, 175, 55), (255, 215, 0), (192, 192, 192), (255, 223, 186)],
        'vignette_intensity': 0.5,
        'color_grade': 'golden_hour',
        'text_style': 'metallic',
        'camera_effects': ['dolly_zoom', 'rack_focus'],
        'layouts': ['centered', 'cinematic', 'floating_cards'],
        'geometric_patterns': ['hexagon_grid'],
    }
}


# ============================================================================
# ADVANCED VIDEO GENERATOR CLASS
# ============================================================================

class AdvancedVideoGenerator:
    """Professional-grade video generator with cinematic effects"""

    def __init__(self, config: VideoConfig = None):
        self.config = config or VideoConfig()
        self.particle_system = ParticleSystem(self.config.width, self.config.height)
        self.advanced_particles = AdvancedParticleSystem(self.config.width, self.config.height)
        self.layout_renderer = LayoutRenderer(self.config.width, self.config.height)
        self.template_manager = TemplateManager(self.config.width, self.config.height)
        self.scenes: List[Scene] = []
        self.product_images: List[Image.Image] = []
        self.fonts = self._load_fonts()

        # Initialize advanced modules
        self.scene_analyzer = SceneAnalyzer() if self.config.enable_smart_composition else None
        self.color_grader = ColorGrader() if self.config.enable_pro_color_grading else None
        self.pro_particle_system = None  # Initialized per-scene
        self.scene_analyses = []  # Store image analyses
        self.smart_composition = None  # Store AI-generated composition

    def _load_fonts(self) -> Dict[str, ImageFont.FreeTypeFont]:
        """Load fonts with fallbacks"""
        fonts = {}
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ]

        sizes = {
            'title': 72,
            'heading': 60,
            'body': 48,
            'caption': 36,
            'small': 28
        }

        for size_name, size in sizes.items():
            font = None
            for path in font_paths:
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except:
                    continue

            if font is None:
                font = ImageFont.load_default()

            fonts[size_name] = font

        return fonts

    def load_product_images(self, image_urls: List[str], max_images: int = 3) -> None:
        """Load and preprocess product images"""
        self.product_images = []

        for url in image_urls[:max_images]:
            try:
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                img = Image.open(BytesIO(response.content)).convert('RGBA')

                # Resize to standard size
                img = img.resize((800, 800), Image.Resampling.LANCZOS)

                # Add shadow
                img = apply_shadow_to_image(img, offset=(15, 15), blur_radius=20, opacity=0.4)

                self.product_images.append(img)
                logger.info(f"Loaded product image: {url[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to load image {url}: {e}")

        if not self.product_images:
            # Create placeholder
            self.product_images.append(self._create_placeholder_image())

    def _create_placeholder_image(self) -> Image.Image:
        """Create a stylish placeholder image"""
        img = Image.new('RGBA', (800, 800), (45, 45, 55, 255))
        draw = ImageDraw.Draw(img)

        # Add gradient overlay
        for y in range(800):
            alpha = int(50 * (y / 800))
            draw.line([(0, y), (800, y)], fill=(255, 255, 255, alpha))

        # Add icon or text
        draw.text((350, 380), "PRODUCT", fill=(150, 150, 160, 255), font=self.fonts['body'])

        return apply_shadow_to_image(img, offset=(15, 15), blur_radius=20, opacity=0.4)

    def analyze_images(self) -> None:
        """Analyze product images for smart composition"""
        if not self.scene_analyzer or not self.product_images:
            return

        self.scene_analyses = []
        for img in self.product_images:
            try:
                analysis = self.scene_analyzer.analyze_image(img)
                self.scene_analyses.append(analysis)
                logger.info(f"Analyzed image: composition={analysis.composition.value}, mood={analysis.mood.value}")
            except Exception as e:
                logger.warning(f"Failed to analyze image: {e}")
                self.scene_analyses.append(None)

    def apply_pro_color_grade(self, frame: Image.Image, scene_progress: float = 0.5) -> Image.Image:
        """Apply professional color grading to frame"""
        if not self.color_grader or not self.config.enable_pro_color_grading:
            return frame

        try:
            preset_name = self.config.color_grade_preset
            intensity = self.config.color_grade_intensity

            # Try to get preset enum
            try:
                preset = ColorGradePreset(preset_name)
                return self.color_grader.apply_preset(frame, preset, intensity)
            except ValueError:
                # Use as custom name
                return apply_color_grade(frame, preset_name, intensity)

        except Exception as e:
            logger.warning(f"Color grading failed: {e}")
            return frame

    def apply_advanced_transition(self, frame1: Image.Image, frame2: Image.Image,
                                  progress: float, transition_type: str = None) -> Image.Image:
        """Apply advanced transition between frames"""
        if not self.config.enable_motion_graphics:
            # Simple blend fallback
            return Image.blend(frame1, frame2, progress)

        try:
            t_type = transition_type or self.config.transition_style

            if t_type == 'dissolve':
                return MorphEngine.cross_dissolve(frame1, frame2, progress)
            elif t_type == 'wipe':
                direction = random.choice(['left', 'right', 'up', 'down'])
                return MorphEngine.wipe_transition(frame1, frame2, progress, direction)
            elif t_type == 'iris':
                return MorphEngine.iris_transition(frame1, frame2, progress)
            elif t_type == 'pixelate':
                return MorphEngine.pixelate_transition(frame1, frame2, progress)
            elif t_type == 'morph':
                # Use liquid morph for organic transition
                return LiquidEffect.blob_morph(
                    Image.blend(frame1, frame2, progress),
                    progress * 0.5,
                    intensity=0.2
                )
            else:
                return MorphEngine.cross_dissolve(frame1, frame2, progress)

        except Exception as e:
            logger.warning(f"Advanced transition failed: {e}")
            return Image.blend(frame1, frame2, progress)

    def apply_liquid_effect_to_frame(self, frame: Image.Image, progress: float,
                                     effect_type: str = 'wave') -> Image.Image:
        """Apply liquid/wave effects to frame"""
        if not self.config.liquid_effects:
            return frame

        try:
            return apply_liquid_effect(frame, progress, effect_type)
        except Exception as e:
            logger.warning(f"Liquid effect failed: {e}")
            return frame

    def apply_3d_transform_to_image(self, img: Image.Image, rx: float = 0,
                                    ry: float = 0, rz: float = 0) -> Image.Image:
        """Apply 3D perspective transform to image"""
        if not self.config.enable_3d_transforms:
            return img

        try:
            return apply_3d_transform(img, rx, ry, rz)
        except Exception as e:
            logger.warning(f"3D transform failed: {e}")
            return img

    def get_pro_particle_frame(self, frame: Image.Image, progress: float,
                               effect_type: str = None) -> Image.Image:
        """Render advanced particle effects onto frame"""
        if not self.config.enable_advanced_particles:
            return frame

        try:
            preset = effect_type or self.config.particle_preset
            return render_particles(frame, preset, progress, duration=5.0)
        except Exception as e:
            logger.warning(f"Pro particle rendering failed: {e}")
            return frame

    def get_smart_ken_burns_preset(self, image_index: int) -> str:
        """Get AI-recommended Ken Burns preset for image"""
        if not self.scene_analyses or image_index >= len(self.scene_analyses):
            return 'zoom_in_center'

        analysis = self.scene_analyses[image_index]
        if analysis and hasattr(analysis, 'recommended_ken_burns'):
            return analysis.recommended_ken_burns

        return 'zoom_in_center'

    def get_smart_color_scheme(self, image_index: int) -> Dict[str, Tuple[int, int, int]]:
        """Get AI-extracted color scheme from image"""
        if not self.scene_analyses or image_index >= len(self.scene_analyses):
            return {
                'primary': (255, 255, 255),
                'secondary': (200, 200, 200),
                'accent': (255, 87, 51),
                'background': (0, 0, 0)
            }

        analysis = self.scene_analyses[image_index]
        if analysis and hasattr(analysis, 'color_palette'):
            return {
                'primary': analysis.color_palette.dominant,
                'secondary': analysis.color_palette.secondary[0] if analysis.color_palette.secondary else analysis.color_palette.dominant,
                'accent': analysis.color_palette.accent,
                'background': analysis.color_palette.background
            }

        return {
            'primary': (255, 255, 255),
            'secondary': (200, 200, 200),
            'accent': (255, 87, 51),
            'background': (0, 0, 0)
        }

    def setup_scenes(self, script: Dict, style: str, duration: int) -> None:
        """Configure scenes based on script and style"""
        self.scenes = []
        style_config = STYLE_CONFIGS.get(style, STYLE_CONFIGS['professional'])

        # Calculate scene durations
        scene_durations = {
            SceneType.HOOK: duration * 0.2,
            SceneType.PITCH: duration * 0.3,
            SceneType.FEATURES: duration * 0.3,
            SceneType.CTA: duration * 0.2
        }

        total_frames = duration * self.config.fps
        current_frame = 0

        scene_configs = [
            (SceneType.HOOK, script.get('hook', 'Amazing Product Alert!')),
            (SceneType.PITCH, script.get('pitch', 'This will change your life!')),
            (SceneType.FEATURES, script.get('features', 'Premium quality features!')),
            (SceneType.CTA, script.get('cta', 'Get Yours Now!'))
        ]

        for i, (scene_type, text) in enumerate(scene_configs):
            scene_duration = scene_durations[scene_type]
            frame_count = int(scene_duration * self.config.fps)
            end_frame = current_frame + frame_count

            scene = Scene(
                type=scene_type,
                text=text,
                duration=scene_duration,
                start_frame=current_frame,
                end_frame=end_frame,
                transition_in=style_config['transitions'][i % len(style_config['transitions'])],
                transition_out=TransitionType.FADE,
                text_animation=style_config['text_animations'][i % len(style_config['text_animations'])],
                ken_burns_preset=style_config['ken_burns'][i % len(style_config['ken_burns'])],
                gradient_preset=style_config['gradients'][i % len(style_config['gradients'])],
                particle_effect=style_config['particles'][i % len(style_config['particles'])],
                accent_color=style_config['accent_colors'][i % len(style_config['accent_colors'])]
            )

            self.scenes.append(scene)
            current_frame = end_frame

        self.style_config = style_config

    def get_current_scene(self, frame_num: int) -> Tuple[Scene, float]:
        """Get the current scene and progress within it"""
        for scene in self.scenes:
            if scene.start_frame <= frame_num < scene.end_frame:
                progress = (frame_num - scene.start_frame) / (scene.end_frame - scene.start_frame)
                return scene, progress

        # Default to last scene
        return self.scenes[-1], 1.0

    def is_in_transition(self, frame_num: int) -> Tuple[bool, float, Scene, Scene]:
        """Check if we're in a transition between scenes"""
        transition_frames = int(self.config.transition_duration * self.config.fps)

        for i, scene in enumerate(self.scenes[:-1]):
            next_scene = self.scenes[i + 1]
            transition_start = scene.end_frame - transition_frames // 2
            transition_end = next_scene.start_frame + transition_frames // 2

            if transition_start <= frame_num < transition_end:
                progress = (frame_num - transition_start) / (transition_end - transition_start)
                return True, progress, scene, next_scene

        return False, 0, None, None

    def render_scene_frame(self, scene: Scene, progress: float, frame_num: int) -> Image.Image:
        """Render a single scene frame with all PRO effects"""
        width, height = self.config.width, self.config.height

        # 1. Create gradient background
        background = GradientBackgrounds.get_preset(scene.gradient_preset, width, height)

        # 2. Apply color grading (PRO)
        if self.config.enable_color_grading:
            color_grade = self.style_config.get('color_grade', 'cinematic_teal_orange')
            background = ColorGrading.apply_lut(background, color_grade)

        # 3. Add geometric patterns (PRO)
        if self.config.enable_geometric_patterns:
            patterns = self.style_config.get('geometric_patterns', [])
            if patterns:
                background = self._add_geometric_patterns(background, scene, progress, frame_num, patterns)

        # 4. Add vignette
        if self.config.enable_vignette:
            vignette_intensity = self.style_config.get('vignette_intensity', 0.4)
            background = VisualEffects.add_vignette(background, vignette_intensity)

        # 5. Add product image with Ken Burns + 3D effects
        if self.product_images and self.config.enable_ken_burns:
            background = self._add_product_image_pro(background, scene, progress, frame_num)

        # 6. Add animated text with PRO effects (neon, metallic, etc.)
        if self.config.enable_advanced_text:
            background = self._add_animated_text_pro(background, scene, progress, frame_num)
        else:
            background = self._add_animated_text(background, scene, progress, frame_num)

        # 7. Add advanced particles (PRO)
        if self.config.enable_particles and scene.particle_effect != 'none':
            background = self._add_advanced_particles(background, scene, progress, frame_num)

        # 7.5. Add dynamic content (pricing, countdown, rating, CTA)
        if self.config.dynamic_content:
            background = self._add_dynamic_content(background, scene, progress, frame_num)

        # 7.6. Apply lens flare effect
        if self.config.enable_lens_flare and scene.type == SceneType.HOOK:
            flare_progress = progress * 0.5  # Slower movement
            flare_x = int(self.config.width * 0.8 - self.config.width * 0.3 * flare_progress)
            flare_y = int(self.config.height * 0.2)
            background = LensFlare.render(background, (flare_x, flare_y), intensity=0.4)

        # 7.7. Apply glitch effect
        if self.config.enable_glitch_effects and scene.type == SceneType.HOOK:
            background = GlitchEffect.animated_glitch(
                background, progress,
                glitch_type="digital",
                peak_time=0.3,
                peak_duration=0.2
            )

        # 8. Add scene-specific effects (enhanced)
        background = self._add_scene_effects_pro(background, scene, progress, frame_num)

        # 9. Apply camera effects (PRO)
        if self.config.enable_camera_effects:
            background = self._apply_camera_effects(background, scene, progress, frame_num)

        # 10. Add film grain (optional)
        if self.config.enable_film_grain:
            background = VisualEffects.add_film_grain(background, intensity=0.03)

        # 11. Add light leaks for cinematic feel
        if scene.type in [SceneType.HOOK, SceneType.CTA] and progress > 0.3:
            leak_intensity = 0.12 * min(1, (progress - 0.3) / 0.3)
            background = VisualEffects.add_light_leak(
                background,
                position='top_right' if frame_num % 60 < 30 else 'top_left',
                color=scene.accent_color,
                intensity=leak_intensity
            )

        return background

    def _add_geometric_patterns(self, frame: Image.Image, scene: Scene,
                                progress: float, frame_num: int,
                                patterns: List[str]) -> Image.Image:
        """Add animated geometric patterns"""
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        width, height = self.config.width, self.config.height
        pattern_name = patterns[list(SceneType).index(scene.type) % len(patterns)]

        if pattern_name == 'rotating_circles':
            pattern = GeometricPatterns.rotating_circles(
                (width, height), progress,
                count=10, color=scene.accent_color, alpha=40
            )
        elif pattern_name == 'hexagon_grid':
            pattern = GeometricPatterns.hexagon_grid(
                (width, height), progress,
                hex_size=60, color=scene.accent_color
            )
        elif pattern_name == 'wave_lines':
            pattern = GeometricPatterns.wave_lines(
                (width, height), progress,
                line_count=8, color=scene.accent_color
            )
        elif pattern_name == 'particle_trail':
            pattern = GeometricPatterns.particle_trail(
                (width, height), progress,
                color=scene.accent_color
            )
        else:
            return frame

        return Image.alpha_composite(frame, pattern)

    def _add_product_image_pro(self, frame: Image.Image, scene: Scene,
                               progress: float, frame_num: int) -> Image.Image:
        """Add product image with PRO 3D effects"""
        if not self.product_images:
            return frame

        # Select image based on scene
        img_index = list(SceneType).index(scene.type) % len(self.product_images)
        product_img = self.product_images[img_index]

        # Apply Ken Burns effect
        kb_preset = KenBurnsEffect.get_preset(scene.ken_burns_preset)
        kb_progress = Easing.ease_in_out_cubic(progress)

        animated_img = KenBurnsEffect.apply(
            product_img, kb_progress,
            start_zoom=kb_preset['start_zoom'],
            end_zoom=kb_preset['end_zoom'],
            start_pos=kb_preset['start_pos'],
            end_pos=kb_preset['end_pos'],
            target_size=(600, 600)
        )

        # Apply 3D rotation effect (PRO)
        if self.config.enable_3d_effects and scene.type in [SceneType.PITCH, SceneType.FEATURES]:
            rotation_angle = 12 * math.sin(frame_num * 0.03)
            animated_img = Transform3D.rotate_y(animated_img, rotation_angle)

        # Add floating animation
        float_offset = math.sin(frame_num * 0.05) * 15

        # Position product image
        img_x = (self.config.width - 600) // 2
        img_y = 200 + int(float_offset)

        # Entrance animation with bounce
        if progress < 0.15:
            entrance_progress = progress / 0.15
            entrance_progress = Easing.ease_out_back(entrance_progress)
            scale = 0.6 + 0.4 * entrance_progress
            new_size = int(600 * scale)
            animated_img = animated_img.resize((new_size, new_size), Image.Resampling.LANCZOS)
            img_x = (self.config.width - new_size) // 2
            img_y = 200 + int((1 - entrance_progress) * 150) + int(float_offset)

        # Convert frame to RGBA
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        # Paste with alpha
        if animated_img.mode == 'RGBA':
            frame.paste(animated_img, (img_x, img_y), animated_img)
        else:
            frame.paste(animated_img, (img_x, img_y))

        return frame

    def _add_animated_text_pro(self, frame: Image.Image, scene: Scene,
                               progress: float, frame_num: int) -> Image.Image:
        """Add PRO animated text with neon, metallic, gradient effects"""
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        text = scene.text
        if not text:
            return frame

        text_style = self.style_config.get('text_style', 'clean')

        # Create text overlay
        text_layer = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # Choose font based on scene type
        if scene.type == SceneType.HOOK:
            font = self.fonts['title']
            max_width = 25
        elif scene.type == SceneType.CTA:
            font = self.fonts['heading']
            max_width = 30
        else:
            font = self.fonts['body']
            max_width = 35

        # Wrap text
        wrapped_text = textwrap.fill(text, width=max_width)
        lines = wrapped_text.split('\n')

        # Text position (lower third)
        text_y_base = self.config.height - 550

        # Calculate alpha based on progress
        if progress < 0.2:
            text_alpha = int(255 * Easing.ease_out_cubic(progress / 0.2))
        else:
            text_alpha = 255

        current_y = text_y_base

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (self.config.width - text_width) // 2

            # Calculate per-line animation
            line_progress = max(0, min(1, (progress * len(lines) - i) / 2))
            y_offset = int((1 - Easing.ease_out_cubic(line_progress)) * 50)

            # Apply text style (PRO)
            if text_style == 'neon' and scene.type in [SceneType.HOOK, SceneType.CTA]:
                AdvancedTextEffects.neon_glow(
                    draw, line,
                    (text_x, current_y + y_offset),
                    font, (255, 255, 255),
                    scene.accent_color,
                    intensity=1.0 + 0.3 * math.sin(frame_num * 0.15)
                )
            elif text_style == 'metallic':
                # Draw metallic text
                metallic_text = AdvancedTextEffects.metallic_text(
                    line, (text_width + 40, 100), font, 'gold'
                )
                if metallic_text.mode == 'RGBA':
                    text_layer.paste(metallic_text, (text_x - 20, current_y + y_offset - 10), metallic_text)
            else:
                # Clean style with shadows
                AdvancedTextEffects.shadow_text(
                    draw, line,
                    (text_x, current_y + y_offset),
                    font, (255, 255, 255),
                    shadow_color=(0, 0, 0),
                    shadow_offset=(4, 4),
                    shadow_blur=6
                )

            current_y += 80

        frame = Image.alpha_composite(frame, text_layer)
        return frame

    def _add_advanced_particles(self, frame: Image.Image, scene: Scene,
                                progress: float, frame_num: int) -> Image.Image:
        """Add PRO advanced particle effects"""
        advanced_particles = self.style_config.get('advanced_particles', ['floating'])
        particle_type = advanced_particles[list(SceneType).index(scene.type) % len(advanced_particles)]

        if particle_type == 'none':
            return frame

        # Emit particles based on type
        if frame_num % 12 == 0:
            cx, cy = self.config.width // 2, self.config.height // 2

            if particle_type == 'starburst':
                self.advanced_particles.emit_starburst(cx, 400, count=8)
            elif particle_type == 'energy':
                self.advanced_particles.emit_energy(cx, cy, count=6, color=scene.accent_color)
            elif particle_type == 'fire':
                self.advanced_particles.emit_fire(cx, self.config.height - 100, count=4)
            elif particle_type == 'confetti':
                self.advanced_particles.emit_confetti(self.config.width // 2, 0, count=4)
            elif particle_type == 'hearts':
                self.advanced_particles.emit_hearts(cx, self.config.height - 300, count=3)
            elif particle_type == 'money':
                self.advanced_particles.emit_money(cx, 0, count=2)
            elif particle_type == 'smoke':
                self.advanced_particles.emit_smoke(cx, self.config.height, count=2)
            elif particle_type == 'floating':
                self.advanced_particles.emit_smoke(cx, self.config.height, count=1)

        # Update and render
        self.advanced_particles.update()
        return self.advanced_particles.render(frame)

    def _add_dynamic_content(self, frame: Image.Image, scene: Scene,
                             progress: float, frame_num: int) -> Image.Image:
        """Add dynamic content elements based on scene type"""
        if not self.config.dynamic_content:
            return frame

        dc = self.config.dynamic_content
        width, height = self.config.width, self.config.height

        # Convert frame for OpenCV compatibility
        if frame.mode != 'RGB':
            frame_rgb = frame.convert('RGB')
        else:
            frame_rgb = frame

        # Convert PIL to numpy for dynamic_content module
        frame_cv = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)

        # Price Animation - show in PITCH scene
        if dc.show_price and dc.original_price and scene.type == SceneType.PITCH:
            price_y = height - 350
            frame_cv = render_price(
                frame_cv,
                dc.original_price,
                dc.sale_price,
                progress,
                (width // 2 - 150, price_y),
                animation=dc.price_animation
            )

        # Countdown Timer - show in FEATURES scene
        if dc.show_countdown and scene.type == SceneType.FEATURES:
            countdown_y = 900
            frame_cv = render_countdown(
                frame_cv,
                dc.countdown_seconds,
                progress,
                (width // 2 - 120, countdown_y),
                style=dc.countdown_style
            )

        # Star Rating - show in FEATURES scene
        if dc.show_rating and scene.type == SceneType.FEATURES:
            rating_y = height - 450
            frame_cv = render_rating(
                frame_cv,
                dc.rating,
                progress,
                (width // 2 - 180, rating_y),
                review_count=dc.review_count,
                animation=dc.rating_animation
            )

        # Review Quote - show in FEATURES scene
        if dc.show_review and dc.review_quote and dc.review_author and scene.type == SceneType.FEATURES:
            quote_y = height - 600
            frame_cv = render_quote(
                frame_cv,
                dc.review_quote,
                dc.review_author,
                progress,
                (width // 2 - 300, quote_y),
                animation="typewriter"
            )

        # CTA Button - show in CTA scene
        if scene.type == SceneType.CTA:
            cta_y = height - 300
            frame_cv = render_cta(
                frame_cv,
                dc.cta_text,
                progress,
                (width // 2 - 120, cta_y),
                style=dc.cta_style,
                color=dc.cta_color
            )

        # Convert back to PIL
        result = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))

        # Preserve alpha channel if original had one
        if frame.mode == 'RGBA':
            result = result.convert('RGBA')

        return result

    def _add_scene_effects_pro(self, frame: Image.Image, scene: Scene,
                               progress: float, frame_num: int) -> Image.Image:
        """Add PRO scene-specific visual effects"""
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        overlay = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = self.config.width, self.config.height

        if scene.type == SceneType.HOOK:
            # Animated border glow with pulse
            border_pulse = (math.sin(frame_num * 0.15) + 1) / 2
            border_alpha = int(120 + border_pulse * 100)
            color = (*scene.accent_color, border_alpha)

            # Top and bottom glowing bars
            bar_height = int(10 + border_pulse * 5)
            draw.rectangle([0, 0, width, bar_height], fill=color)
            draw.rectangle([0, height - bar_height, width, height], fill=color)

            # Animated corner brackets
            bracket_size = int(80 + border_pulse * 20)
            bracket_width = 4

            # Top-left
            draw.rectangle([0, 0, bracket_size, bracket_width], fill=color)
            draw.rectangle([0, 0, bracket_width, bracket_size], fill=color)
            # Top-right
            draw.rectangle([width - bracket_size, 0, width, bracket_width], fill=color)
            draw.rectangle([width - bracket_width, 0, width, bracket_size], fill=color)
            # Bottom-left
            draw.rectangle([0, height - bracket_width, bracket_size, height], fill=color)
            draw.rectangle([0, height - bracket_size, bracket_width, height], fill=color)
            # Bottom-right
            draw.rectangle([width - bracket_size, height - bracket_width, width, height], fill=color)
            draw.rectangle([width - bracket_width, height - bracket_size, width, height], fill=color)

        elif scene.type == SceneType.CTA:
            # Urgency pulsing corners
            pulse = (math.sin(frame_num * 0.25) + 1) / 2
            alpha = int(100 + pulse * 120)
            corner_size = int(120 + pulse * 40)
            color = (*scene.accent_color, alpha)

            # Triangular corners
            draw.polygon([(0, 0), (corner_size, 0), (0, corner_size)], fill=color)
            draw.polygon([(width, 0), (width - corner_size, 0), (width, corner_size)], fill=color)
            draw.polygon([(0, height), (corner_size, height), (0, height - corner_size)], fill=color)
            draw.polygon([(width, height), (width - corner_size, height), (width, height - corner_size)], fill=color)

            # Add pulsing "LIMITED TIME" badge
            if progress > 0.4:
                badge_progress = (progress - 0.4) / 0.6
                badge_alpha = int(255 * Easing.ease_out_back(min(1, badge_progress * 2)))
                badge_y = height - 180

                badge_pulse = 1 + 0.05 * math.sin(frame_num * 0.2)
                badge_width = int(280 * badge_pulse)
                badge_height = int(50 * badge_pulse)
                badge_x = (width - badge_width) // 2

                draw.rounded_rectangle(
                    [badge_x, badge_y, badge_x + badge_width, badge_y + badge_height],
                    radius=badge_height // 2,
                    fill=(*scene.accent_color, badge_alpha)
                )

        elif scene.type == SceneType.FEATURES:
            # Subtle animated side accents
            accent_alpha = int(80 + math.sin(frame_num * 0.08) * 30)
            accent_width = 6

            # Animated line height
            line_progress = min(1, progress * 2)
            line_height = int((height - 600) * Easing.ease_out_cubic(line_progress))
            line_y = (height - line_height) // 2

            draw.rectangle([0, line_y, accent_width, line_y + line_height],
                          fill=(*scene.accent_color, accent_alpha))
            draw.rectangle([width - accent_width, line_y, width, line_y + line_height],
                          fill=(*scene.accent_color, accent_alpha))

        elif scene.type == SceneType.PITCH:
            # Subtle gradient overlay at bottom
            for y in range(height - 300, height):
                alpha = int(60 * ((y - (height - 300)) / 300))
                draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

        frame = Image.alpha_composite(frame, overlay)
        return frame

    def _apply_camera_effects(self, frame: Image.Image, scene: Scene,
                              progress: float, frame_num: int) -> Image.Image:
        """Apply PRO camera effects"""
        camera_effects = self.style_config.get('camera_effects', [])

        if not camera_effects:
            return frame

        # Apply effects based on scene
        if scene.type == SceneType.HOOK and 'shake' in camera_effects:
            # Camera shake on hook entrance
            if progress < 0.2:
                shake_intensity = 12 * (1 - progress / 0.2)
                frame = CameraEffects.shake(frame, intensity=shake_intensity, frame_num=frame_num)

        if scene.type == SceneType.CTA and 'zoom_pulse' in camera_effects:
            # Zoom pulse on CTA
            frame = CameraEffects.zoom_pulse(frame, progress, intensity=0.03)

        if 'dolly_zoom' in camera_effects and scene.type == SceneType.PITCH:
            # Subtle dolly zoom
            frame = CameraEffects.dolly_zoom(frame, progress, intensity=0.05)

        return frame

    def _add_product_image(self, frame: Image.Image, scene: Scene, progress: float,
                          frame_num: int) -> Image.Image:
        """Add product image with Ken Burns effect"""
        if not self.product_images:
            return frame

        # Select image based on scene
        img_index = list(SceneType).index(scene.type) % len(self.product_images)
        product_img = self.product_images[img_index]

        # Apply Ken Burns effect
        kb_preset = KenBurnsEffect.get_preset(scene.ken_burns_preset)

        # Adjust progress for smoother animation across scene
        kb_progress = Easing.ease_in_out_cubic(progress)

        animated_img = KenBurnsEffect.apply(
            product_img,
            kb_progress,
            start_zoom=kb_preset['start_zoom'],
            end_zoom=kb_preset['end_zoom'],
            start_pos=kb_preset['start_pos'],
            end_pos=kb_preset['end_pos'],
            target_size=(600, 600)
        )

        # Add floating animation
        float_offset = math.sin(frame_num * 0.05) * 15

        # Position product image
        img_x = (self.config.width - 600) // 2
        img_y = 200 + int(float_offset)

        # Entrance animation
        if progress < 0.2:
            entrance_progress = progress / 0.2
            entrance_progress = Easing.ease_out_back(entrance_progress)
            scale = 0.8 + 0.2 * entrance_progress
            new_size = int(600 * scale)
            animated_img = animated_img.resize((new_size, new_size), Image.Resampling.LANCZOS)
            img_x = (self.config.width - new_size) // 2
            img_y = 200 + int((1 - entrance_progress) * 100) + int(float_offset)

        # Convert frame to RGBA
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        # Paste with alpha
        if animated_img.mode == 'RGBA':
            frame.paste(animated_img, (img_x, img_y), animated_img)
        else:
            frame.paste(animated_img, (img_x, img_y))

        return frame

    def _add_animated_text(self, frame: Image.Image, scene: Scene, progress: float,
                          frame_num: int) -> Image.Image:
        """Add animated text to frame"""
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        text = scene.text
        if not text:
            return frame

        # Create text overlay
        text_layer = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # Choose font based on scene type
        if scene.type == SceneType.HOOK:
            font = self.fonts['title']
            max_width = 25
        elif scene.type == SceneType.CTA:
            font = self.fonts['heading']
            max_width = 30
        else:
            font = self.fonts['body']
            max_width = 35

        # Wrap text
        wrapped_text = textwrap.fill(text, width=max_width)
        lines = wrapped_text.split('\n')

        # Calculate text position (lower third)
        text_y_base = self.config.height - 600

        # Animation based on type
        if scene.text_animation == TextAnimation.TYPEWRITER:
            visible_text, _ = KineticTypography.typewriter_effect(wrapped_text, progress)
            lines = visible_text.split('\n')

        elif scene.text_animation == TextAnimation.WORD_BY_WORD:
            words_data = KineticTypography.word_by_word_reveal(wrapped_text, progress)
            # Render word by word with effects
            x_offset = 100
            y_offset = text_y_base

            for word_info in words_data:
                word = word_info['word']
                alpha = word_info['alpha']
                scale = word_info['scale']
                y_off = word_info['y_offset']

                # Draw word with effects
                self._draw_text_with_effects(
                    draw, word,
                    (x_offset, y_offset + y_off),
                    font, scene.accent_color, alpha
                )

                # Move to next word position
                bbox = draw.textbbox((0, 0), word + " ", font=font)
                x_offset += bbox[2] - bbox[0]

                if x_offset > self.config.width - 150:
                    x_offset = 100
                    y_offset += 70

            frame = Image.alpha_composite(frame, text_layer)
            return frame

        # Standard line-by-line rendering with effects
        current_y = text_y_base

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            # Get text dimensions
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (self.config.width - text_width) // 2

            # Calculate per-line animation
            line_progress = max(0, min(1, (progress * len(lines) - i) / 2))

            # Apply animation effects
            alpha = int(255 * Easing.ease_out_cubic(line_progress))
            y_offset = 0
            x_offset = 0

            if scene.text_animation == TextAnimation.SLIDE_UP:
                y_offset = int((1 - Easing.ease_out_cubic(line_progress)) * 80)

            elif scene.text_animation == TextAnimation.BOUNCE_IN:
                if line_progress < 1:
                    bounce = Easing.ease_out_bounce(line_progress)
                    y_offset = int((1 - bounce) * -120)

            elif scene.text_animation == TextAnimation.SCALE_UP:
                if line_progress < 1:
                    # Scale effect approximated by alpha
                    alpha = int(255 * Easing.ease_out_back(line_progress))

            elif scene.text_animation == TextAnimation.WAVE:
                wave_offset = math.sin((frame_num * 0.1) + i * 0.5) * 10
                y_offset = int(wave_offset)

            elif scene.text_animation == TextAnimation.GLITCH:
                if random.random() < 0.05:
                    x_offset = random.randint(-15, 15)
                    y_offset = random.randint(-5, 5)

            # Draw text with shadow and glow
            self._draw_text_with_effects(
                draw, line,
                (text_x + x_offset, current_y + y_offset),
                font, scene.accent_color, alpha
            )

            current_y += 80

        frame = Image.alpha_composite(frame, text_layer)
        return frame

    def _draw_text_with_effects(self, draw: ImageDraw.Draw, text: str,
                                position: Tuple[int, int], font: ImageFont.FreeTypeFont,
                                color: Tuple[int, int, int], alpha: int) -> None:
        """Draw text with shadow, outline, and glow effects"""
        x, y = position

        # Multiple shadow layers for depth
        shadow_offsets = [(6, 6), (4, 4), (2, 2)]
        for i, (sx, sy) in enumerate(shadow_offsets):
            shadow_alpha = int(alpha * 0.3 / (i + 1))
            draw.text((x + sx, y + sy), text, font=font, fill=(0, 0, 0, shadow_alpha))

        # Glow effect (lighter version behind)
        glow_alpha = int(alpha * 0.4)
        for dx in [-2, 2]:
            for dy in [-2, 2]:
                draw.text((x + dx, y + dy), text, font=font, fill=(*color, glow_alpha))

        # Main text
        draw.text((x, y), text, font=font, fill=(255, 255, 255, alpha))

    def _add_particles(self, frame: Image.Image, scene: Scene, progress: float,
                      frame_num: int) -> Image.Image:
        """Add particle effects to frame"""
        # Emit particles based on scene type
        if frame_num % 10 == 0:  # Emit every 10 frames
            if scene.particle_effect == 'sparkles':
                self.particle_system.emit_sparkles(count=5, center_x=self.config.width // 2,
                                                   center_y=400)
            elif scene.particle_effect == 'confetti':
                self.particle_system.emit_confetti(count=3)
            elif scene.particle_effect == 'floating':
                self.particle_system.emit_floating_particles(count=2)

        # Update and render particles
        self.particle_system.update()
        frame = self.particle_system.render(frame)

        return frame

    def _add_scene_effects(self, frame: Image.Image, scene: Scene, progress: float,
                          frame_num: int) -> Image.Image:
        """Add scene-specific visual effects"""
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        overlay = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if scene.type == SceneType.HOOK:
            # Animated border glow
            border_pulse = (math.sin(frame_num * 0.15) + 1) / 2
            border_alpha = int(100 + border_pulse * 80)
            color = (*scene.accent_color, border_alpha)

            # Top and bottom glowing bars
            bar_height = int(8 + border_pulse * 4)
            draw.rectangle([0, 0, self.config.width, bar_height], fill=color)
            draw.rectangle([0, self.config.height - bar_height, self.config.width, self.config.height],
                          fill=color)

            # Corner accents
            corner_size = 80
            for corner in [(0, 0), (self.config.width - corner_size, 0),
                          (0, self.config.height - corner_size),
                          (self.config.width - corner_size, self.config.height - corner_size)]:
                draw.rectangle([corner[0], corner[1],
                               corner[0] + corner_size, corner[1] + corner_size],
                              fill=(*scene.accent_color, int(border_alpha * 0.5)))

        elif scene.type == SceneType.CTA:
            # Pulsing corners and urgency effect
            pulse = (math.sin(frame_num * 0.2) + 1) / 2
            alpha = int(80 + pulse * 100)
            color = (*scene.accent_color, alpha)

            # Corner triangles
            corner_size = int(100 + pulse * 30)
            # Top-left
            draw.polygon([(0, 0), (corner_size, 0), (0, corner_size)], fill=color)
            # Top-right
            draw.polygon([(self.config.width, 0), (self.config.width - corner_size, 0),
                         (self.config.width, corner_size)], fill=color)
            # Bottom-left
            draw.polygon([(0, self.config.height), (corner_size, self.config.height),
                         (0, self.config.height - corner_size)], fill=color)
            # Bottom-right
            draw.polygon([(self.config.width, self.config.height),
                         (self.config.width - corner_size, self.config.height),
                         (self.config.width, self.config.height - corner_size)], fill=color)

            # "Limited Time" badge animation
            if progress > 0.3:
                badge_progress = (progress - 0.3) / 0.7
                badge_alpha = int(255 * Easing.ease_out_back(min(1, badge_progress * 2)))
                badge_y = int(self.config.height - 200 - (1 - Easing.ease_out_back(min(1, badge_progress * 2))) * 50)

                # Badge background
                badge_width, badge_height = 300, 60
                badge_x = (self.config.width - badge_width) // 2
                draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_width, badge_y + badge_height],
                                       radius=30, fill=(*scene.accent_color, badge_alpha))

        elif scene.type == SceneType.FEATURES:
            # Subtle side accents
            accent_alpha = int(60 + math.sin(frame_num * 0.1) * 20)
            accent_width = 5

            # Left accent line
            draw.rectangle([0, 300, accent_width, self.config.height - 300],
                          fill=(*scene.accent_color, accent_alpha))
            # Right accent line
            draw.rectangle([self.config.width - accent_width, 300,
                           self.config.width, self.config.height - 300],
                          fill=(*scene.accent_color, accent_alpha))

        frame = Image.alpha_composite(frame, overlay)

        # Add light leak for certain scenes
        if scene.type in [SceneType.HOOK, SceneType.CTA] and progress > 0.5:
            leak_intensity = 0.15 * (progress - 0.5) / 0.5
            positions = ['top_right', 'top_left']
            position = positions[frame_num // 30 % len(positions)]
            frame = VisualEffects.add_light_leak(frame, position=position,
                                                 color=scene.accent_color,
                                                 intensity=leak_intensity)

        return frame

    def apply_transition(self, frame1: Image.Image, frame2: Image.Image,
                        progress: float, transition_type: TransitionType) -> Image.Image:
        """Apply transition between two frames"""
        if transition_type == TransitionType.FADE:
            return SceneTransitions.fade_transition(frame1, frame2, progress)
        elif transition_type == TransitionType.DISSOLVE:
            return SceneTransitions.dissolve_transition(frame1, frame2, progress)
        elif transition_type == TransitionType.WIPE_LEFT:
            return SceneTransitions.wipe_transition(frame1, frame2, progress, 'left')
        elif transition_type == TransitionType.WIPE_RIGHT:
            return SceneTransitions.wipe_transition(frame1, frame2, progress, 'right')
        elif transition_type == TransitionType.ZOOM_IN:
            return SceneTransitions.zoom_blur_transition(frame1, frame2, progress, zoom_in=True)
        elif transition_type == TransitionType.ZOOM_OUT:
            return SceneTransitions.zoom_blur_transition(frame1, frame2, progress, zoom_in=False)
        elif transition_type == TransitionType.BLUR_TRANSITION:
            return SceneTransitions.zoom_blur_transition(frame1, frame2, progress, zoom_in=True)
        elif transition_type == TransitionType.FLASH:
            return SceneTransitions.flash_transition(frame1, frame2, progress)
        elif transition_type == TransitionType.SLIDE_LEFT:
            return SceneTransitions.slide_transition(frame1, frame2, progress, 'left')
        elif transition_type == TransitionType.CIRCLE_REVEAL:
            return SceneTransitions.circle_reveal_transition(frame1, frame2, progress)
        elif transition_type == TransitionType.DIAGONAL_WIPE:
            return SceneTransitions.diagonal_wipe_transition(frame1, frame2, progress)
        else:
            return SceneTransitions.fade_transition(frame1, frame2, progress)

    def generate_frame(self, frame_num: int) -> Image.Image:
        """Generate a single frame with all effects and transitions"""
        # Check if we're in a transition
        in_transition, trans_progress, scene1, scene2 = self.is_in_transition(frame_num)

        if in_transition and scene1 and scene2:
            # Render both scenes and blend
            frame1 = self.render_scene_frame(scene1, 1.0, frame_num)
            frame2 = self.render_scene_frame(scene2, 0.0, frame_num)

            # Apply transition
            return self.apply_transition(frame1, frame2, trans_progress, scene2.transition_in)
        else:
            # Render current scene
            scene, progress = self.get_current_scene(frame_num)
            return self.render_scene_frame(scene, progress, frame_num)

    def generate_video(self, output_path: str, total_duration: int) -> bool:
        """Generate the complete video"""
        try:
            total_frames = total_duration * self.config.fps

            # Setup video writer
            temp_path = output_path.replace('.mp4', '_temp.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                temp_path, fourcc, self.config.fps,
                (self.config.width, self.config.height)
            )

            if not video_writer.isOpened():
                logger.error("Failed to open video writer")
                return False

            logger.info(f"Generating {total_frames} frames for {total_duration}s video...")

            for frame_num in range(total_frames):
                if frame_num % 60 == 0:
                    progress_pct = (frame_num / total_frames) * 100
                    logger.info(f"Progress: {progress_pct:.1f}% ({frame_num}/{total_frames} frames)")

                # Generate frame
                frame = self.generate_frame(frame_num)

                # Convert to OpenCV format
                if frame.mode == 'RGBA':
                    frame = frame.convert('RGB')
                frame_np = np.array(frame)
                frame_cv2 = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                video_writer.write(frame_cv2)

            video_writer.release()

            # Convert to web-compatible format
            success = self._convert_to_web_format(temp_path, output_path)

            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return success

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return False

    def _convert_to_web_format(self, input_path: str, output_path: str) -> bool:
        """Convert video to web-compatible H.264 format"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            logger.info(f"Video saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"FFmpeg conversion failed: {e}")
            return False


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_advanced_video(job_data: Dict) -> Optional[str]:
    """
    Main entry point for advanced video generation.

    Args:
        job_data: Dictionary containing job_id, script, images, style, duration

    Returns:
        Path to generated video or None if failed
    """
    try:
        job_id = job_data.get('job_id')
        script = job_data.get('script', {})
        images = job_data.get('images', [])
        style = job_data.get('style', 'professional')
        duration = job_data.get('duration', 30)

        logger.info(f"Starting advanced video generation for job {job_id}")
        logger.info(f"Style: {style}, Duration: {duration}s")

        # Create generator with config
        config = VideoConfig(
            width=1080,
            height=1920,
            fps=30,
            enable_particles=True,
            enable_ken_burns=True,
            enable_vignette=True,
            enable_film_grain=False
        )

        generator = AdvancedVideoGenerator(config)

        # Load product images
        generator.load_product_images(images)

        # Setup scenes
        generator.setup_scenes(script, style, duration)

        # Generate video
        os.makedirs("static/videos", exist_ok=True)
        output_path = f"static/videos/{job_id}.mp4"

        success = generator.generate_video(output_path, duration)

        if success:
            logger.info(f"Advanced video generated successfully: {output_path}")
            return f"/static/videos/{job_id}.mp4"
        else:
            logger.error("Advanced video generation failed")
            return None

    except Exception as e:
        logger.error(f"Error in generate_advanced_video: {e}")
        return None
