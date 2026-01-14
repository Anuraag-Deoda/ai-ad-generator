"""
Industry-Specific Templates
Pre-built video templates optimized for different industries and use cases
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class TemplateCategory(Enum):
    """Categories of templates"""
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    FOOD = "food"
    FASHION = "fashion"
    TECH = "tech"
    SERVICES = "services"
    FITNESS = "fitness"
    BEAUTY = "beauty"


@dataclass
class SceneConfig:
    """Configuration for a single scene in a template"""
    type: str  # hero, price_reveal, features, cta, etc.
    duration_percent: float  # Percentage of total duration
    layout: str = "centered"
    ken_burns: str = "zoom_in_center"
    text_animation: str = "fade_in"
    particles: Optional[str] = None
    transition_in: str = "fade"
    transition_out: str = "fade"
    dynamic_content: Optional[str] = None  # price_animation, countdown, rating, cta
    filter: Optional[str] = None
    text_position: str = "center"  # center, top, bottom
    show_product: bool = True


@dataclass
class TemplateConfig:
    """Complete template configuration"""
    name: str
    description: str
    category: TemplateCategory
    scenes: List[SceneConfig]
    color_scheme: str = "default"
    color_grading: str = "none"
    default_particles: Optional[str] = None
    vignette_intensity: float = 0.3
    text_style: str = "modern"
    recommended_duration: int = 30


# Industry Templates Database
INDUSTRY_TEMPLATES: Dict[str, Dict[str, TemplateConfig]] = {
    "ecommerce": {
        "product_showcase": TemplateConfig(
            name="Product Showcase",
            description="Clean product presentation with price reveal",
            category=TemplateCategory.ECOMMERCE,
            scenes=[
                SceneConfig(
                    type="hero",
                    duration_percent=0.25,
                    layout="centered",
                    ken_burns="zoom_in_center",
                    text_animation="scale_up",
                    particles="sparkles",
                    text_position="bottom"
                ),
                SceneConfig(
                    type="price_reveal",
                    duration_percent=0.25,
                    layout="split_horizontal",
                    ken_burns="pull_back_reveal",
                    dynamic_content="price_animation",
                    particles="confetti"
                ),
                SceneConfig(
                    type="features",
                    duration_percent=0.25,
                    layout="floating_cards",
                    ken_burns="pan_left_to_right",
                    text_animation="slide_up"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.25,
                    layout="centered",
                    ken_burns="dramatic_push",
                    dynamic_content="cta_button",
                    particles="energy"
                )
            ],
            color_scheme="vibrant",
            color_grading="vibrant_pop",
            recommended_duration=30
        ),

        "flash_sale": TemplateConfig(
            name="Flash Sale",
            description="Urgent sale promotion with countdown",
            category=TemplateCategory.ECOMMERCE,
            scenes=[
                SceneConfig(
                    type="hook",
                    duration_percent=0.20,
                    layout="fullscreen",
                    ken_burns="zoom_in_center",
                    text_animation="glitch",
                    particles="fire",
                    filter="high_contrast"
                ),
                SceneConfig(
                    type="product",
                    duration_percent=0.25,
                    layout="centered",
                    ken_burns="dramatic_push",
                    dynamic_content="price_animation"
                ),
                SceneConfig(
                    type="countdown",
                    duration_percent=0.30,
                    layout="split_horizontal",
                    ken_burns="zoom_out_center",
                    dynamic_content="countdown_timer",
                    particles="confetti"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.25,
                    layout="centered",
                    ken_burns="zoom_in_center",
                    dynamic_content="cta_button",
                    text_animation="bounce_in"
                )
            ],
            color_scheme="urgent_red",
            color_grading="high_contrast",
            default_particles="fire",
            vignette_intensity=0.4,
            recommended_duration=15
        ),

        "before_after": TemplateConfig(
            name="Before & After",
            description="Transformation showcase",
            category=TemplateCategory.ECOMMERCE,
            scenes=[
                SceneConfig(
                    type="problem",
                    duration_percent=0.30,
                    layout="split_horizontal",
                    ken_burns="pan_left_to_right",
                    filter="desaturated",
                    text_animation="fade_in",
                    text_position="top"
                ),
                SceneConfig(
                    type="solution",
                    duration_percent=0.35,
                    layout="fullscreen",
                    ken_burns="zoom_in_center",
                    transition_in="wipe_right",
                    particles="sparkles",
                    filter="vibrant"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.35,
                    layout="centered",
                    ken_burns="pull_back_reveal",
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="contrast",
            recommended_duration=30
        ),

        "social_proof": TemplateConfig(
            name="Social Proof",
            description="Reviews and ratings focused",
            category=TemplateCategory.ECOMMERCE,
            scenes=[
                SceneConfig(
                    type="hero",
                    duration_percent=0.20,
                    layout="centered",
                    ken_burns="zoom_in_center"
                ),
                SceneConfig(
                    type="rating",
                    duration_percent=0.25,
                    layout="split_horizontal",
                    dynamic_content="star_rating",
                    particles="sparkles"
                ),
                SceneConfig(
                    type="review",
                    duration_percent=0.30,
                    layout="fullscreen",
                    dynamic_content="review_quote",
                    show_product=False
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.25,
                    layout="centered",
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="trust",
            recommended_duration=30
        )
    },

    "saas": {
        "feature_highlight": TemplateConfig(
            name="Feature Highlight",
            description="Showcase key product features",
            category=TemplateCategory.SAAS,
            scenes=[
                SceneConfig(
                    type="problem_statement",
                    duration_percent=0.25,
                    layout="minimal",
                    ken_burns="zoom_in_center",
                    text_animation="typewriter",
                    filter="desaturated"
                ),
                SceneConfig(
                    type="solution_intro",
                    duration_percent=0.25,
                    layout="centered",
                    ken_burns="pull_back_reveal",
                    text_animation="slide_up",
                    particles="floating"
                ),
                SceneConfig(
                    type="feature_demo",
                    duration_percent=0.30,
                    layout="split_vertical",
                    ken_burns="pan_left_to_right",
                    text_animation="word_by_word"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    layout="centered",
                    dynamic_content="cta_button",
                    text_animation="fade_in"
                )
            ],
            color_scheme="tech_gradient",
            color_grading="cool_blue",
            text_style="clean",
            recommended_duration=30
        ),

        "comparison": TemplateConfig(
            name="Comparison",
            description="Before/after or vs competitor",
            category=TemplateCategory.SAAS,
            scenes=[
                SceneConfig(
                    type="before",
                    duration_percent=0.30,
                    layout="split_vertical",
                    filter="dull",
                    text_animation="fade_in",
                    text_position="top"
                ),
                SceneConfig(
                    type="after",
                    duration_percent=0.35,
                    layout="fullscreen",
                    filter="vibrant",
                    ken_burns="zoom_in_center",
                    particles="energy",
                    transition_in="wipe_right"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.35,
                    layout="centered",
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="professional",
            recommended_duration=30
        ),

        "demo_style": TemplateConfig(
            name="Demo Style",
            description="Product demo walkthrough",
            category=TemplateCategory.SAAS,
            scenes=[
                SceneConfig(
                    type="intro",
                    duration_percent=0.20,
                    layout="centered",
                    ken_burns="zoom_out_center",
                    text_animation="scale_up"
                ),
                SceneConfig(
                    type="step_1",
                    duration_percent=0.20,
                    layout="split_horizontal",
                    ken_burns="pan_left_to_right",
                    text_animation="slide_up"
                ),
                SceneConfig(
                    type="step_2",
                    duration_percent=0.20,
                    layout="split_horizontal",
                    ken_burns="pan_right_to_left",
                    text_animation="slide_up"
                ),
                SceneConfig(
                    type="result",
                    duration_percent=0.20,
                    layout="fullscreen",
                    ken_burns="dramatic_push",
                    particles="sparkles"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    layout="centered",
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="minimal",
            text_style="clean",
            recommended_duration=45
        )
    },

    "food": {
        "sizzle_reel": TemplateConfig(
            name="Sizzle Reel",
            description="Appetizing food showcase",
            category=TemplateCategory.FOOD,
            scenes=[
                SceneConfig(
                    type="hero_shot",
                    duration_percent=0.30,
                    layout="fullscreen",
                    ken_burns="zoom_in_center",
                    filter="warm",
                    particles="steam"
                ),
                SceneConfig(
                    type="detail_shots",
                    duration_percent=0.30,
                    layout="floating_cards",
                    ken_burns="pan_left_to_right",
                    transition_in="dissolve",
                    filter="warm"
                ),
                SceneConfig(
                    type="plating",
                    duration_percent=0.20,
                    layout="centered",
                    ken_burns="pull_back_reveal",
                    particles="sparkles"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    layout="centered",
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="warm",
            color_grading="warm_appetite",
            vignette_intensity=0.35,
            recommended_duration=15
        ),

        "menu_item": TemplateConfig(
            name="Menu Item",
            description="Single dish highlight with price",
            category=TemplateCategory.FOOD,
            scenes=[
                SceneConfig(
                    type="reveal",
                    duration_percent=0.35,
                    layout="centered",
                    ken_burns="zoom_in_center",
                    filter="warm",
                    particles="steam"
                ),
                SceneConfig(
                    type="price",
                    duration_percent=0.30,
                    layout="split_horizontal",
                    dynamic_content="price_animation"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.35,
                    layout="centered",
                    dynamic_content="cta_button"
                )
            ],
            color_grading="golden_hour",
            recommended_duration=15
        )
    },

    "fashion": {
        "lookbook": TemplateConfig(
            name="Lookbook",
            description="Editorial fashion showcase",
            category=TemplateCategory.FASHION,
            scenes=[
                SceneConfig(
                    type="hero",
                    duration_percent=0.30,
                    layout="fullscreen",
                    ken_burns="dramatic_push",
                    filter="editorial",
                    particles="sparkles"
                ),
                SceneConfig(
                    type="details",
                    duration_percent=0.25,
                    layout="floating_cards",
                    ken_burns="pan_left_to_right",
                    text_animation="fade_in"
                ),
                SceneConfig(
                    type="lifestyle",
                    duration_percent=0.25,
                    layout="cinematic",
                    ken_burns="pull_back_reveal",
                    filter="editorial"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    layout="centered",
                    dynamic_content="cta_button",
                    text_animation="slide_up"
                )
            ],
            color_scheme="minimal",
            color_grading="muted_pastel",
            text_style="elegant",
            recommended_duration=30
        ),

        "new_arrival": TemplateConfig(
            name="New Arrival",
            description="Product launch announcement",
            category=TemplateCategory.FASHION,
            scenes=[
                SceneConfig(
                    type="teaser",
                    duration_percent=0.20,
                    layout="cinematic",
                    filter="high_contrast",
                    text_animation="glitch"
                ),
                SceneConfig(
                    type="reveal",
                    duration_percent=0.35,
                    layout="fullscreen",
                    ken_burns="zoom_in_center",
                    particles="sparkles",
                    transition_in="flash"
                ),
                SceneConfig(
                    type="price",
                    duration_percent=0.25,
                    dynamic_content="price_animation"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    dynamic_content="cta_button"
                )
            ],
            color_grading="noir",
            recommended_duration=15
        )
    },

    "tech": {
        "unboxing": TemplateConfig(
            name="Unboxing",
            description="Product reveal experience",
            category=TemplateCategory.TECH,
            scenes=[
                SceneConfig(
                    type="box_reveal",
                    duration_percent=0.25,
                    layout="centered",
                    ken_burns="zoom_in_center",
                    transition_in="zoom_blur",
                    particles="energy"
                ),
                SceneConfig(
                    type="product_hero",
                    duration_percent=0.30,
                    layout="fullscreen",
                    ken_burns="orbit",
                    particles="sparkles"
                ),
                SceneConfig(
                    type="specs",
                    duration_percent=0.25,
                    layout="split_vertical",
                    text_animation="typewriter"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="dark_tech",
            color_grading="cyberpunk",
            default_particles="energy",
            recommended_duration=30
        ),

        "specs_reveal": TemplateConfig(
            name="Specs Reveal",
            description="Technical specifications showcase",
            category=TemplateCategory.TECH,
            scenes=[
                SceneConfig(
                    type="hero",
                    duration_percent=0.25,
                    layout="centered",
                    ken_burns="zoom_in_center"
                ),
                SceneConfig(
                    type="spec_1",
                    duration_percent=0.20,
                    layout="split_horizontal",
                    text_animation="slide_up"
                ),
                SceneConfig(
                    type="spec_2",
                    duration_percent=0.20,
                    layout="split_horizontal",
                    text_animation="slide_up"
                ),
                SceneConfig(
                    type="spec_3",
                    duration_percent=0.15,
                    layout="split_horizontal",
                    text_animation="slide_up"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="dark_tech",
            text_style="technical",
            recommended_duration=30
        )
    },

    "fitness": {
        "transformation": TemplateConfig(
            name="Transformation",
            description="Before/after fitness results",
            category=TemplateCategory.FITNESS,
            scenes=[
                SceneConfig(
                    type="before",
                    duration_percent=0.25,
                    layout="split_horizontal",
                    filter="desaturated",
                    text_position="top"
                ),
                SceneConfig(
                    type="journey",
                    duration_percent=0.25,
                    layout="floating_cards",
                    ken_burns="pan_left_to_right"
                ),
                SceneConfig(
                    type="after",
                    duration_percent=0.30,
                    layout="fullscreen",
                    ken_burns="dramatic_push",
                    particles="energy",
                    filter="vibrant"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="energetic",
            color_grading="vibrant_pop",
            recommended_duration=30
        ),

        "workout_promo": TemplateConfig(
            name="Workout Promo",
            description="Gym/program promotion",
            category=TemplateCategory.FITNESS,
            scenes=[
                SceneConfig(
                    type="hook",
                    duration_percent=0.20,
                    layout="fullscreen",
                    ken_burns="zoom_in_center",
                    text_animation="glitch",
                    particles="fire"
                ),
                SceneConfig(
                    type="benefits",
                    duration_percent=0.35,
                    layout="floating_cards",
                    text_animation="bounce_in"
                ),
                SceneConfig(
                    type="social_proof",
                    duration_percent=0.25,
                    dynamic_content="star_rating"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    dynamic_content="cta_button",
                    text_animation="shake"
                )
            ],
            color_scheme="bold",
            default_particles="energy",
            recommended_duration=15
        )
    },

    "beauty": {
        "product_reveal": TemplateConfig(
            name="Product Reveal",
            description="Elegant beauty product showcase",
            category=TemplateCategory.BEAUTY,
            scenes=[
                SceneConfig(
                    type="teaser",
                    duration_percent=0.20,
                    layout="cinematic",
                    filter="soft_glow",
                    particles="sparkles"
                ),
                SceneConfig(
                    type="reveal",
                    duration_percent=0.30,
                    layout="centered",
                    ken_burns="zoom_in_center",
                    particles="floating"
                ),
                SceneConfig(
                    type="benefits",
                    duration_percent=0.30,
                    layout="floating_cards",
                    text_animation="fade_in"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.20,
                    dynamic_content="cta_button"
                )
            ],
            color_scheme="elegant",
            color_grading="muted_pastel",
            text_style="elegant",
            recommended_duration=30
        ),

        "skincare_routine": TemplateConfig(
            name="Skincare Routine",
            description="Step-by-step product routine",
            category=TemplateCategory.BEAUTY,
            scenes=[
                SceneConfig(
                    type="intro",
                    duration_percent=0.15,
                    layout="centered",
                    text_animation="fade_in"
                ),
                SceneConfig(
                    type="step_1",
                    duration_percent=0.20,
                    layout="split_horizontal",
                    ken_burns="pan_left_to_right"
                ),
                SceneConfig(
                    type="step_2",
                    duration_percent=0.20,
                    layout="split_horizontal",
                    ken_burns="pan_right_to_left"
                ),
                SceneConfig(
                    type="step_3",
                    duration_percent=0.20,
                    layout="split_horizontal",
                    ken_burns="pan_left_to_right"
                ),
                SceneConfig(
                    type="result",
                    duration_percent=0.15,
                    layout="fullscreen",
                    particles="sparkles"
                ),
                SceneConfig(
                    type="cta",
                    duration_percent=0.10,
                    dynamic_content="cta_button"
                )
            ],
            color_grading="soft_glow",
            recommended_duration=45
        )
    }
}


# Color Schemes
COLOR_SCHEMES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "default": {
        "primary": (13, 110, 253),
        "secondary": (108, 117, 125),
        "accent": (255, 193, 7),
        "background_start": (30, 30, 50),
        "background_end": (10, 10, 30)
    },
    "vibrant": {
        "primary": (255, 87, 51),
        "secondary": (255, 195, 0),
        "accent": (0, 230, 118),
        "background_start": (45, 15, 60),
        "background_end": (15, 5, 30)
    },
    "urgent_red": {
        "primary": (220, 53, 69),
        "secondary": (255, 193, 7),
        "accent": (255, 255, 255),
        "background_start": (80, 20, 20),
        "background_end": (30, 10, 10)
    },
    "tech_gradient": {
        "primary": (0, 212, 255),
        "secondary": (147, 51, 234),
        "accent": (16, 185, 129),
        "background_start": (15, 23, 42),
        "background_end": (30, 41, 59)
    },
    "minimal": {
        "primary": (30, 30, 30),
        "secondary": (100, 100, 100),
        "accent": (200, 200, 200),
        "background_start": (250, 250, 250),
        "background_end": (230, 230, 230)
    },
    "warm": {
        "primary": (255, 138, 76),
        "secondary": (255, 195, 113),
        "accent": (255, 87, 51),
        "background_start": (50, 30, 20),
        "background_end": (30, 15, 10)
    },
    "dark_tech": {
        "primary": (0, 255, 136),
        "secondary": (0, 212, 255),
        "accent": (255, 0, 128),
        "background_start": (10, 10, 15),
        "background_end": (5, 5, 10)
    },
    "elegant": {
        "primary": (212, 175, 55),
        "secondary": (180, 150, 100),
        "accent": (255, 215, 0),
        "background_start": (40, 35, 30),
        "background_end": (20, 18, 15)
    },
    "professional": {
        "primary": (0, 82, 155),
        "secondary": (41, 128, 185),
        "accent": (46, 204, 113),
        "background_start": (44, 62, 80),
        "background_end": (30, 40, 50)
    },
    "bold": {
        "primary": (255, 87, 51),
        "secondary": (255, 195, 0),
        "accent": (138, 43, 226),
        "background_start": (20, 20, 40),
        "background_end": (10, 10, 20)
    },
    "trust": {
        "primary": (46, 125, 50),
        "secondary": (76, 175, 80),
        "accent": (255, 193, 7),
        "background_start": (30, 50, 35),
        "background_end": (15, 30, 20)
    },
    "contrast": {
        "primary": (255, 255, 255),
        "secondary": (200, 200, 200),
        "accent": (255, 87, 51),
        "background_start": (0, 0, 0),
        "background_end": (30, 30, 30)
    },
    "energetic": {
        "primary": (255, 0, 102),
        "secondary": (255, 102, 0),
        "accent": (0, 255, 255),
        "background_start": (40, 10, 50),
        "background_end": (20, 5, 30)
    }
}


class IndustryTemplateRenderer:
    """Applies industry-specific templates to video generation"""

    def __init__(self):
        self.templates = INDUSTRY_TEMPLATES
        self.color_schemes = COLOR_SCHEMES

    def get_industries(self) -> List[str]:
        """Get list of available industries"""
        return list(self.templates.keys())

    def get_templates_for_industry(self, industry: str) -> List[Dict[str, str]]:
        """Get available templates for an industry"""
        if industry not in self.templates:
            return []

        return [
            {
                "id": template_id,
                "name": config.name,
                "description": config.description,
                "recommended_duration": config.recommended_duration
            }
            for template_id, config in self.templates[industry].items()
        ]

    def get_template(self, industry: str, template_id: str) -> Optional[TemplateConfig]:
        """Get a specific template configuration"""
        if industry not in self.templates:
            return None
        return self.templates[industry].get(template_id)

    def get_color_scheme(self, scheme_name: str) -> Dict[str, Tuple[int, int, int]]:
        """Get colors for a scheme"""
        return self.color_schemes.get(scheme_name, self.color_schemes["default"])

    def apply_template_to_config(
        self,
        template: TemplateConfig,
        duration: int,
        fps: int = 30
    ) -> Dict[str, Any]:
        """
        Convert template to video generation config

        Returns a dict with scene configurations ready for video generator
        """
        total_frames = duration * fps
        scenes_config = []
        current_frame = 0

        for scene in template.scenes:
            scene_frames = int(total_frames * scene.duration_percent)
            end_frame = current_frame + scene_frames

            scenes_config.append({
                "type": scene.type,
                "start_frame": current_frame,
                "end_frame": end_frame,
                "duration": scene_frames / fps,
                "layout": scene.layout,
                "ken_burns": scene.ken_burns,
                "text_animation": scene.text_animation,
                "particles": scene.particles or template.default_particles,
                "transition_in": scene.transition_in,
                "transition_out": scene.transition_out,
                "dynamic_content": scene.dynamic_content,
                "filter": scene.filter,
                "text_position": scene.text_position,
                "show_product": scene.show_product
            })

            current_frame = end_frame

        color_scheme = self.get_color_scheme(template.color_scheme)

        return {
            "scenes": scenes_config,
            "color_scheme": color_scheme,
            "color_grading": template.color_grading,
            "vignette_intensity": template.vignette_intensity,
            "text_style": template.text_style,
            "total_frames": total_frames,
            "fps": fps
        }

    def get_scene_text_mapping(self, template: TemplateConfig, script: Dict) -> Dict[str, str]:
        """
        Map script sections to template scenes

        Args:
            template: The template configuration
            script: The generated script with hook, pitch, features, cta

        Returns:
            Dict mapping scene types to text content
        """
        mapping = {}

        for scene in template.scenes:
            scene_type = scene.type

            if scene_type in ["hook", "teaser", "problem_statement", "problem"]:
                mapping[scene_type] = script.get("hook", "")
            elif scene_type in ["hero", "hero_shot", "product", "reveal", "product_hero", "box_reveal"]:
                mapping[scene_type] = script.get("pitch", "")
            elif scene_type in ["features", "benefits", "detail_shots", "details", "lifestyle"]:
                features = script.get("features", [])
                mapping[scene_type] = " | ".join(features[:3]) if features else ""
            elif scene_type in ["cta", "result"]:
                mapping[scene_type] = script.get("cta", "Shop Now")
            elif scene_type in ["price_reveal", "price", "countdown"]:
                mapping[scene_type] = ""  # Dynamic content handles this
            elif scene_type in ["rating", "review", "social_proof"]:
                mapping[scene_type] = ""  # Dynamic content handles this
            elif scene_type.startswith("step_") or scene_type.startswith("spec_"):
                # For multi-step templates, distribute features
                features = script.get("features", [])
                step_num = int(scene_type.split("_")[1]) - 1
                if step_num < len(features):
                    mapping[scene_type] = features[step_num]
                else:
                    mapping[scene_type] = ""
            else:
                # Default to pitch for unknown scene types
                mapping[scene_type] = script.get("pitch", "")

        return mapping


# Global instance
template_renderer = IndustryTemplateRenderer()


def get_template(industry: str, template_id: str) -> Optional[TemplateConfig]:
    """Convenience function to get a template"""
    return template_renderer.get_template(industry, template_id)


def list_templates(industry: str) -> List[Dict[str, str]]:
    """Convenience function to list templates for an industry"""
    return template_renderer.get_templates_for_industry(industry)


def list_industries() -> List[str]:
    """Convenience function to list all industries"""
    return template_renderer.get_industries()


def apply_template(template: TemplateConfig, duration: int, fps: int = 30) -> Dict[str, Any]:
    """Convenience function to apply a template"""
    return template_renderer.apply_template_to_config(template, duration, fps)
