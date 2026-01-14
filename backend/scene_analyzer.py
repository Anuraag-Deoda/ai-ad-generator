"""
AI-Powered Scene Analysis Module

Analyzes product images and content to automatically determine:
- Optimal Ken Burns effects based on image composition
- Best color schemes based on product colors
- Focus points for dynamic effects
- Scene pacing recommendations
- Visual hierarchy optimization
"""

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import colorsys
import math


class CompositionType(Enum):
    CENTERED = "centered"
    RULE_OF_THIRDS = "rule_of_thirds"
    DIAGONAL = "diagonal"
    SYMMETRICAL = "symmetrical"
    ASYMMETRICAL = "asymmetrical"
    MINIMALIST = "minimalist"
    BUSY = "busy"


class ImageMood(Enum):
    BRIGHT = "bright"
    DARK = "dark"
    WARM = "warm"
    COOL = "cool"
    VIBRANT = "vibrant"
    MUTED = "muted"
    HIGH_CONTRAST = "high_contrast"
    LOW_CONTRAST = "low_contrast"


@dataclass
class FocusPoint:
    """Represents a point of interest in an image"""
    x: float  # 0-1 normalized
    y: float  # 0-1 normalized
    weight: float  # 0-1 importance
    type: str  # 'face', 'object', 'text', 'edge', 'center'


@dataclass
class ColorPalette:
    """Extracted color palette from image"""
    dominant: Tuple[int, int, int]
    secondary: List[Tuple[int, int, int]]
    accent: Tuple[int, int, int]
    background: Tuple[int, int, int]
    is_light: bool
    warmth: float  # -1 (cool) to 1 (warm)


@dataclass
class SceneAnalysis:
    """Complete analysis of a scene/image"""
    composition: CompositionType
    mood: ImageMood
    focus_points: List[FocusPoint]
    color_palette: ColorPalette
    recommended_ken_burns: str
    recommended_duration: float
    recommended_effects: List[str]
    visual_complexity: float  # 0-1
    brightness: float  # 0-1
    contrast: float  # 0-1
    saturation: float  # 0-1
    edge_density: float  # 0-1
    text_regions: List[Dict[str, Any]]


class SceneAnalyzer:
    """AI-powered scene and image analyzer"""

    def __init__(self):
        self.cascade_face = None
        try:
            self.cascade_face = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            pass

    def analyze_image(self, image: Image.Image) -> SceneAnalysis:
        """Perform complete analysis of an image"""
        # Convert to numpy for OpenCV operations
        img_np = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Run all analyses
        focus_points = self._find_focus_points(img_cv)
        color_palette = self._extract_color_palette(img_np)
        composition = self._analyze_composition(img_cv, focus_points)
        mood = self._analyze_mood(img_np, color_palette)

        brightness = self._calculate_brightness(img_cv)
        contrast = self._calculate_contrast(img_cv)
        saturation = self._calculate_saturation(img_np)
        edge_density = self._calculate_edge_density(img_cv)
        visual_complexity = self._calculate_visual_complexity(img_cv, edge_density)

        text_regions = self._detect_text_regions(img_cv)

        # Generate recommendations
        recommended_kb = self._recommend_ken_burns(composition, focus_points, visual_complexity)
        recommended_duration = self._recommend_duration(visual_complexity, len(text_regions))
        recommended_effects = self._recommend_effects(mood, color_palette, visual_complexity)

        return SceneAnalysis(
            composition=composition,
            mood=mood,
            focus_points=focus_points,
            color_palette=color_palette,
            recommended_ken_burns=recommended_kb,
            recommended_duration=recommended_duration,
            recommended_effects=recommended_effects,
            visual_complexity=visual_complexity,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            edge_density=edge_density,
            text_regions=text_regions
        )

    def _find_focus_points(self, img_cv: np.ndarray) -> List[FocusPoint]:
        """Find points of interest using multiple detection methods"""
        focus_points = []
        h, w = img_cv.shape[:2]

        # 1. Face detection
        if self.cascade_face is not None:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.cascade_face.detectMultiScale(gray, 1.1, 4)
            for (x, y, fw, fh) in faces:
                cx = (x + fw / 2) / w
                cy = (y + fh / 2) / h
                focus_points.append(FocusPoint(cx, cy, 0.95, 'face'))

        # 2. Edge-based saliency
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get largest contours as focus areas
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for i, contour in enumerate(sorted_contours):
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"] / w
                cy = M["m01"] / M["m00"] / h
                weight = 0.8 - (i * 0.1)
                focus_points.append(FocusPoint(cx, cy, weight, 'object'))

        # 3. Color saliency using LAB color space
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Find regions with high color saturation
        color_saliency = np.sqrt(a.astype(float)**2 + b.astype(float)**2)
        color_saliency = (color_saliency / color_saliency.max() * 255).astype(np.uint8)

        # Threshold and find center of mass
        _, thresh = cv2.threshold(color_saliency, 128, 255, cv2.THRESH_BINARY)
        M = cv2.moments(thresh)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"] / w
            cy = M["m01"] / M["m00"] / h
            focus_points.append(FocusPoint(cx, cy, 0.7, 'color'))

        # 4. Always add center as a fallback
        focus_points.append(FocusPoint(0.5, 0.5, 0.3, 'center'))

        # Deduplicate nearby points
        return self._deduplicate_focus_points(focus_points)

    def _deduplicate_focus_points(self, points: List[FocusPoint], threshold: float = 0.15) -> List[FocusPoint]:
        """Remove focus points that are too close together"""
        if not points:
            return []

        # Sort by weight (highest first)
        sorted_points = sorted(points, key=lambda p: p.weight, reverse=True)
        result = [sorted_points[0]]

        for point in sorted_points[1:]:
            is_duplicate = False
            for existing in result:
                dist = math.sqrt((point.x - existing.x)**2 + (point.y - existing.y)**2)
                if dist < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                result.append(point)

        return result[:5]  # Keep top 5

    def _extract_color_palette(self, img_np: np.ndarray) -> ColorPalette:
        """Extract dominant colors using k-means clustering"""
        # Resize for faster processing
        small = cv2.resize(img_np, (100, 100))
        pixels = small.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Count pixels per cluster
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_colors = [tuple(map(int, centers[i])) for i in sorted_indices]

        # Determine dominant, secondary, accent
        dominant = sorted_colors[0]
        secondary = sorted_colors[1:4] if len(sorted_colors) > 1 else [dominant]

        # Find most saturated color for accent
        accent = dominant
        max_saturation = 0
        for color in sorted_colors:
            r, g, b = color
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            if s > max_saturation:
                max_saturation = s
                accent = color

        # Determine background (darkest or lightest)
        brightness_values = [(c, sum(c)/3) for c in sorted_colors]
        brightness_values.sort(key=lambda x: x[1])
        background = brightness_values[0][0]  # Darkest

        # Calculate warmth
        avg_r = np.mean(pixels[:, 0])
        avg_b = np.mean(pixels[:, 2])
        warmth = (avg_r - avg_b) / 255  # -1 to 1

        # Is it light overall?
        is_light = np.mean(pixels) > 127

        return ColorPalette(
            dominant=dominant,
            secondary=list(secondary),
            accent=accent,
            background=background,
            is_light=is_light,
            warmth=warmth
        )

    def _analyze_composition(self, img_cv: np.ndarray, focus_points: List[FocusPoint]) -> CompositionType:
        """Analyze image composition type"""
        if not focus_points:
            return CompositionType.CENTERED

        # Get primary focus point
        primary = max(focus_points, key=lambda p: p.weight)

        # Check rule of thirds
        thirds_x = [1/3, 2/3]
        thirds_y = [1/3, 2/3]

        near_thirds = False
        for tx in thirds_x:
            for ty in thirds_y:
                if abs(primary.x - tx) < 0.1 and abs(primary.y - ty) < 0.1:
                    near_thirds = True
                    break

        if near_thirds:
            return CompositionType.RULE_OF_THIRDS

        # Check if centered
        if abs(primary.x - 0.5) < 0.15 and abs(primary.y - 0.5) < 0.15:
            return CompositionType.CENTERED

        # Check for diagonal composition
        diag1 = abs(primary.x - primary.y) < 0.2
        diag2 = abs(primary.x - (1 - primary.y)) < 0.2
        if diag1 or diag2:
            return CompositionType.DIAGONAL

        # Check symmetry
        h, w = img_cv.shape[:2]
        left_half = img_cv[:, :w//2]
        right_half = cv2.flip(img_cv[:, w//2:], 1)

        # Resize to same size for comparison
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]

        diff = cv2.absdiff(left_half, right_half)
        symmetry_score = 1 - (np.mean(diff) / 255)

        if symmetry_score > 0.7:
            return CompositionType.SYMMETRICAL

        return CompositionType.ASYMMETRICAL

    def _analyze_mood(self, img_np: np.ndarray, palette: ColorPalette) -> ImageMood:
        """Determine the mood/tone of the image"""
        brightness = np.mean(img_np) / 255

        if brightness > 0.65:
            if palette.warmth > 0.2:
                return ImageMood.BRIGHT
            return ImageMood.COOL
        elif brightness < 0.35:
            return ImageMood.DARK

        if palette.warmth > 0.3:
            return ImageMood.WARM
        elif palette.warmth < -0.3:
            return ImageMood.COOL

        # Check saturation for vibrant/muted
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv[:, :, 1]) / 255

        if avg_saturation > 0.5:
            return ImageMood.VIBRANT
        elif avg_saturation < 0.2:
            return ImageMood.MUTED

        return ImageMood.HIGH_CONTRAST

    def _calculate_brightness(self, img_cv: np.ndarray) -> float:
        """Calculate overall brightness 0-1"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) / 255

    def _calculate_contrast(self, img_cv: np.ndarray) -> float:
        """Calculate contrast level 0-1"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return np.std(gray) / 128  # Normalize to 0-1 range

    def _calculate_saturation(self, img_np: np.ndarray) -> float:
        """Calculate average saturation 0-1"""
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        return np.mean(hsv[:, :, 1]) / 255

    def _calculate_edge_density(self, img_cv: np.ndarray) -> float:
        """Calculate density of edges 0-1"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size

    def _calculate_visual_complexity(self, img_cv: np.ndarray, edge_density: float) -> float:
        """Calculate overall visual complexity 0-1"""
        # Combine edge density with color variance
        colors = img_cv.reshape(-1, 3)
        color_variance = np.mean(np.std(colors, axis=0)) / 128

        return min(1.0, (edge_density + color_variance) / 2)

    def _detect_text_regions(self, img_cv: np.ndarray) -> List[Dict[str, Any]]:
        """Detect regions that might contain text"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Use MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        text_regions = []
        for region in regions[:10]:  # Limit to 10 regions
            x, y, rw, rh = cv2.boundingRect(region)
            aspect_ratio = rw / max(rh, 1)

            # Text-like regions have certain aspect ratios
            if 0.1 < aspect_ratio < 10 and rw > 20 and rh > 10:
                text_regions.append({
                    'x': x / w,
                    'y': y / h,
                    'width': rw / w,
                    'height': rh / h,
                    'area': (rw * rh) / (w * h)
                })

        return text_regions

    def _recommend_ken_burns(self, composition: CompositionType,
                           focus_points: List[FocusPoint],
                           complexity: float) -> str:
        """Recommend Ken Burns effect based on analysis"""

        if not focus_points:
            return 'zoom_in_slow'

        primary = max(focus_points, key=lambda p: p.weight)

        # Simple images benefit from more dramatic movement
        if complexity < 0.3:
            if composition == CompositionType.CENTERED:
                return 'zoom_in_slow'
            elif composition == CompositionType.RULE_OF_THIRDS:
                return 'drift'
            else:
                return 'orbit'

        # Complex images need subtler movement
        if complexity > 0.6:
            return 'slow_zoom'

        # Based on focus point position
        if primary.x < 0.3:
            return 'pan_right'
        elif primary.x > 0.7:
            return 'pan_left'
        elif primary.y < 0.3:
            return 'tilt_down'
        elif primary.y > 0.7:
            return 'tilt_up'

        # Default based on composition
        composition_effects = {
            CompositionType.CENTERED: 'zoom_in_slow',
            CompositionType.RULE_OF_THIRDS: 'drift',
            CompositionType.DIAGONAL: 'orbit',
            CompositionType.SYMMETRICAL: 'zoom_out',
            CompositionType.ASYMMETRICAL: 'spiral_in',
        }

        return composition_effects.get(composition, 'zoom_in_slow')

    def _recommend_duration(self, complexity: float, text_region_count: int) -> float:
        """Recommend scene duration in seconds"""
        base_duration = 3.0

        # More complex images need more time
        base_duration += complexity * 2.0

        # Text needs reading time
        base_duration += text_region_count * 0.5

        return min(8.0, max(2.0, base_duration))

    def _recommend_effects(self, mood: ImageMood, palette: ColorPalette,
                          complexity: float) -> List[str]:
        """Recommend visual effects based on image analysis"""
        effects = []

        # Mood-based recommendations
        mood_effects = {
            ImageMood.BRIGHT: ['glow', 'lens_flare', 'sparkles'],
            ImageMood.DARK: ['vignette', 'film_grain', 'contrast_boost'],
            ImageMood.WARM: ['warm_filter', 'soft_glow', 'golden_hour'],
            ImageMood.COOL: ['cool_filter', 'blue_tint', 'crisp_edges'],
            ImageMood.VIBRANT: ['saturation_boost', 'color_pop', 'neon_glow'],
            ImageMood.MUTED: ['vintage', 'desaturate', 'soft_focus'],
            ImageMood.HIGH_CONTRAST: ['sharpen', 'clarity', 'dramatic'],
            ImageMood.LOW_CONTRAST: ['contrast_boost', 'levels_adjust'],
        }

        effects.extend(mood_effects.get(mood, [])[:2])

        # Add particle effects for low complexity images
        if complexity < 0.4:
            effects.append('particles')

        # Light images can have lens flare
        if palette.is_light:
            effects.append('subtle_flare')

        # Warm images benefit from golden particles
        if palette.warmth > 0.2:
            effects.append('golden_dust')

        return list(set(effects))[:4]


class SmartSceneComposer:
    """Intelligently compose scenes based on analysis"""

    def __init__(self):
        self.analyzer = SceneAnalyzer()

    def compose_scenes(self, images: List[Image.Image],
                      script: Dict[str, Any],
                      duration: int) -> List[Dict[str, Any]]:
        """Create optimized scene composition"""

        analyses = [self.analyzer.analyze_image(img) for img in images]

        scenes = []
        scene_types = ['hook', 'pitch', 'features', 'cta']

        # Calculate time allocation
        time_weights = {'hook': 0.2, 'pitch': 0.3, 'features': 0.3, 'cta': 0.2}

        for i, scene_type in enumerate(scene_types):
            scene_duration = duration * time_weights[scene_type]

            # Select best image for this scene
            if i < len(analyses):
                analysis = analyses[i % len(analyses)]
                image_index = i % len(images)
            else:
                analysis = analyses[0]
                image_index = 0

            scene = {
                'type': scene_type,
                'duration': scene_duration,
                'image_index': image_index,
                'ken_burns': analysis.recommended_ken_burns,
                'effects': analysis.recommended_effects,
                'focus_point': (
                    analysis.focus_points[0].x if analysis.focus_points else 0.5,
                    analysis.focus_points[0].y if analysis.focus_points else 0.5
                ),
                'color_scheme': {
                    'primary': analysis.color_palette.dominant,
                    'secondary': analysis.color_palette.secondary[0] if analysis.color_palette.secondary else analysis.color_palette.dominant,
                    'accent': analysis.color_palette.accent,
                    'is_light': analysis.color_palette.is_light
                },
                'composition': analysis.composition.value,
                'mood': analysis.mood.value,
                'complexity': analysis.visual_complexity
            }

            scenes.append(scene)

        return self._optimize_scene_flow(scenes)

    def _optimize_scene_flow(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize transitions between scenes"""

        for i in range(len(scenes) - 1):
            current = scenes[i]
            next_scene = scenes[i + 1]

            # Determine transition based on complexity change
            complexity_diff = abs(current['complexity'] - next_scene['complexity'])

            if complexity_diff > 0.3:
                current['transition'] = 'wipe'
            elif current['mood'] != next_scene['mood']:
                current['transition'] = 'dissolve'
            else:
                current['transition'] = 'fade'

            # Set transition duration based on complexity
            current['transition_duration'] = 0.5 + (complexity_diff * 0.5)

        # Last scene has no transition
        if scenes:
            scenes[-1]['transition'] = None
            scenes[-1]['transition_duration'] = 0

        return scenes


# Convenience functions
def analyze_product_images(images: List[Image.Image]) -> List[SceneAnalysis]:
    """Analyze multiple product images"""
    analyzer = SceneAnalyzer()
    return [analyzer.analyze_image(img) for img in images]


def get_smart_scene_composition(images: List[Image.Image],
                               script: Dict[str, Any],
                               duration: int = 30) -> List[Dict[str, Any]]:
    """Get AI-optimized scene composition"""
    composer = SmartSceneComposer()
    return composer.compose_scenes(images, script, duration)


def extract_color_theme(image: Image.Image) -> Dict[str, Any]:
    """Extract color theme from single image"""
    analyzer = SceneAnalyzer()
    analysis = analyzer.analyze_image(image)

    return {
        'dominant': analysis.color_palette.dominant,
        'secondary': analysis.color_palette.secondary,
        'accent': analysis.color_palette.accent,
        'background': analysis.color_palette.background,
        'is_light': analysis.color_palette.is_light,
        'warmth': analysis.color_palette.warmth,
        'mood': analysis.mood.value
    }
