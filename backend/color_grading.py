"""
Professional Color Grading Module

Cinema-quality color grading effects:
- LUT-style color transformations
- Film emulation presets
- Color correction tools
- HDR tone mapping
- Split toning
- Selective color adjustments
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any
from enum import Enum
import math


class ColorGradePreset(Enum):
    # Cinematic
    CINEMATIC_TEAL_ORANGE = "cinematic_teal_orange"
    CINEMATIC_BLOCKBUSTER = "cinematic_blockbuster"
    CINEMATIC_NOIR = "cinematic_noir"
    CINEMATIC_WARM = "cinematic_warm"
    CINEMATIC_COLD = "cinematic_cold"

    # Film Emulation
    FILM_KODAK_PORTRA = "film_kodak_portra"
    FILM_FUJI_VELVIA = "film_fuji_velvia"
    FILM_KODAK_EKTAR = "film_kodak_ektar"
    FILM_ILFORD_BW = "film_ilford_bw"

    # Modern/Social
    INSTAGRAM_CLARENDON = "instagram_clarendon"
    INSTAGRAM_VALENCIA = "instagram_valencia"
    VSCO_A6 = "vsco_a6"
    VINTAGE_FADED = "vintage_faded"

    # Product/Commercial
    PRODUCT_CLEAN = "product_clean"
    PRODUCT_VIBRANT = "product_vibrant"
    PRODUCT_LUXURY = "product_luxury"
    FOOD_WARM = "food_warm"

    # Mood
    MOODY_DARK = "moody_dark"
    BRIGHT_AIRY = "bright_airy"
    GOLDEN_HOUR = "golden_hour"
    BLUE_HOUR = "blue_hour"


@dataclass
class ColorGradeSettings:
    """Settings for color grading"""
    # Basic adjustments
    exposure: float = 0.0  # -2 to 2
    contrast: float = 0.0  # -1 to 1
    highlights: float = 0.0  # -1 to 1
    shadows: float = 0.0  # -1 to 1
    whites: float = 0.0  # -1 to 1
    blacks: float = 0.0  # -1 to 1

    # Color adjustments
    temperature: float = 0.0  # -1 (cool) to 1 (warm)
    tint: float = 0.0  # -1 (green) to 1 (magenta)
    vibrance: float = 0.0  # -1 to 1
    saturation: float = 0.0  # -1 to 1

    # Split toning
    highlight_hue: int = 45  # 0-360
    highlight_saturation: float = 0.0  # 0-1
    shadow_hue: int = 220  # 0-360
    shadow_saturation: float = 0.0  # 0-1
    balance: float = 0.0  # -1 to 1

    # Advanced
    clarity: float = 0.0  # -1 to 1
    dehaze: float = 0.0  # -1 to 1
    vignette: float = 0.0  # 0 to 1
    grain: float = 0.0  # 0 to 1
    fade: float = 0.0  # 0 to 1

    # Color curves (optional)
    red_curve: Optional[List[Tuple[int, int]]] = None
    green_curve: Optional[List[Tuple[int, int]]] = None
    blue_curve: Optional[List[Tuple[int, int]]] = None
    luma_curve: Optional[List[Tuple[int, int]]] = None


class ColorGrader:
    """Professional color grading engine"""

    # Preset definitions
    PRESETS: Dict[ColorGradePreset, ColorGradeSettings] = {
        ColorGradePreset.CINEMATIC_TEAL_ORANGE: ColorGradeSettings(
            contrast=0.15, shadows=-0.1, highlights=-0.05,
            temperature=0.1, saturation=-0.1,
            highlight_hue=35, highlight_saturation=0.25,
            shadow_hue=195, shadow_saturation=0.3,
            vignette=0.2, fade=0.05
        ),
        ColorGradePreset.CINEMATIC_BLOCKBUSTER: ColorGradeSettings(
            contrast=0.25, blacks=-0.1, highlights=-0.15,
            temperature=0.05, saturation=0.1,
            highlight_hue=45, highlight_saturation=0.15,
            shadow_hue=210, shadow_saturation=0.2,
            clarity=0.2, vignette=0.25
        ),
        ColorGradePreset.CINEMATIC_NOIR: ColorGradeSettings(
            contrast=0.4, shadows=-0.2, highlights=-0.1,
            saturation=-0.8, vibrance=-0.3,
            clarity=0.3, vignette=0.4, grain=0.15
        ),
        ColorGradePreset.CINEMATIC_WARM: ColorGradeSettings(
            exposure=0.1, contrast=0.1, shadows=0.1,
            temperature=0.3, tint=0.05, saturation=0.1,
            highlight_hue=40, highlight_saturation=0.2,
            vignette=0.15
        ),
        ColorGradePreset.CINEMATIC_COLD: ColorGradeSettings(
            contrast=0.15, highlights=-0.1,
            temperature=-0.3, saturation=-0.1,
            shadow_hue=220, shadow_saturation=0.25,
            vignette=0.2, dehaze=0.1
        ),
        ColorGradePreset.FILM_KODAK_PORTRA: ColorGradeSettings(
            exposure=0.05, contrast=-0.05, highlights=-0.1, shadows=0.15,
            temperature=0.1, saturation=-0.1, vibrance=0.1,
            highlight_hue=50, highlight_saturation=0.1,
            shadow_hue=30, shadow_saturation=0.15,
            fade=0.1, grain=0.08
        ),
        ColorGradePreset.FILM_FUJI_VELVIA: ColorGradeSettings(
            contrast=0.2, saturation=0.3, vibrance=0.2,
            temperature=-0.05,
            highlight_hue=200, highlight_saturation=0.1,
            clarity=0.1, grain=0.05
        ),
        ColorGradePreset.FILM_KODAK_EKTAR: ColorGradeSettings(
            contrast=0.15, saturation=0.2, vibrance=0.15,
            temperature=0.05,
            highlight_hue=45, highlight_saturation=0.15,
            shadow_hue=200, shadow_saturation=0.1,
            grain=0.06
        ),
        ColorGradePreset.FILM_ILFORD_BW: ColorGradeSettings(
            contrast=0.3, saturation=-1.0,
            highlights=-0.1, shadows=0.1,
            clarity=0.2, grain=0.12
        ),
        ColorGradePreset.INSTAGRAM_CLARENDON: ColorGradeSettings(
            contrast=0.2, highlights=0.1, shadows=-0.1,
            saturation=0.15, vibrance=0.1,
            temperature=-0.05
        ),
        ColorGradePreset.INSTAGRAM_VALENCIA: ColorGradeSettings(
            exposure=0.1, contrast=-0.05,
            temperature=0.15, saturation=-0.1,
            fade=0.15, vignette=0.1
        ),
        ColorGradePreset.VSCO_A6: ColorGradeSettings(
            exposure=0.05, contrast=-0.1, shadows=0.2,
            temperature=0.1, saturation=-0.15, vibrance=0.1,
            fade=0.2, grain=0.1
        ),
        ColorGradePreset.VINTAGE_FADED: ColorGradeSettings(
            contrast=-0.15, highlights=-0.1, shadows=0.2,
            temperature=0.1, saturation=-0.2,
            highlight_hue=45, highlight_saturation=0.15,
            fade=0.3, grain=0.15, vignette=0.2
        ),
        ColorGradePreset.PRODUCT_CLEAN: ColorGradeSettings(
            exposure=0.1, contrast=0.1, highlights=0.05,
            saturation=0.05, vibrance=0.1,
            clarity=0.15, dehaze=0.1
        ),
        ColorGradePreset.PRODUCT_VIBRANT: ColorGradeSettings(
            contrast=0.15, saturation=0.25, vibrance=0.2,
            clarity=0.2, dehaze=0.15
        ),
        ColorGradePreset.PRODUCT_LUXURY: ColorGradeSettings(
            contrast=0.2, blacks=-0.1, highlights=-0.1,
            saturation=-0.1, vibrance=0.1,
            highlight_hue=45, highlight_saturation=0.1,
            shadow_hue=270, shadow_saturation=0.1,
            vignette=0.15, clarity=0.1
        ),
        ColorGradePreset.FOOD_WARM: ColorGradeSettings(
            exposure=0.1, contrast=0.1, shadows=0.1,
            temperature=0.25, saturation=0.15, vibrance=0.2,
            highlight_hue=40, highlight_saturation=0.15,
            clarity=0.1
        ),
        ColorGradePreset.MOODY_DARK: ColorGradeSettings(
            exposure=-0.2, contrast=0.25, shadows=-0.2, blacks=-0.15,
            saturation=-0.15,
            shadow_hue=220, shadow_saturation=0.2,
            vignette=0.35, clarity=0.15
        ),
        ColorGradePreset.BRIGHT_AIRY: ColorGradeSettings(
            exposure=0.2, contrast=-0.1, highlights=0.15, shadows=0.2,
            temperature=0.05, saturation=-0.1, vibrance=0.05,
            fade=0.1
        ),
        ColorGradePreset.GOLDEN_HOUR: ColorGradeSettings(
            exposure=0.1, contrast=0.1, shadows=0.15,
            temperature=0.35, tint=0.05, saturation=0.1,
            highlight_hue=40, highlight_saturation=0.25,
            shadow_hue=30, shadow_saturation=0.15,
            vignette=0.1
        ),
        ColorGradePreset.BLUE_HOUR: ColorGradeSettings(
            exposure=-0.05, contrast=0.15, shadows=0.1,
            temperature=-0.3, saturation=-0.05,
            highlight_hue=220, highlight_saturation=0.15,
            shadow_hue=240, shadow_saturation=0.25,
            vignette=0.15
        ),
    }

    def __init__(self):
        pass

    def apply_preset(self, image: Image.Image, preset: ColorGradePreset,
                    intensity: float = 1.0) -> Image.Image:
        """Apply a color grade preset"""
        settings = self.PRESETS.get(preset)
        if not settings:
            return image

        # Scale settings by intensity
        scaled_settings = self._scale_settings(settings, intensity)
        return self.apply_grade(image, scaled_settings)

    def apply_grade(self, image: Image.Image, settings: ColorGradeSettings) -> Image.Image:
        """Apply full color grade with all settings"""
        img = image.convert('RGB')
        img_np = np.array(img).astype(np.float32)

        # Apply adjustments in order
        img_np = self._apply_exposure(img_np, settings.exposure)
        img_np = self._apply_contrast(img_np, settings.contrast)
        img_np = self._apply_highlights_shadows(img_np, settings.highlights, settings.shadows)
        img_np = self._apply_whites_blacks(img_np, settings.whites, settings.blacks)
        img_np = self._apply_temperature_tint(img_np, settings.temperature, settings.tint)
        img_np = self._apply_saturation(img_np, settings.saturation)
        img_np = self._apply_vibrance(img_np, settings.vibrance)
        img_np = self._apply_split_toning(img_np, settings)
        img_np = self._apply_fade(img_np, settings.fade)

        # Clip values
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        result = Image.fromarray(img_np)

        # Apply PIL-based effects
        if settings.clarity != 0:
            result = self._apply_clarity(result, settings.clarity)

        if settings.dehaze != 0:
            result = self._apply_dehaze(result, settings.dehaze)

        if settings.vignette > 0:
            result = self._apply_vignette(result, settings.vignette)

        if settings.grain > 0:
            result = self._apply_grain(result, settings.grain)

        # Apply curves if provided
        if any([settings.red_curve, settings.green_curve, settings.blue_curve, settings.luma_curve]):
            result = self._apply_curves(result, settings)

        return result

    def _scale_settings(self, settings: ColorGradeSettings, intensity: float) -> ColorGradeSettings:
        """Scale all settings by intensity"""
        return ColorGradeSettings(
            exposure=settings.exposure * intensity,
            contrast=settings.contrast * intensity,
            highlights=settings.highlights * intensity,
            shadows=settings.shadows * intensity,
            whites=settings.whites * intensity,
            blacks=settings.blacks * intensity,
            temperature=settings.temperature * intensity,
            tint=settings.tint * intensity,
            vibrance=settings.vibrance * intensity,
            saturation=settings.saturation * intensity,
            highlight_hue=settings.highlight_hue,
            highlight_saturation=settings.highlight_saturation * intensity,
            shadow_hue=settings.shadow_hue,
            shadow_saturation=settings.shadow_saturation * intensity,
            balance=settings.balance,
            clarity=settings.clarity * intensity,
            dehaze=settings.dehaze * intensity,
            vignette=settings.vignette * intensity,
            grain=settings.grain * intensity,
            fade=settings.fade * intensity,
        )

    def _apply_exposure(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust exposure"""
        if amount == 0:
            return img
        factor = 2 ** amount
        return img * factor

    def _apply_contrast(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust contrast"""
        if amount == 0:
            return img
        factor = 1 + amount
        return (img - 128) * factor + 128

    def _apply_highlights_shadows(self, img: np.ndarray, highlights: float,
                                 shadows: float) -> np.ndarray:
        """Adjust highlights and shadows separately"""
        if highlights == 0 and shadows == 0:
            return img

        # Convert to luminance for masking
        luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Create smooth masks
        highlight_mask = np.clip((luma - 128) / 127, 0, 1)
        shadow_mask = np.clip((128 - luma) / 128, 0, 1)

        # Apply adjustments
        highlight_adjust = highlights * 50 * highlight_mask[:, :, np.newaxis]
        shadow_adjust = shadows * 50 * shadow_mask[:, :, np.newaxis]

        return img + highlight_adjust + shadow_adjust

    def _apply_whites_blacks(self, img: np.ndarray, whites: float,
                            blacks: float) -> np.ndarray:
        """Adjust white and black points"""
        if whites == 0 and blacks == 0:
            return img

        luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Whites affect very bright areas
        white_mask = np.clip((luma - 200) / 55, 0, 1)
        # Blacks affect very dark areas
        black_mask = np.clip((55 - luma) / 55, 0, 1)

        white_adjust = whites * 30 * white_mask[:, :, np.newaxis]
        black_adjust = blacks * 30 * black_mask[:, :, np.newaxis]

        return img + white_adjust + black_adjust

    def _apply_temperature_tint(self, img: np.ndarray, temperature: float,
                               tint: float) -> np.ndarray:
        """Adjust color temperature and tint"""
        if temperature == 0 and tint == 0:
            return img

        result = img.copy()

        # Temperature: shift between blue and orange
        if temperature != 0:
            temp_shift = temperature * 30
            result[:, :, 0] += temp_shift  # Red
            result[:, :, 2] -= temp_shift  # Blue

        # Tint: shift between green and magenta
        if tint != 0:
            tint_shift = tint * 20
            result[:, :, 1] -= tint_shift  # Green
            result[:, :, 0] += tint_shift * 0.5  # Red
            result[:, :, 2] += tint_shift * 0.5  # Blue

        return result

    def _apply_saturation(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust saturation"""
        if amount == 0:
            return img

        # Convert to HSV
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Adjust saturation
        factor = 1 + amount
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

        # Convert back
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32)

    def _apply_vibrance(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust vibrance (smart saturation that affects less saturated colors more)"""
        if amount == 0:
            return img

        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Calculate saturation weight (less saturated = more effect)
        sat_normalized = hsv[:, :, 1] / 255
        weight = 1 - sat_normalized  # Inverse: low sat = high weight

        # Apply vibrance with weight
        factor = 1 + amount * weight
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32)

    def _apply_split_toning(self, img: np.ndarray, settings: ColorGradeSettings) -> np.ndarray:
        """Apply split toning to highlights and shadows"""
        if settings.highlight_saturation == 0 and settings.shadow_saturation == 0:
            return img

        # Calculate luminance
        luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        result = img.copy()

        # Highlight toning
        if settings.highlight_saturation > 0:
            highlight_mask = np.clip((luma - 128) / 127, 0, 1)
            highlight_mask = highlight_mask ** 2  # Smooth falloff

            # Convert hue to RGB
            h_color = self._hue_to_rgb(settings.highlight_hue)
            for i in range(3):
                result[:, :, i] += (h_color[i] - 128) * settings.highlight_saturation * highlight_mask * 0.5

        # Shadow toning
        if settings.shadow_saturation > 0:
            shadow_mask = np.clip((128 - luma) / 128, 0, 1)
            shadow_mask = shadow_mask ** 2

            s_color = self._hue_to_rgb(settings.shadow_hue)
            for i in range(3):
                result[:, :, i] += (s_color[i] - 128) * settings.shadow_saturation * shadow_mask * 0.5

        return result

    def _hue_to_rgb(self, hue: int) -> Tuple[int, int, int]:
        """Convert hue (0-360) to RGB color"""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _apply_fade(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Apply fade/lift to blacks"""
        if amount == 0:
            return img
        lift = amount * 40
        return img + lift * (1 - img / 255)

    def _apply_clarity(self, image: Image.Image, amount: float) -> Image.Image:
        """Apply clarity (local contrast)"""
        if amount == 0:
            return image

        # Apply unsharp mask for clarity
        blurred = image.filter(ImageFilter.GaussianBlur(radius=20))
        img_np = np.array(image).astype(np.float32)
        blur_np = np.array(blurred).astype(np.float32)

        # High-pass
        high_pass = img_np - blur_np

        # Add back scaled
        factor = 1 + amount
        result = img_np + high_pass * (factor - 1) * 0.5

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_dehaze(self, image: Image.Image, amount: float) -> Image.Image:
        """Apply dehazing"""
        if amount == 0:
            return image

        img_np = np.array(image).astype(np.float32) / 255

        # Simple dehaze using dark channel prior approximation
        min_channel = np.min(img_np, axis=2)
        atmospheric = np.percentile(min_channel, 99)

        # Transmission estimate
        transmission = 1 - amount * min_channel / max(atmospheric, 0.01)
        transmission = np.clip(transmission, 0.1, 1)

        # Dehaze
        result = np.zeros_like(img_np)
        for i in range(3):
            result[:, :, i] = (img_np[:, :, i] - atmospheric) / transmission + atmospheric

        return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))

    def _apply_vignette(self, image: Image.Image, amount: float) -> Image.Image:
        """Apply vignette effect"""
        if amount == 0:
            return image

        w, h = image.size
        img_np = np.array(image).astype(np.float32)

        # Create radial gradient mask
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2

        # Normalized distance from center
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        normalized_dist = dist / max_dist

        # Vignette falloff
        vignette = 1 - (normalized_dist ** 2) * amount
        vignette = np.clip(vignette, 0, 1)

        # Apply to all channels
        for i in range(3):
            img_np[:, :, i] *= vignette

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    def _apply_grain(self, image: Image.Image, amount: float) -> Image.Image:
        """Apply film grain"""
        if amount == 0:
            return image

        img_np = np.array(image).astype(np.float32)
        h, w = img_np.shape[:2]

        # Generate grain
        grain = np.random.normal(0, amount * 30, (h, w))

        # Apply to luminance-weighted
        for i in range(3):
            img_np[:, :, i] += grain

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    def _apply_curves(self, image: Image.Image, settings: ColorGradeSettings) -> Image.Image:
        """Apply tone curves"""
        img_np = np.array(image)

        # Build LUTs from curves
        if settings.luma_curve:
            lut = self._curve_to_lut(settings.luma_curve)
            # Apply to all channels
            for i in range(3):
                img_np[:, :, i] = lut[img_np[:, :, i]]

        if settings.red_curve:
            lut = self._curve_to_lut(settings.red_curve)
            img_np[:, :, 0] = lut[img_np[:, :, 0]]

        if settings.green_curve:
            lut = self._curve_to_lut(settings.green_curve)
            img_np[:, :, 1] = lut[img_np[:, :, 1]]

        if settings.blue_curve:
            lut = self._curve_to_lut(settings.blue_curve)
            img_np[:, :, 2] = lut[img_np[:, :, 2]]

        return Image.fromarray(img_np)

    def _curve_to_lut(self, curve_points: List[Tuple[int, int]]) -> np.ndarray:
        """Convert curve control points to 256-entry LUT"""
        # Sort points by x
        points = sorted(curve_points, key=lambda p: p[0])

        # Interpolate
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find surrounding points
            lower_idx = 0
            for j, p in enumerate(points):
                if p[0] <= i:
                    lower_idx = j

            if lower_idx >= len(points) - 1:
                lut[i] = points[-1][1]
            else:
                p1 = points[lower_idx]
                p2 = points[lower_idx + 1]
                t = (i - p1[0]) / max(p2[0] - p1[0], 1)
                lut[i] = int(p1[1] + (p2[1] - p1[1]) * t)

        return lut


class HDRToneMapper:
    """HDR tone mapping for high dynamic range effects"""

    @staticmethod
    def reinhard(image: Image.Image, key: float = 0.18,
                white_point: float = 1.0) -> Image.Image:
        """Reinhard tone mapping"""
        img = np.array(image).astype(np.float32) / 255

        # Calculate luminance
        luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Log average luminance
        log_avg = np.exp(np.mean(np.log(luma + 1e-6)))

        # Scaled luminance
        l_scaled = key * luma / log_avg

        # Reinhard operator
        l_mapped = l_scaled * (1 + l_scaled / (white_point ** 2)) / (1 + l_scaled)

        # Apply to colors
        ratio = l_mapped / (luma + 1e-6)
        result = img * ratio[:, :, np.newaxis]

        return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))

    @staticmethod
    def aces_filmic(image: Image.Image, exposure: float = 1.0) -> Image.Image:
        """ACES filmic tone mapping"""
        img = np.array(image).astype(np.float32) / 255 * exposure

        # ACES parameters
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        # Apply ACES curve
        result = (img * (a * img + b)) / (img * (c * img + d) + e)

        return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))


# Convenience functions
def apply_color_grade(image: Image.Image, preset: str,
                     intensity: float = 1.0) -> Image.Image:
    """Apply color grade preset by name"""
    grader = ColorGrader()
    try:
        preset_enum = ColorGradePreset(preset)
        return grader.apply_preset(image, preset_enum, intensity)
    except ValueError:
        return image


def get_available_presets() -> List[Dict[str, str]]:
    """Get list of available color grade presets"""
    return [
        {'id': preset.value, 'name': preset.value.replace('_', ' ').title()}
        for preset in ColorGradePreset
    ]


def create_custom_grade(image: Image.Image, **kwargs) -> Image.Image:
    """Apply custom color grade with keyword arguments"""
    settings = ColorGradeSettings(**kwargs)
    grader = ColorGrader()
    return grader.apply_grade(image, settings)
