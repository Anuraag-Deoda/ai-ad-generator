"""
Advanced Motion Graphics Module

Professional-grade motion graphics effects:
- 3D transformations and perspective
- Morphing and shape animations
- Path-based animations
- Liquid/fluid effects
- Advanced easing functions
- Procedural animations
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import cv2
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any
from enum import Enum


class EasingType(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BOUNCE = "bounce"
    ELASTIC = "elastic"
    BACK = "back"
    EXPO = "expo"
    CIRC = "circ"
    SPRING = "spring"


class Easing:
    """Professional easing functions"""

    @staticmethod
    def apply(t: float, easing_type: EasingType) -> float:
        """Apply easing function to normalized time value"""
        t = max(0, min(1, t))

        if easing_type == EasingType.LINEAR:
            return t

        elif easing_type == EasingType.EASE_IN:
            return t * t * t

        elif easing_type == EasingType.EASE_OUT:
            return 1 - (1 - t) ** 3

        elif easing_type == EasingType.EASE_IN_OUT:
            if t < 0.5:
                return 4 * t * t * t
            else:
                return 1 - (-2 * t + 2) ** 3 / 2

        elif easing_type == EasingType.BOUNCE:
            return Easing._bounce_out(t)

        elif easing_type == EasingType.ELASTIC:
            return Easing._elastic_out(t)

        elif easing_type == EasingType.BACK:
            return Easing._back_out(t)

        elif easing_type == EasingType.EXPO:
            if t == 0:
                return 0
            return 2 ** (10 * (t - 1)) if t < 1 else 1

        elif easing_type == EasingType.CIRC:
            return 1 - math.sqrt(1 - t * t)

        elif easing_type == EasingType.SPRING:
            return Easing._spring(t)

        return t

    @staticmethod
    def _bounce_out(t: float) -> float:
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

    @staticmethod
    def _elastic_out(t: float) -> float:
        if t == 0 or t == 1:
            return t
        return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3) + 1

    @staticmethod
    def _back_out(t: float) -> float:
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

    @staticmethod
    def _spring(t: float) -> float:
        return 1 - math.cos(t * 4.5 * math.pi) * math.exp(-t * 6)


@dataclass
class Transform3D:
    """3D transformation parameters"""
    rotate_x: float = 0  # degrees
    rotate_y: float = 0
    rotate_z: float = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    translate_x: float = 0
    translate_y: float = 0
    translate_z: float = 0
    perspective: float = 1000  # distance to viewer


class Transform3DEngine:
    """Apply 3D transformations to images"""

    @staticmethod
    def apply_perspective(image: Image.Image, transform: Transform3D) -> Image.Image:
        """Apply 3D perspective transformation"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        # Calculate rotation matrices
        rx = np.radians(transform.rotate_x)
        ry = np.radians(transform.rotate_y)
        rz = np.radians(transform.rotate_z)

        # Rotation matrix around X
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])

        # Rotation matrix around Y
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])

        # Rotation matrix around Z
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combined rotation
        R = Rz @ Ry @ Rx

        # Define corners in 3D space (centered at origin)
        corners_3d = np.array([
            [-w/2, -h/2, 0],
            [w/2, -h/2, 0],
            [w/2, h/2, 0],
            [-w/2, h/2, 0]
        ], dtype=np.float32)

        # Apply rotation
        rotated = corners_3d @ R.T

        # Apply translation
        rotated[:, 0] += transform.translate_x
        rotated[:, 1] += transform.translate_y
        rotated[:, 2] += transform.translate_z

        # Apply perspective projection
        p = transform.perspective
        projected = np.zeros((4, 2), dtype=np.float32)
        for i, point in enumerate(rotated):
            z = point[2] + p
            if z > 0:
                projected[i, 0] = (point[0] * p / z) + w / 2
                projected[i, 1] = (point[1] * p / z) + h / 2
            else:
                projected[i] = [w/2, h/2]

        # Original corners
        src_pts = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, projected)

        # Apply transform
        result = cv2.warpPerspective(img_np, M, (w, h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

        return Image.fromarray(result)

    @staticmethod
    def card_flip(image: Image.Image, progress: float,
                 direction: str = 'horizontal') -> Image.Image:
        """Animate a card flip effect"""
        # Progress 0-0.5: front face rotating away
        # Progress 0.5-1: back face rotating in

        if progress < 0.5:
            angle = progress * 180
        else:
            angle = (1 - progress) * 180

        if direction == 'horizontal':
            transform = Transform3D(rotate_y=angle)
        else:
            transform = Transform3D(rotate_x=angle)

        return Transform3DEngine.apply_perspective(image, transform)

    @staticmethod
    def cube_rotate(image: Image.Image, progress: float,
                   axis: str = 'y') -> Image.Image:
        """Rotate image as if on a cube face"""
        angle = progress * 90

        if axis == 'y':
            transform = Transform3D(rotate_y=angle, translate_z=-100)
        elif axis == 'x':
            transform = Transform3D(rotate_x=angle, translate_z=-100)
        else:
            transform = Transform3D(rotate_z=angle)

        return Transform3DEngine.apply_perspective(image, transform)

    @staticmethod
    def tilt_shift(image: Image.Image, focus_y: float = 0.5,
                  blur_amount: int = 15) -> Image.Image:
        """Create tilt-shift miniature effect"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        # Create blur mask
        mask = np.zeros((h, w), dtype=np.float32)

        focus_row = int(h * focus_y)
        focus_height = h // 4

        for y in range(h):
            dist = abs(y - focus_row)
            if dist < focus_height:
                mask[y, :] = 0
            else:
                mask[y, :] = min(1.0, (dist - focus_height) / (h / 3))

        # Apply graduated blur
        blurred = cv2.GaussianBlur(img_np, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)

        # Blend based on mask
        mask_3d = np.stack([mask] * img_np.shape[2], axis=2)
        result = (img_np * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)

        return Image.fromarray(result)


class MorphEngine:
    """Image morphing and shape animations"""

    @staticmethod
    def cross_dissolve(img1: Image.Image, img2: Image.Image,
                      progress: float) -> Image.Image:
        """Simple cross-dissolve between two images"""
        # Ensure same size
        size = img1.size
        img2 = img2.resize(size, Image.Resampling.LANCZOS)

        return Image.blend(img1, img2, progress)

    @staticmethod
    def wipe_transition(img1: Image.Image, img2: Image.Image,
                       progress: float, direction: str = 'left') -> Image.Image:
        """Wipe transition between images"""
        w, h = img1.size
        img2 = img2.resize((w, h), Image.Resampling.LANCZOS)

        result = img1.copy()

        if direction == 'left':
            split = int(w * progress)
            result.paste(img2.crop((0, 0, split, h)), (0, 0))

        elif direction == 'right':
            split = int(w * (1 - progress))
            result.paste(img2.crop((split, 0, w, h)), (split, 0))

        elif direction == 'up':
            split = int(h * progress)
            result.paste(img2.crop((0, 0, w, split)), (0, 0))

        elif direction == 'down':
            split = int(h * (1 - progress))
            result.paste(img2.crop((0, split, w, h)), (0, split))

        return result

    @staticmethod
    def iris_transition(img1: Image.Image, img2: Image.Image,
                       progress: float, center: Tuple[float, float] = (0.5, 0.5)) -> Image.Image:
        """Circular iris wipe transition"""
        w, h = img1.size
        img2 = img2.resize((w, h), Image.Resampling.LANCZOS)

        # Create circular mask
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)

        cx, cy = int(w * center[0]), int(h * center[1])
        max_radius = math.sqrt(w*w + h*h)
        radius = int(max_radius * progress)

        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)

        # Apply Gaussian blur to mask for soft edge
        mask = mask.filter(ImageFilter.GaussianBlur(20))

        # Composite
        result = Image.composite(img2, img1, mask)
        return result

    @staticmethod
    def pixelate_transition(img1: Image.Image, img2: Image.Image,
                           progress: float) -> Image.Image:
        """Pixelate transition effect"""
        w, h = img1.size
        img2 = img2.resize((w, h), Image.Resampling.LANCZOS)

        if progress < 0.5:
            # First half: pixelate img1
            source = img1
            pixel_progress = progress * 2  # 0 to 1
        else:
            # Second half: de-pixelate into img2
            source = img2
            pixel_progress = 1 - (progress - 0.5) * 2  # 1 to 0

        # Calculate pixel size (1 = normal, higher = more pixelated)
        max_pixel_size = 50
        pixel_size = max(1, int(pixel_progress * max_pixel_size))

        if pixel_size > 1:
            small = source.resize((w // pixel_size, h // pixel_size), Image.Resampling.NEAREST)
            result = small.resize((w, h), Image.Resampling.NEAREST)
        else:
            result = source

        return result


class LiquidEffect:
    """Fluid/liquid motion effects"""

    @staticmethod
    def wave_distortion(image: Image.Image, progress: float,
                       amplitude: float = 20, frequency: float = 3) -> Image.Image:
        """Apply wave distortion effect"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        # Create displacement maps
        x_indices = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        y_indices = np.tile(np.arange(h).reshape(-1, 1), (1, w)).astype(np.float32)

        # Wave displacement
        phase = progress * 2 * np.pi
        x_displacement = amplitude * np.sin(frequency * 2 * np.pi * y_indices / h + phase)
        y_displacement = amplitude * np.sin(frequency * 2 * np.pi * x_indices / w + phase)

        # Apply displacement
        map_x = (x_indices + x_displacement).astype(np.float32)
        map_y = (y_indices + y_displacement).astype(np.float32)

        result = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(result)

    @staticmethod
    def ripple_effect(image: Image.Image, progress: float,
                     center: Tuple[float, float] = (0.5, 0.5),
                     wavelength: float = 30, amplitude: float = 10) -> Image.Image:
        """Create ripple/water drop effect"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        cx, cy = int(w * center[0]), int(h * center[1])

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Calculate distance from center
        dx = x_coords - cx
        dy = y_coords - cy
        distance = np.sqrt(dx*dx + dy*dy)

        # Wave equation
        phase = progress * 4 * np.pi
        wave = amplitude * np.sin(distance / wavelength * 2 * np.pi - phase)

        # Attenuate with distance
        max_dist = math.sqrt(w*w + h*h) / 2
        attenuation = 1 - np.clip(distance / max_dist, 0, 1)
        wave *= attenuation

        # Displacement in radial direction
        with np.errstate(divide='ignore', invalid='ignore'):
            dx_norm = np.where(distance > 0, dx / distance, 0)
            dy_norm = np.where(distance > 0, dy / distance, 0)

        map_x = (x_coords + wave * dx_norm).astype(np.float32)
        map_y = (y_coords + wave * dy_norm).astype(np.float32)

        result = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(result)

    @staticmethod
    def blob_morph(image: Image.Image, progress: float,
                  intensity: float = 0.3) -> Image.Image:
        """Organic blob-like morphing"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        # Create multiple noise layers for organic movement
        t = progress * 2 * np.pi

        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Simplex-like noise approximation using sin combinations
        noise_x = (
            np.sin(x_coords / 50 + t) * 15 +
            np.sin(y_coords / 30 + t * 1.3) * 10 +
            np.sin((x_coords + y_coords) / 70 + t * 0.7) * 8
        )

        noise_y = (
            np.sin(y_coords / 50 + t * 1.1) * 15 +
            np.sin(x_coords / 30 + t * 0.9) * 10 +
            np.sin((x_coords - y_coords) / 70 + t * 1.4) * 8
        )

        map_x = (x_coords + noise_x * intensity).astype(np.float32)
        map_y = (y_coords + noise_y * intensity).astype(np.float32)

        result = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(result)


class PathAnimation:
    """Animate objects along paths"""

    @staticmethod
    def bezier_point(p0: Tuple[float, float], p1: Tuple[float, float],
                    p2: Tuple[float, float], p3: Tuple[float, float],
                    t: float) -> Tuple[float, float]:
        """Calculate point on cubic bezier curve"""
        x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        return (x, y)

    @staticmethod
    def animate_along_path(image: Image.Image, canvas_size: Tuple[int, int],
                          path_points: List[Tuple[float, float]],
                          progress: float,
                          easing: EasingType = EasingType.EASE_IN_OUT) -> Image.Image:
        """Move image along a path defined by control points"""
        canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        t = Easing.apply(progress, easing)

        # Calculate position on path
        if len(path_points) == 4:
            # Cubic bezier
            pos = PathAnimation.bezier_point(
                path_points[0], path_points[1],
                path_points[2], path_points[3], t
            )
        elif len(path_points) >= 2:
            # Linear interpolation between points
            segment_count = len(path_points) - 1
            segment = min(int(t * segment_count), segment_count - 1)
            segment_t = (t * segment_count) - segment

            p1 = path_points[segment]
            p2 = path_points[segment + 1]

            pos = (
                p1[0] + (p2[0] - p1[0]) * segment_t,
                p1[1] + (p2[1] - p1[1]) * segment_t
            )
        else:
            pos = path_points[0] if path_points else (0.5, 0.5)

        # Convert normalized coordinates to pixels
        x = int(pos[0] * canvas_size[0] - image.width / 2)
        y = int(pos[1] * canvas_size[1] - image.height / 2)

        canvas.paste(image, (x, y), image if image.mode == 'RGBA' else None)
        return canvas

    @staticmethod
    def orbit_animation(image: Image.Image, canvas_size: Tuple[int, int],
                       progress: float, radius: float = 0.3,
                       center: Tuple[float, float] = (0.5, 0.5),
                       rotations: float = 1) -> Image.Image:
        """Animate image in circular orbit"""
        canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        angle = progress * rotations * 2 * np.pi

        # Calculate position
        cx = center[0] * canvas_size[0]
        cy = center[1] * canvas_size[1]
        r = radius * min(canvas_size)

        x = int(cx + r * np.cos(angle) - image.width / 2)
        y = int(cy + r * np.sin(angle) - image.height / 2)

        canvas.paste(image, (x, y), image if image.mode == 'RGBA' else None)
        return canvas


class ProceduralMotion:
    """Procedural motion generation"""

    @staticmethod
    def shake(intensity: float, progress: float, frequency: float = 10) -> Tuple[float, float]:
        """Generate shake motion offset"""
        t = progress * frequency * 2 * np.pi

        # Decay over time
        decay = 1 - progress

        x = intensity * np.sin(t * 1.1) * decay * np.random.uniform(0.8, 1.2)
        y = intensity * np.cos(t * 1.3) * decay * np.random.uniform(0.8, 1.2)

        return (x, y)

    @staticmethod
    def breathe(progress: float, intensity: float = 0.05) -> float:
        """Generate breathing scale animation"""
        return 1 + intensity * np.sin(progress * 2 * np.pi)

    @staticmethod
    def wiggle(progress: float, frequency: float = 5,
              amplitude: float = 10) -> Tuple[float, float]:
        """Generate wiggle motion"""
        t = progress * frequency * 2 * np.pi

        # Use multiple frequencies for more organic movement
        x = amplitude * (np.sin(t) + 0.5 * np.sin(t * 2.3) + 0.3 * np.sin(t * 3.7))
        y = amplitude * (np.cos(t * 1.1) + 0.5 * np.cos(t * 2.1) + 0.3 * np.cos(t * 3.3))

        return (x, y)

    @staticmethod
    def pendulum(progress: float, amplitude: float = 30,
                damping: float = 0.3) -> float:
        """Generate pendulum swing angle"""
        t = progress * 4 * np.pi
        decay = np.exp(-damping * progress * 10)
        return amplitude * np.sin(t) * decay


# Convenience functions
def apply_3d_transform(image: Image.Image, rx: float = 0, ry: float = 0,
                      rz: float = 0, perspective: float = 1000) -> Image.Image:
    """Apply 3D rotation to image"""
    transform = Transform3D(rotate_x=rx, rotate_y=ry, rotate_z=rz, perspective=perspective)
    return Transform3DEngine.apply_perspective(image, transform)


def transition_morph(img1: Image.Image, img2: Image.Image,
                    progress: float, transition_type: str = 'dissolve') -> Image.Image:
    """Apply transition between two images"""
    if transition_type == 'dissolve':
        return MorphEngine.cross_dissolve(img1, img2, progress)
    elif transition_type == 'wipe':
        return MorphEngine.wipe_transition(img1, img2, progress)
    elif transition_type == 'iris':
        return MorphEngine.iris_transition(img1, img2, progress)
    elif transition_type == 'pixelate':
        return MorphEngine.pixelate_transition(img1, img2, progress)
    return MorphEngine.cross_dissolve(img1, img2, progress)


def apply_liquid_effect(image: Image.Image, progress: float,
                       effect_type: str = 'wave') -> Image.Image:
    """Apply liquid/fluid effect"""
    if effect_type == 'wave':
        return LiquidEffect.wave_distortion(image, progress)
    elif effect_type == 'ripple':
        return LiquidEffect.ripple_effect(image, progress)
    elif effect_type == 'blob':
        return LiquidEffect.blob_morph(image, progress)
    return image
