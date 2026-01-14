"""
Advanced Particle System Module

Professional particle effects for video generation:
- Configurable emitters
- Physics simulation
- Multiple particle types
- Blend modes and rendering
- Presets for common effects
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum


class ParticleShape(Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    STAR = "star"
    SPARKLE = "sparkle"
    SMOKE = "smoke"
    GLOW = "glow"
    LINE = "line"
    TRIANGLE = "triangle"
    HEART = "heart"
    CUSTOM = "custom"


class BlendMode(Enum):
    NORMAL = "normal"
    ADD = "add"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"


class EmitterShape(Enum):
    POINT = "point"
    LINE = "line"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    EDGE = "edge"


@dataclass
class ParticleConfig:
    """Configuration for individual particles"""
    shape: ParticleShape = ParticleShape.CIRCLE
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    color_variance: float = 0.0  # 0-1 variance in color
    color_over_life: Optional[List[Tuple[int, int, int, int]]] = None

    size: float = 10.0
    size_variance: float = 0.2
    size_over_life: Optional[List[float]] = None  # Multipliers

    speed: float = 100.0  # pixels per second
    speed_variance: float = 0.3
    direction: float = 270.0  # degrees (270 = up)
    direction_variance: float = 30.0  # spread in degrees

    lifetime: float = 2.0  # seconds
    lifetime_variance: float = 0.3

    gravity: Tuple[float, float] = (0, 0)  # x, y acceleration
    drag: float = 0.0  # 0-1 velocity reduction per second

    rotation: float = 0.0  # initial rotation
    rotation_speed: float = 0.0  # degrees per second
    rotation_variance: float = 0.0

    fade_in: float = 0.1  # portion of life for fade in
    fade_out: float = 0.3  # portion of life for fade out

    blur: float = 0.0  # blur amount
    glow: float = 0.0  # glow intensity


@dataclass
class EmitterConfig:
    """Configuration for particle emitter"""
    shape: EmitterShape = EmitterShape.POINT
    position: Tuple[float, float] = (0.5, 0.5)  # normalized 0-1
    size: Tuple[float, float] = (0.1, 0.1)  # for non-point shapes

    emission_rate: float = 50.0  # particles per second
    burst_count: int = 0  # one-time burst
    max_particles: int = 500

    start_delay: float = 0.0  # seconds before starting
    duration: float = -1  # -1 = infinite

    particle_config: ParticleConfig = field(default_factory=ParticleConfig)


@dataclass
class Particle:
    """Individual particle instance"""
    x: float
    y: float
    vx: float
    vy: float
    size: float
    color: Tuple[int, int, int, int]
    rotation: float
    rotation_speed: float
    age: float
    lifetime: float
    config: ParticleConfig


class ParticleSystem:
    """Advanced particle system with physics simulation"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.emitters: List[Tuple[EmitterConfig, List[Particle]]] = []
        self.time = 0.0
        self.emission_accumulators: List[float] = []

    def add_emitter(self, config: EmitterConfig):
        """Add a particle emitter"""
        self.emitters.append((config, []))
        self.emission_accumulators.append(0.0)

    def clear(self):
        """Clear all emitters and particles"""
        self.emitters = []
        self.emission_accumulators = []
        self.time = 0.0

    def update(self, dt: float):
        """Update all particles"""
        self.time += dt

        for i, (config, particles) in enumerate(self.emitters):
            # Check if emitter is active
            if self.time < config.start_delay:
                continue
            if config.duration > 0 and self.time > config.start_delay + config.duration:
                # Just update existing particles
                self._update_particles(particles, config.particle_config, dt)
                continue

            # Emit new particles
            if config.burst_count > 0 and self.emission_accumulators[i] == 0:
                # One-time burst
                for _ in range(config.burst_count):
                    if len(particles) < config.max_particles:
                        particles.append(self._create_particle(config))
                self.emission_accumulators[i] = 1  # Mark as burst done

            elif config.emission_rate > 0:
                # Continuous emission
                self.emission_accumulators[i] += config.emission_rate * dt
                while self.emission_accumulators[i] >= 1.0:
                    if len(particles) < config.max_particles:
                        particles.append(self._create_particle(config))
                    self.emission_accumulators[i] -= 1.0

            # Update existing particles
            self._update_particles(particles, config.particle_config, dt)

    def _create_particle(self, config: EmitterConfig) -> Particle:
        """Create a new particle"""
        pc = config.particle_config

        # Calculate spawn position based on emitter shape
        x, y = self._get_spawn_position(config)

        # Calculate velocity
        direction = pc.direction + random.uniform(-pc.direction_variance, pc.direction_variance)
        speed = pc.speed * (1 + random.uniform(-pc.speed_variance, pc.speed_variance))
        vx = speed * math.cos(math.radians(direction))
        vy = speed * math.sin(math.radians(direction))

        # Calculate other properties
        size = pc.size * (1 + random.uniform(-pc.size_variance, pc.size_variance))
        lifetime = pc.lifetime * (1 + random.uniform(-pc.lifetime_variance, pc.lifetime_variance))

        # Color with variance
        color = list(pc.color)
        if pc.color_variance > 0:
            for i in range(3):
                variance = int(pc.color_variance * 50)
                color[i] = max(0, min(255, color[i] + random.randint(-variance, variance)))
        color = tuple(color)

        rotation = pc.rotation + random.uniform(-pc.rotation_variance, pc.rotation_variance)
        rotation_speed = pc.rotation_speed + random.uniform(-pc.rotation_variance * 10, pc.rotation_variance * 10)

        return Particle(
            x=x, y=y, vx=vx, vy=vy,
            size=size, color=color,
            rotation=rotation, rotation_speed=rotation_speed,
            age=0, lifetime=lifetime, config=pc
        )

    def _get_spawn_position(self, config: EmitterConfig) -> Tuple[float, float]:
        """Get spawn position based on emitter shape"""
        cx = config.position[0] * self.width
        cy = config.position[1] * self.height

        if config.shape == EmitterShape.POINT:
            return (cx, cy)

        elif config.shape == EmitterShape.LINE:
            w = config.size[0] * self.width
            offset = random.uniform(-w/2, w/2)
            return (cx + offset, cy)

        elif config.shape == EmitterShape.CIRCLE:
            r = config.size[0] * min(self.width, self.height) / 2
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, r)
            return (cx + dist * math.cos(angle), cy + dist * math.sin(angle))

        elif config.shape == EmitterShape.RECTANGLE:
            w = config.size[0] * self.width
            h = config.size[1] * self.height
            return (cx + random.uniform(-w/2, w/2), cy + random.uniform(-h/2, h/2))

        elif config.shape == EmitterShape.EDGE:
            # Spawn from edges of screen
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                return (random.uniform(0, self.width), 0)
            elif edge == 'bottom':
                return (random.uniform(0, self.width), self.height)
            elif edge == 'left':
                return (0, random.uniform(0, self.height))
            else:
                return (self.width, random.uniform(0, self.height))

        return (cx, cy)

    def _update_particles(self, particles: List[Particle], config: ParticleConfig, dt: float):
        """Update particle physics"""
        to_remove = []

        for i, p in enumerate(particles):
            p.age += dt

            # Check lifetime
            if p.age >= p.lifetime:
                to_remove.append(i)
                continue

            # Apply gravity
            p.vx += config.gravity[0] * dt
            p.vy += config.gravity[1] * dt

            # Apply drag
            if config.drag > 0:
                drag_factor = 1 - config.drag * dt
                p.vx *= drag_factor
                p.vy *= drag_factor

            # Update position
            p.x += p.vx * dt
            p.y += p.vy * dt

            # Update rotation
            p.rotation += p.rotation_speed * dt

        # Remove dead particles (in reverse order)
        for i in reversed(to_remove):
            particles.pop(i)

    def render(self, background: Image.Image, blend_mode: BlendMode = BlendMode.ADD) -> Image.Image:
        """Render all particles onto background"""
        # Create particle layer
        particle_layer = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))

        for config, particles in self.emitters:
            for p in particles:
                self._render_particle(particle_layer, p)

        # Blend with background
        return self._blend_layers(background, particle_layer, blend_mode)

    def _render_particle(self, layer: Image.Image, p: Particle):
        """Render single particle"""
        # Calculate life progress
        life_progress = p.age / p.lifetime

        # Calculate alpha based on fade in/out
        alpha = 255
        if life_progress < p.config.fade_in:
            alpha = int(255 * (life_progress / p.config.fade_in))
        elif life_progress > (1 - p.config.fade_out):
            alpha = int(255 * ((1 - life_progress) / p.config.fade_out))

        # Calculate size over life
        size = p.size
        if p.config.size_over_life:
            idx = int(life_progress * (len(p.config.size_over_life) - 1))
            idx = min(idx, len(p.config.size_over_life) - 1)
            size *= p.config.size_over_life[idx]

        # Calculate color over life
        color = p.color
        if p.config.color_over_life:
            idx = int(life_progress * (len(p.config.color_over_life) - 1))
            idx = min(idx, len(p.config.color_over_life) - 1)
            color = p.config.color_over_life[idx]

        # Apply alpha
        color = (color[0], color[1], color[2], int(color[3] * alpha / 255))

        # Skip if fully transparent
        if color[3] == 0 or size < 1:
            return

        # Create particle image
        particle_img = self._create_particle_shape(p.config.shape, int(size * 2), color, p.rotation)

        # Apply blur if needed
        if p.config.blur > 0:
            particle_img = particle_img.filter(ImageFilter.GaussianBlur(p.config.blur * size / 10))

        # Apply glow if needed
        if p.config.glow > 0:
            glow_img = particle_img.filter(ImageFilter.GaussianBlur(p.config.glow * size / 5))
            particle_img = Image.blend(particle_img, glow_img, 0.5)

        # Paste onto layer
        x = int(p.x - particle_img.width / 2)
        y = int(p.y - particle_img.height / 2)

        if 0 <= x < self.width and 0 <= y < self.height:
            layer.paste(particle_img, (x, y), particle_img)

    def _create_particle_shape(self, shape: ParticleShape, size: int,
                              color: Tuple[int, int, int, int],
                              rotation: float = 0) -> Image.Image:
        """Create particle shape image"""
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        center = size // 2
        radius = size // 2 - 1

        if shape == ParticleShape.CIRCLE:
            draw.ellipse([1, 1, size-1, size-1], fill=color)

        elif shape == ParticleShape.SQUARE:
            margin = size // 4
            draw.rectangle([margin, margin, size-margin, size-margin], fill=color)

        elif shape == ParticleShape.STAR:
            points = self._star_points(center, center, radius, radius // 2, 5)
            draw.polygon(points, fill=color)

        elif shape == ParticleShape.SPARKLE:
            # Four-pointed sparkle
            points = self._sparkle_points(center, center, radius)
            draw.polygon(points, fill=color)

        elif shape == ParticleShape.SMOKE:
            # Soft gradient circle
            for r in range(radius, 0, -1):
                alpha = int(color[3] * (r / radius) ** 2)
                c = (color[0], color[1], color[2], alpha)
                draw.ellipse([center-r, center-r, center+r, center+r], fill=c)

        elif shape == ParticleShape.GLOW:
            # Radial gradient
            for r in range(radius, 0, -1):
                alpha = int(color[3] * ((radius - r) / radius))
                c = (color[0], color[1], color[2], alpha)
                draw.ellipse([center-r, center-r, center+r, center+r], fill=c)

        elif shape == ParticleShape.LINE:
            draw.line([center, 0, center, size], fill=color, width=2)

        elif shape == ParticleShape.TRIANGLE:
            points = self._regular_polygon_points(center, center, radius, 3)
            draw.polygon(points, fill=color)

        elif shape == ParticleShape.HEART:
            points = self._heart_points(center, center, radius)
            draw.polygon(points, fill=color)

        # Apply rotation
        if rotation != 0:
            img = img.rotate(-rotation, resample=Image.BICUBIC, expand=False)

        return img

    def _star_points(self, cx: int, cy: int, outer_r: int, inner_r: int, points: int) -> List[Tuple[int, int]]:
        """Generate star polygon points"""
        result = []
        angle_step = math.pi / points
        for i in range(points * 2):
            r = outer_r if i % 2 == 0 else inner_r
            angle = i * angle_step - math.pi / 2
            result.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        return result

    def _sparkle_points(self, cx: int, cy: int, r: int) -> List[Tuple[int, int]]:
        """Generate 4-pointed sparkle"""
        inner_r = r // 4
        return [
            (cx, cy - r), (cx + inner_r, cy - inner_r),
            (cx + r, cy), (cx + inner_r, cy + inner_r),
            (cx, cy + r), (cx - inner_r, cy + inner_r),
            (cx - r, cy), (cx - inner_r, cy - inner_r)
        ]

    def _regular_polygon_points(self, cx: int, cy: int, r: int, sides: int) -> List[Tuple[int, int]]:
        """Generate regular polygon points"""
        result = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides - math.pi / 2
            result.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        return result

    def _heart_points(self, cx: int, cy: int, r: int) -> List[Tuple[int, int]]:
        """Generate heart shape points"""
        points = []
        for t in range(0, 360, 10):
            t_rad = math.radians(t)
            x = r * 0.5 * (16 * math.sin(t_rad) ** 3)
            y = -r * 0.5 * (13 * math.cos(t_rad) - 5 * math.cos(2*t_rad) - 2 * math.cos(3*t_rad) - math.cos(4*t_rad))
            points.append((int(cx + x), int(cy + y)))
        return points

    def _blend_layers(self, bg: Image.Image, fg: Image.Image, mode: BlendMode) -> Image.Image:
        """Blend foreground onto background"""
        bg_rgba = bg.convert('RGBA')

        if mode == BlendMode.NORMAL:
            bg_rgba.paste(fg, (0, 0), fg)
            return bg_rgba

        bg_np = np.array(bg_rgba).astype(np.float32)
        fg_np = np.array(fg).astype(np.float32)

        # Extract alpha channel
        fg_alpha = fg_np[:, :, 3:4] / 255

        if mode == BlendMode.ADD:
            result = bg_np + fg_np * fg_alpha
        elif mode == BlendMode.MULTIPLY:
            result = bg_np * (1 - fg_alpha) + (bg_np * fg_np / 255) * fg_alpha
        elif mode == BlendMode.SCREEN:
            result = bg_np * (1 - fg_alpha) + (255 - (255 - bg_np) * (255 - fg_np) / 255) * fg_alpha
        elif mode == BlendMode.OVERLAY:
            mask = bg_np < 128
            overlay = np.where(mask, 2 * bg_np * fg_np / 255, 255 - 2 * (255 - bg_np) * (255 - fg_np) / 255)
            result = bg_np * (1 - fg_alpha) + overlay * fg_alpha
        elif mode == BlendMode.SOFT_LIGHT:
            result = bg_np * (1 - fg_alpha) + ((255 - 2 * fg_np) * bg_np ** 2 / 255 + 2 * fg_np * bg_np) / 255 * fg_alpha
        else:
            result = bg_np

        result[:, :, 3] = 255  # Keep full alpha
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


# Preset particle effects
class ParticlePresets:
    """Pre-configured particle effect presets"""

    @staticmethod
    def confetti(width: int, height: int) -> ParticleSystem:
        """Colorful confetti celebration effect"""
        ps = ParticleSystem(width, height)

        colors = [
            (255, 87, 51, 255),   # Orange
            (255, 215, 0, 255),   # Gold
            (0, 191, 255, 255),   # Blue
            (255, 105, 180, 255), # Pink
            (50, 205, 50, 255),   # Green
        ]

        for color in colors:
            config = EmitterConfig(
                shape=EmitterShape.EDGE,
                position=(0.5, 0),
                emission_rate=20,
                particle_config=ParticleConfig(
                    shape=ParticleShape.SQUARE,
                    color=color,
                    size=15,
                    size_variance=0.5,
                    speed=200,
                    direction=270,
                    direction_variance=60,
                    lifetime=3.0,
                    gravity=(0, 150),
                    rotation_speed=180,
                    rotation_variance=90,
                    fade_out=0.3
                )
            )
            ps.add_emitter(config)

        return ps

    @staticmethod
    def sparkles(width: int, height: int) -> ParticleSystem:
        """Magical sparkle effect"""
        ps = ParticleSystem(width, height)

        config = EmitterConfig(
            shape=EmitterShape.RECTANGLE,
            position=(0.5, 0.5),
            size=(0.8, 0.8),
            emission_rate=30,
            particle_config=ParticleConfig(
                shape=ParticleShape.SPARKLE,
                color=(255, 255, 255, 255),
                color_variance=0.1,
                size=20,
                size_variance=0.6,
                size_over_life=[0.5, 1.0, 1.2, 0.8, 0.0],
                speed=20,
                direction_variance=180,
                lifetime=1.5,
                glow=0.8,
                fade_in=0.2,
                fade_out=0.4
            )
        )
        ps.add_emitter(config)

        return ps

    @staticmethod
    def fire(width: int, height: int, position: Tuple[float, float] = (0.5, 0.8)) -> ParticleSystem:
        """Fire/flame effect"""
        ps = ParticleSystem(width, height)

        # Core fire
        config = EmitterConfig(
            shape=EmitterShape.LINE,
            position=position,
            size=(0.1, 0),
            emission_rate=60,
            particle_config=ParticleConfig(
                shape=ParticleShape.GLOW,
                color=(255, 150, 50, 200),
                color_over_life=[
                    (255, 255, 200, 255),
                    (255, 180, 50, 200),
                    (255, 100, 20, 150),
                    (100, 50, 20, 100),
                    (50, 20, 10, 0)
                ],
                size=40,
                size_variance=0.4,
                size_over_life=[0.3, 0.8, 1.0, 0.6, 0.2],
                speed=100,
                direction=270,
                direction_variance=20,
                lifetime=1.0,
                gravity=(0, -50),
                blur=2
            )
        )
        ps.add_emitter(config)

        # Sparks
        spark_config = EmitterConfig(
            shape=EmitterShape.LINE,
            position=position,
            size=(0.05, 0),
            emission_rate=15,
            particle_config=ParticleConfig(
                shape=ParticleShape.CIRCLE,
                color=(255, 200, 100, 255),
                size=5,
                size_variance=0.5,
                speed=150,
                direction=270,
                direction_variance=40,
                lifetime=0.8,
                gravity=(0, -100),
                glow=0.5,
                fade_out=0.5
            )
        )
        ps.add_emitter(spark_config)

        return ps

    @staticmethod
    def snow(width: int, height: int) -> ParticleSystem:
        """Gentle snowfall effect"""
        ps = ParticleSystem(width, height)

        config = EmitterConfig(
            shape=EmitterShape.LINE,
            position=(0.5, -0.05),
            size=(1.2, 0),
            emission_rate=40,
            particle_config=ParticleConfig(
                shape=ParticleShape.CIRCLE,
                color=(255, 255, 255, 200),
                size=8,
                size_variance=0.6,
                speed=50,
                direction=90,
                direction_variance=15,
                lifetime=5.0,
                gravity=(10, 20),
                drag=0.02,
                blur=1,
                fade_in=0.1,
                fade_out=0.2
            )
        )
        ps.add_emitter(config)

        return ps

    @staticmethod
    def smoke(width: int, height: int, position: Tuple[float, float] = (0.5, 0.9)) -> ParticleSystem:
        """Smoke/mist effect"""
        ps = ParticleSystem(width, height)

        config = EmitterConfig(
            shape=EmitterShape.CIRCLE,
            position=position,
            size=(0.1, 0.1),
            emission_rate=15,
            particle_config=ParticleConfig(
                shape=ParticleShape.SMOKE,
                color=(200, 200, 200, 100),
                size=80,
                size_variance=0.3,
                size_over_life=[0.3, 0.6, 1.0, 1.2, 1.3],
                speed=30,
                direction=270,
                direction_variance=30,
                lifetime=4.0,
                gravity=(5, -10),
                drag=0.1,
                blur=5,
                fade_in=0.3,
                fade_out=0.5
            )
        )
        ps.add_emitter(config)

        return ps

    @staticmethod
    def dust_motes(width: int, height: int) -> ParticleSystem:
        """Floating dust particles in light"""
        ps = ParticleSystem(width, height)

        config = EmitterConfig(
            shape=EmitterShape.RECTANGLE,
            position=(0.5, 0.5),
            size=(1.0, 1.0),
            emission_rate=20,
            particle_config=ParticleConfig(
                shape=ParticleShape.GLOW,
                color=(255, 240, 200, 80),
                size=6,
                size_variance=0.5,
                speed=10,
                direction_variance=180,
                lifetime=8.0,
                gravity=(2, -1),
                drag=0.05,
                glow=0.3,
                fade_in=0.3,
                fade_out=0.3
            )
        )
        ps.add_emitter(config)

        return ps

    @staticmethod
    def energy_burst(width: int, height: int, position: Tuple[float, float] = (0.5, 0.5)) -> ParticleSystem:
        """Energy/power burst effect"""
        ps = ParticleSystem(width, height)

        config = EmitterConfig(
            shape=EmitterShape.POINT,
            position=position,
            burst_count=100,
            particle_config=ParticleConfig(
                shape=ParticleShape.LINE,
                color=(100, 200, 255, 255),
                color_variance=0.2,
                size=30,
                size_variance=0.5,
                speed=400,
                direction_variance=180,
                lifetime=0.8,
                drag=0.3,
                rotation_speed=0,
                glow=0.8,
                fade_out=0.6
            )
        )
        ps.add_emitter(config)

        return ps

    @staticmethod
    def hearts(width: int, height: int) -> ParticleSystem:
        """Floating hearts effect"""
        ps = ParticleSystem(width, height)

        config = EmitterConfig(
            shape=EmitterShape.LINE,
            position=(0.5, 1.1),
            size=(0.8, 0),
            emission_rate=10,
            particle_config=ParticleConfig(
                shape=ParticleShape.HEART,
                color=(255, 100, 150, 200),
                color_variance=0.2,
                size=30,
                size_variance=0.4,
                speed=80,
                direction=270,
                direction_variance=30,
                lifetime=4.0,
                gravity=(5, 0),
                rotation_speed=20,
                rotation_variance=10,
                fade_out=0.4
            )
        )
        ps.add_emitter(config)

        return ps


# Convenience functions
def create_particle_effect(effect_type: str, width: int, height: int,
                          **kwargs) -> ParticleSystem:
    """Create particle system by effect name"""
    effects = {
        'confetti': ParticlePresets.confetti,
        'sparkles': ParticlePresets.sparkles,
        'fire': ParticlePresets.fire,
        'snow': ParticlePresets.snow,
        'smoke': ParticlePresets.smoke,
        'dust': ParticlePresets.dust_motes,
        'energy': ParticlePresets.energy_burst,
        'hearts': ParticlePresets.hearts,
    }

    creator = effects.get(effect_type.lower())
    if creator:
        return creator(width, height, **kwargs) if kwargs else creator(width, height)

    # Return default sparkles
    return ParticlePresets.sparkles(width, height)


def render_particles(background: Image.Image, effect_type: str,
                    progress: float, duration: float = 5.0,
                    **kwargs) -> Image.Image:
    """Render particle effect at specific progress point"""
    ps = create_particle_effect(effect_type, background.width, background.height, **kwargs)

    # Simulate up to current time
    current_time = progress * duration
    dt = 1/30  # 30fps simulation steps

    t = 0
    while t < current_time:
        ps.update(dt)
        t += dt

    return ps.render(background, BlendMode.ADD)
