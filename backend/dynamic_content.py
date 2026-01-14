"""
Dynamic Content Renderer
Renders animated pricing, countdown timers, star ratings, review quotes, and CTAs
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime
import math


@dataclass
class PriceDisplay:
    """Configuration for animated price display"""
    original_price: str
    sale_price: Optional[str] = None
    currency: str = "$"
    show_discount_percent: bool = True
    animation: str = "drop"  # drop, slide, flash, bounce


@dataclass
class CountdownTimer:
    """Configuration for countdown timer"""
    total_seconds: int
    label: str = "OFFER ENDS IN"
    style: str = "flip"  # flip, digital, minimal, urgent


@dataclass
class StarRating:
    """Configuration for star rating display"""
    rating: float  # 0-5
    review_count: Optional[int] = None
    animation: str = "fill"  # fill, pop, glow, cascade


@dataclass
class ReviewQuote:
    """Configuration for review quote display"""
    quote: str
    author: str
    avatar_url: Optional[str] = None
    animation: str = "typewriter"  # typewriter, fade, slide


@dataclass
class CTAButton:
    """Configuration for CTA button"""
    text: str
    style: str = "pulse"  # pulse, shake, glow, swipe_up, bounce
    icon: Optional[str] = None  # arrow, cart, heart, none
    color: Tuple[int, int, int] = (255, 87, 51)  # Default orange-red


class Easing:
    """Easing functions for smooth animations"""

    @staticmethod
    def ease_out_cubic(t: float) -> float:
        return 1 - pow(1 - t, 3)

    @staticmethod
    def ease_out_bounce(t: float) -> float:
        if t < 1/2.75:
            return 7.5625 * t * t
        elif t < 2/2.75:
            t -= 1.5/2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5/2.75:
            t -= 2.25/2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625/2.75
            return 7.5625 * t * t + 0.984375

    @staticmethod
    def ease_out_elastic(t: float) -> float:
        if t == 0 or t == 1:
            return t
        return pow(2, -10 * t) * math.sin((t - 0.075) * (2 * math.pi) / 0.3) + 1

    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2


class DynamicContentRenderer:
    """Renders dynamic content elements on video frames"""

    def __init__(self):
        self.font_cache = {}

    def _get_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get cached font"""
        key = (size, bold)
        if key not in self.font_cache:
            try:
                font_path = "/System/Library/Fonts/Helvetica.ttc" if not bold else "/System/Library/Fonts/HelveticaNeue.ttc"
                self.font_cache[key] = ImageFont.truetype(font_path, size)
            except:
                self.font_cache[key] = ImageFont.load_default()
        return self.font_cache[key]

    def _pil_to_cv2(self, pil_img: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)

    def _cv2_to_pil(self, cv2_img: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format"""
        if len(cv2_img.shape) == 3 and cv2_img.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA))
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    def render_price_animation(
        self,
        frame: np.ndarray,
        config: PriceDisplay,
        progress: float,
        position: Tuple[int, int],
        style_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Render animated price with strikethrough and sale price drop effect

        Timeline:
        - 0-30%: Show original price
        - 30-50%: Strikethrough animation
        - 50-100%: Sale price drops in with highlight
        """
        pil_img = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        x, y = position
        original_font = self._get_font(48)
        sale_font = self._get_font(72, bold=True)
        small_font = self._get_font(24)

        original_text = f"{config.currency}{config.original_price}"

        # Calculate discount percentage
        discount_percent = 0
        if config.sale_price and config.show_discount_percent:
            try:
                orig = float(config.original_price.replace(',', ''))
                sale = float(config.sale_price.replace(',', ''))
                discount_percent = int((1 - sale / orig) * 100)
            except:
                pass

        if config.animation == "drop":
            # Phase 1: Original price appears
            if progress < 0.3:
                phase_progress = progress / 0.3
                alpha = int(255 * Easing.ease_out_cubic(phase_progress))
                draw.text((x, y), original_text, fill=(200, 200, 200, alpha), font=original_font)

            # Phase 2: Strikethrough animation
            elif progress < 0.5:
                phase_progress = (progress - 0.3) / 0.2
                draw.text((x, y), original_text, fill=(150, 150, 150, 255), font=original_font)

                # Animated strikethrough
                bbox = draw.textbbox((x, y), original_text, font=original_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                strike_width = int(text_width * Easing.ease_out_cubic(phase_progress))
                strike_y = y + text_height // 2
                draw.line([(x, strike_y), (x + strike_width, strike_y)], fill=(255, 80, 80, 255), width=3)

            # Phase 3: Sale price drops in
            else:
                # Keep original with full strikethrough
                draw.text((x, y), original_text, fill=(120, 120, 120, 255), font=original_font)
                bbox = draw.textbbox((x, y), original_text, font=original_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                strike_y = y + text_height // 2
                draw.line([(x, strike_y), (x + text_width, strike_y)], fill=(255, 80, 80, 255), width=3)

                if config.sale_price:
                    phase_progress = (progress - 0.5) / 0.5
                    sale_text = f"{config.currency}{config.sale_price}"

                    # Drop animation with bounce
                    drop_offset = int(100 * (1 - Easing.ease_out_bounce(phase_progress)))
                    sale_y = y + text_height + 20 - drop_offset

                    # Glow effect
                    if phase_progress > 0.5:
                        glow_alpha = int(100 * math.sin((phase_progress - 0.5) * math.pi))
                        for offset in range(3, 0, -1):
                            draw.text(
                                (x - offset, sale_y - offset),
                                sale_text,
                                fill=(255, 200, 0, glow_alpha // offset),
                                font=sale_font
                            )

                    draw.text((x, sale_y), sale_text, fill=(255, 220, 50, 255), font=sale_font)

                    # Discount badge
                    if discount_percent > 0 and phase_progress > 0.3:
                        badge_alpha = int(255 * min(1, (phase_progress - 0.3) / 0.3))
                        badge_text = f"-{discount_percent}%"
                        sale_bbox = draw.textbbox((x, sale_y), sale_text, font=sale_font)
                        badge_x = sale_bbox[2] + 15
                        badge_y = sale_y + 10

                        # Badge background
                        badge_bbox = draw.textbbox((badge_x, badge_y), badge_text, font=small_font)
                        padding = 8
                        draw.rounded_rectangle(
                            [badge_bbox[0] - padding, badge_bbox[1] - padding,
                             badge_bbox[2] + padding, badge_bbox[3] + padding],
                            radius=5,
                            fill=(255, 50, 50, badge_alpha)
                        )
                        draw.text((badge_x, badge_y), badge_text, fill=(255, 255, 255, badge_alpha), font=small_font)

        elif config.animation == "flash":
            # Flash animation - price appears with bright flash
            if progress < 0.2:
                # Flash in
                flash_intensity = int(255 * (1 - progress / 0.2))
                overlay = Image.new('RGBA', pil_img.size, (255, 255, 255, flash_intensity))
                pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
                draw = ImageDraw.Draw(pil_img, 'RGBA')

            if config.sale_price:
                sale_text = f"{config.currency}{config.sale_price}"
                draw.text((x, y), sale_text, fill=(255, 220, 50, 255), font=sale_font)

                # Original price smaller above
                draw.text((x, y - 40), original_text, fill=(150, 150, 150, 200), font=small_font)
                bbox = draw.textbbox((x, y - 40), original_text, font=small_font)
                draw.line([(bbox[0], (bbox[1] + bbox[3])//2), (bbox[2], (bbox[1] + bbox[3])//2)],
                         fill=(255, 80, 80, 200), width=2)

        return self._pil_to_cv2(pil_img)

    def render_countdown_timer(
        self,
        frame: np.ndarray,
        config: CountdownTimer,
        progress: float,
        position: Tuple[int, int],
        style_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """Render flip-clock style countdown timer"""
        pil_img = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        x, y = position

        # Calculate remaining time based on progress (decreasing)
        remaining = int(config.total_seconds * (1 - progress * 0.9))  # Don't go to 0

        hours = remaining // 3600
        minutes = (remaining % 3600) // 60
        seconds = remaining % 60

        label_font = self._get_font(20)
        digit_font = self._get_font(48, bold=True)
        small_font = self._get_font(14)

        # Draw label
        draw.text((x, y), config.label, fill=(255, 255, 255, 200), font=label_font)

        digit_y = y + 35
        segment_width = 70

        if config.style == "flip":
            # Flip clock style
            segments = [
                (hours, "HRS"),
                (minutes, "MIN"),
                (seconds, "SEC")
            ]

            for i, (value, label) in enumerate(segments):
                seg_x = x + i * (segment_width + 15)

                # Background card
                draw.rounded_rectangle(
                    [seg_x, digit_y, seg_x + segment_width, digit_y + 60],
                    radius=8,
                    fill=(30, 30, 30, 230)
                )

                # Digit
                digit_text = f"{value:02d}"
                bbox = draw.textbbox((0, 0), digit_text, font=digit_font)
                text_w = bbox[2] - bbox[0]
                draw.text(
                    (seg_x + (segment_width - text_w) // 2, digit_y + 5),
                    digit_text,
                    fill=style_color,
                    font=digit_font
                )

                # Label below
                label_bbox = draw.textbbox((0, 0), label, font=small_font)
                label_w = label_bbox[2] - label_bbox[0]
                draw.text(
                    (seg_x + (segment_width - label_w) // 2, digit_y + 65),
                    label,
                    fill=(180, 180, 180, 255),
                    font=small_font
                )

                # Colon separator
                if i < len(segments) - 1:
                    colon_x = seg_x + segment_width + 3
                    # Blinking effect
                    colon_alpha = 255 if int(progress * 10) % 2 == 0 else 150
                    draw.text((colon_x, digit_y + 10), ":", fill=(255, 255, 255, colon_alpha), font=digit_font)

        elif config.style == "urgent":
            # Urgent pulsing style
            pulse = 0.8 + 0.2 * math.sin(progress * math.pi * 8)

            # Red background
            total_width = 250
            draw.rounded_rectangle(
                [x, digit_y, x + total_width, digit_y + 70],
                radius=10,
                fill=(180, 30, 30, int(200 * pulse))
            )

            # Time string
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            bbox = draw.textbbox((0, 0), time_str, font=digit_font)
            text_w = bbox[2] - bbox[0]
            draw.text(
                (x + (total_width - text_w) // 2, digit_y + 10),
                time_str,
                fill=(255, 255, 255, 255),
                font=digit_font
            )

        elif config.style == "minimal":
            # Clean minimal style
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            draw.text((x, digit_y), time_str, fill=style_color, font=digit_font)

        return self._pil_to_cv2(pil_img)

    def render_star_rating(
        self,
        frame: np.ndarray,
        config: StarRating,
        progress: float,
        position: Tuple[int, int],
        style_color: Tuple[int, int, int] = (255, 200, 50)
    ) -> np.ndarray:
        """Render animated star rating"""
        pil_img = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        x, y = position
        star_size = 32
        star_gap = 8

        rating_font = self._get_font(36, bold=True)
        count_font = self._get_font(20)

        def draw_star(cx: int, cy: int, size: int, fill_percent: float, glow: bool = False):
            """Draw a star shape"""
            points = []
            for i in range(5):
                # Outer points
                angle = math.radians(-90 + i * 72)
                points.append((
                    cx + int(size * math.cos(angle)),
                    cy + int(size * math.sin(angle))
                ))
                # Inner points
                angle = math.radians(-90 + i * 72 + 36)
                points.append((
                    cx + int(size * 0.4 * math.cos(angle)),
                    cy + int(size * 0.4 * math.sin(angle))
                ))

            # Draw glow
            if glow:
                for offset in range(4, 0, -1):
                    glow_points = [(p[0], p[1]) for p in points]
                    draw.polygon(glow_points, fill=(255, 200, 50, 30))

            # Draw star
            if fill_percent >= 1:
                draw.polygon(points, fill=style_color)
            elif fill_percent <= 0:
                draw.polygon(points, fill=(80, 80, 80, 200))
            else:
                # Partial fill (draw gray first, then overlay filled portion)
                draw.polygon(points, fill=(80, 80, 80, 200))
                # Simple partial - just adjust alpha
                partial_color = (*style_color[:3], int(255 * fill_percent))
                draw.polygon(points, fill=partial_color)

        # Animation phases
        if config.animation == "fill":
            # Stars fill in sequence
            for i in range(5):
                star_x = x + i * (star_size * 2 + star_gap) + star_size
                star_y = y + star_size

                # Calculate fill for this star
                star_progress = progress * 5 - i
                fill = max(0, min(1, star_progress))

                # Determine if star should be filled based on rating
                if i < int(config.rating):
                    target_fill = 1.0
                elif i == int(config.rating):
                    target_fill = config.rating - int(config.rating)
                else:
                    target_fill = 0.0

                actual_fill = target_fill * fill
                glow = fill > 0.5 and target_fill > 0

                draw_star(star_x, star_y, star_size, actual_fill, glow)

        elif config.animation == "pop":
            # Stars pop in with scale
            for i in range(5):
                star_delay = i * 0.15
                star_progress = max(0, min(1, (progress - star_delay) / 0.2))
                scale = Easing.ease_out_elastic(star_progress) if star_progress > 0 else 0

                star_x = x + i * (star_size * 2 + star_gap) + star_size
                star_y = y + star_size

                if i < int(config.rating):
                    target_fill = 1.0
                elif i == int(config.rating):
                    target_fill = config.rating - int(config.rating)
                else:
                    target_fill = 0.0

                if scale > 0:
                    draw_star(star_x, star_y, int(star_size * scale), target_fill, target_fill > 0)

        # Rating number and count
        if progress > 0.7:
            text_alpha = int(255 * min(1, (progress - 0.7) / 0.3))
            rating_text = f"{config.rating:.1f}"
            rating_x = x + 5 * (star_size * 2 + star_gap) + 20

            draw.text((rating_x, y), rating_text, fill=(*style_color, text_alpha), font=rating_font)

            if config.review_count:
                count_text = f"({config.review_count:,} reviews)"
                draw.text((rating_x, y + 40), count_text, fill=(180, 180, 180, text_alpha), font=count_font)

        return self._pil_to_cv2(pil_img)

    def render_review_quote(
        self,
        frame: np.ndarray,
        config: ReviewQuote,
        progress: float,
        position: Tuple[int, int],
        max_width: int = 600
    ) -> np.ndarray:
        """Render animated review quote with typewriter effect"""
        pil_img = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        x, y = position
        quote_font = self._get_font(28)
        author_font = self._get_font(22)
        quote_mark_font = self._get_font(72)

        # Background card
        card_padding = 30
        card_height = 180

        if progress > 0.05:
            card_alpha = int(200 * min(1, progress / 0.2))
            draw.rounded_rectangle(
                [x, y, x + max_width, y + card_height],
                radius=15,
                fill=(20, 20, 20, card_alpha)
            )

        if config.animation == "typewriter":
            # Quote marks appear first
            if progress > 0.1:
                quote_alpha = int(255 * min(1, (progress - 0.1) / 0.1))
                draw.text((x + 15, y + 10), '"', fill=(255, 200, 50, quote_alpha), font=quote_mark_font)

            # Typewriter effect for quote
            if progress > 0.2:
                type_progress = (progress - 0.2) / 0.5
                chars_to_show = int(len(config.quote) * min(1, type_progress))
                visible_quote = config.quote[:chars_to_show]

                # Word wrap
                words = visible_quote.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    bbox = draw.textbbox((0, 0), test_line, font=quote_font)
                    if bbox[2] - bbox[0] < max_width - 80:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)

                quote_y = y + 50
                for line in lines:
                    draw.text((x + 30, quote_y), line, fill=(255, 255, 255, 255), font=quote_font)
                    quote_y += 35

                # Cursor blink
                if type_progress < 1:
                    if int(progress * 10) % 2 == 0:
                        bbox = draw.textbbox((x + 30, quote_y - 35), lines[-1] if lines else "", font=quote_font)
                        cursor_x = bbox[2] + 5
                        draw.rectangle([cursor_x, quote_y - 30, cursor_x + 3, quote_y - 5], fill=(255, 255, 255, 255))

            # Author appears
            if progress > 0.75:
                author_alpha = int(255 * min(1, (progress - 0.75) / 0.25))
                author_text = f"- {config.author}"
                draw.text((x + 30, y + card_height - 45), author_text, fill=(180, 180, 180, author_alpha), font=author_font)

        return self._pil_to_cv2(pil_img)

    def render_cta_button(
        self,
        frame: np.ndarray,
        config: CTAButton,
        progress: float,
        position: Tuple[int, int]
    ) -> np.ndarray:
        """Render animated CTA button"""
        pil_img = self._cv2_to_pil(frame)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        x, y = position
        button_font = self._get_font(32, bold=True)

        # Calculate button dimensions
        text_bbox = draw.textbbox((0, 0), config.text, font=button_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        padding_x = 50
        padding_y = 20
        btn_width = text_w + padding_x * 2
        btn_height = text_h + padding_y * 2

        # Animation effects
        offset_x = 0
        offset_y = 0
        scale = 1.0
        glow_intensity = 0

        if config.style == "pulse":
            # Pulsing glow effect
            pulse = 0.9 + 0.1 * math.sin(progress * math.pi * 4)
            scale = pulse
            glow_intensity = int(50 + 30 * math.sin(progress * math.pi * 4))

        elif config.style == "shake":
            # Shake effect
            if progress > 0.3:
                shake_progress = (progress - 0.3) * 10
                offset_x = int(5 * math.sin(shake_progress * math.pi * 3) * (1 - progress))

        elif config.style == "bounce":
            # Bounce in
            if progress < 0.3:
                bounce_progress = progress / 0.3
                scale = Easing.ease_out_bounce(bounce_progress)
                offset_y = int(50 * (1 - scale))

        elif config.style == "glow":
            # Intensifying glow
            glow_intensity = int(80 * progress)

        elif config.style == "swipe_up":
            # Swipe up animation
            if progress < 0.4:
                offset_y = int(100 * (1 - Easing.ease_out_cubic(progress / 0.4)))

            # Arrow animation after button appears
            if progress > 0.5:
                arrow_progress = (progress - 0.5) / 0.5
                arrow_offset = int(10 * math.sin(arrow_progress * math.pi * 4))
                arrow_y = y - 40 + arrow_offset
                # Draw up arrow
                arrow_points = [
                    (x + btn_width // 2 - 15, arrow_y + 15),
                    (x + btn_width // 2, arrow_y),
                    (x + btn_width // 2 + 15, arrow_y + 15)
                ]
                draw.line(arrow_points, fill=(255, 255, 255, 200), width=3)

        # Calculate final position with offsets
        final_x = x + offset_x
        final_y = y + offset_y

        # Apply scale
        scaled_width = int(btn_width * scale)
        scaled_height = int(btn_height * scale)
        scale_offset_x = (btn_width - scaled_width) // 2
        scale_offset_y = (btn_height - scaled_height) // 2

        # Draw glow
        if glow_intensity > 0:
            for i in range(3, 0, -1):
                glow_rect = [
                    final_x + scale_offset_x - i * 3,
                    final_y + scale_offset_y - i * 3,
                    final_x + scale_offset_x + scaled_width + i * 3,
                    final_y + scale_offset_y + scaled_height + i * 3
                ]
                draw.rounded_rectangle(
                    glow_rect,
                    radius=15,
                    fill=(*config.color, glow_intensity // i)
                )

        # Draw button background
        btn_rect = [
            final_x + scale_offset_x,
            final_y + scale_offset_y,
            final_x + scale_offset_x + scaled_width,
            final_y + scale_offset_y + scaled_height
        ]
        draw.rounded_rectangle(btn_rect, radius=12, fill=(*config.color, 255))

        # Draw text
        text_x = final_x + scale_offset_x + (scaled_width - text_w) // 2
        text_y = final_y + scale_offset_y + (scaled_height - text_h) // 2 - 5
        draw.text((text_x, text_y), config.text, fill=(255, 255, 255, 255), font=button_font)

        # Draw icon if specified
        if config.icon:
            icon_x = text_x + text_w + 15
            icon_y = text_y + text_h // 2

            if config.icon == "arrow":
                # Right arrow
                arrow_points = [
                    (icon_x, icon_y - 8),
                    (icon_x + 12, icon_y),
                    (icon_x, icon_y + 8)
                ]
                draw.polygon(arrow_points, fill=(255, 255, 255, 255))
            elif config.icon == "cart":
                # Simple cart icon
                draw.rectangle([icon_x, icon_y - 6, icon_x + 14, icon_y + 6], outline=(255, 255, 255, 255), width=2)
                draw.ellipse([icon_x + 2, icon_y + 8, icon_x + 6, icon_y + 12], fill=(255, 255, 255, 255))
                draw.ellipse([icon_x + 10, icon_y + 8, icon_x + 14, icon_y + 12], fill=(255, 255, 255, 255))

        return self._pil_to_cv2(pil_img)


# Global instance
dynamic_renderer = DynamicContentRenderer()


def render_price(frame, original, sale, progress, position, animation="drop"):
    """Convenience function for price rendering"""
    config = PriceDisplay(original_price=original, sale_price=sale, animation=animation)
    return dynamic_renderer.render_price_animation(frame, config, progress, position)


def render_countdown(frame, seconds, progress, position, style="flip"):
    """Convenience function for countdown rendering"""
    config = CountdownTimer(total_seconds=seconds, style=style)
    return dynamic_renderer.render_countdown_timer(frame, config, progress, position)


def render_rating(frame, rating, progress, position, review_count=None, animation="fill"):
    """Convenience function for rating rendering"""
    config = StarRating(rating=rating, review_count=review_count, animation=animation)
    return dynamic_renderer.render_star_rating(frame, config, progress, position)


def render_quote(frame, quote, author, progress, position, animation="typewriter"):
    """Convenience function for quote rendering"""
    config = ReviewQuote(quote=quote, author=author, animation=animation)
    return dynamic_renderer.render_review_quote(frame, config, progress, position)


def render_cta(frame, text, progress, position, style="pulse", color=(255, 87, 51)):
    """Convenience function for CTA rendering"""
    config = CTAButton(text=text, style=style, color=color)
    return dynamic_renderer.render_cta_button(frame, config, progress, position)
