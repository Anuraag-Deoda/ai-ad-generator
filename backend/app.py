import shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import validators
import logging
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import textwrap
import uuid
import os
import json
import subprocess
from dotenv import load_dotenv
from scraper import scrape_product_data
from ai_service import (
    generate_ad_script,
    generate_ad_script_v2,
    generate_video_metadata,
    generate_hook_variations,
    generate_ab_hooks_enhanced,
    analyze_product_sentiment,
    get_industry_config,
    get_platform_config,
    INDUSTRY_PROMPTS,
    PLATFORM_COPY_GUIDELINES,
    AIServiceError
)
from advanced_video_generator import generate_advanced_video, AdvancedVideoGenerator, VideoConfig, DynamicContentConfig
from dynamic_content import (
    DynamicContentRenderer, PriceDisplay, CountdownTimer,
    StarRating, ReviewQuote, CTAButton
)
from industry_templates import (
    IndustryTemplateRenderer, INDUSTRY_TEMPLATES, COLOR_SCHEMES,
    list_templates, list_industries
)
from image_processor import ImageProcessor, process_product_image
from brand_kit import BrandKit, BrandKitManager, apply_brand_kit_to_frame
from audio_engine import AudioEngine, AudioSettings, generate_video_with_audio
from platform_variants import (
    PLATFORM_CONFIGS, get_all_platforms, get_platform_config as get_variant_platform_config,
    VariantGenerator, generate_platform_variants
)
from hooks_database import HooksManager, get_emotional_triggers, get_platform_guidelines


load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# In-memory cache for recent scrapes (in production, use Redis)
scrape_cache = {}
CACHE_DURATION = 300  # 5 minutes

@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze product URL and return structured data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        url = data.get('url', '').strip()

        # Validate URL
        if not url:
            return jsonify({'error': 'URL is required'}), 400
            
        if not validators.url(url):
            return jsonify({'error': 'Invalid URL format'}), 400

        # Check cache first
        cache_key = url.lower()
        if cache_key in scrape_cache:
            cache_entry = scrape_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < CACHE_DURATION:
                logger.info(f"Returning cached result for {url}")
                return jsonify({
                    'cached': True,
                    **cache_entry['data']
                }), 200

        # Log the scraping attempt
        logger.info(f"Starting scrape for URL: {url}")
        
        # Perform scraping
        result = scrape_product_data(url)

        # Handle errors
        if 'error' in result:
            logger.error(f"Scraping failed for {url}: {result['error']}")
            return jsonify({'error': result['error']}), 500

        # Validate minimum required data
        if not result.get('title') and not result.get('price'):
            return jsonify({'error': 'Could not extract sufficient product data'}), 500

        # Cache successful result
        scrape_cache[cache_key] = {
            'timestamp': time.time(),
            'data': result
        }

        # Clean up old cache entries (simple cleanup)
        current_time = time.time()
        expired_keys = [k for k, v in scrape_cache.items() 
                       if current_time - v['timestamp'] > CACHE_DURATION]
        for key in expired_keys:
            del scrape_cache[key]

        logger.info(f"Successfully scraped product: {result.get('title', 'Unknown')[:50]}")
        
        return jsonify({
            'success': True,
            'cached': False,
            **result
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in analyze_url: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500


@app.route('/api/generate-content', methods=['POST'])
def generate_content():
    """Enhanced content generation with multiple AI features"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        job_id = data.get('job_id') or str(uuid.uuid4())
        product = data.get('product')
        
        # Enhanced parameters
        style = data.get('style', 'energetic')  # energetic, professional, casual, luxury, playful
        duration = data.get('duration', 30)     # 15, 30, or 60 seconds
        include_variations = data.get('include_variations', False)
        include_metadata = data.get('include_metadata', True)

        if not product or not product.get('title'):
            return jsonify({'error': 'Invalid product data - title required'}), 400

        logger.info(f"Generating content for job {job_id} with style: {style}, duration: {duration}s")

        # Generate main script
        script = generate_ad_script(product, style=style, duration=duration)
        
        # Generate additional content based on parameters
        output = {
            "job_id": job_id,
            "title": product.get('title'),
            "price": product.get('price'),
            "images": product.get('images'),
            "script": script,
            "style": style,
            "duration": duration,
            "generated_at": time.time()
        }

        # Add hook variations for A/B testing
        if include_variations:
            try:
                output["hook_variations"] = generate_hook_variations(product, count=5)
                logger.info("Generated hook variations successfully")
            except Exception as e:
                logger.warning(f"Failed to generate hook variations: {str(e)}")
                output["hook_variations"] = []

        # Add video production metadata
        if include_metadata:
            try:
                output["video_metadata"] = generate_video_metadata(product, script)
                output["product_analysis"] = analyze_product_sentiment(product)
                logger.info("Generated video metadata and product analysis")
            except Exception as e:
                logger.warning(f"Failed to generate metadata: {str(e)}")
                output["video_metadata"] = {}
                output["product_analysis"] = {}

        # Save to file for Remotion
        os.makedirs("data/jobs", exist_ok=True)
        with open(f"data/jobs/{job_id}.json", "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Successfully generated content for: {product.get('title', 'Unknown')[:50]}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "script": script,
            "style": style,
            "duration": duration,
            "has_metadata": include_metadata,
            "has_variations": include_variations
        }), 200


    except Exception as e:
        logger.error(f"Error optimizing script: {str(e)}")
        return jsonify({'error': 'Failed to optimize script'}), 500


@app.route('/api/batch-generate', methods=['POST'])
def batch_generate():
    """Generate multiple ad variations for the same product"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        product = data.get('product')
        styles = data.get('styles', ['energetic', 'professional', 'casual'])
        durations = data.get('durations', [30])
        
        if not product or not product.get('title'):
            return jsonify({'error': 'Invalid product data'}), 400

        batch_id = str(uuid.uuid4())
        results = []
        
        logger.info(f"Starting batch generation for {len(styles)} styles and {len(durations)} durations")

        for style in styles:
            for duration in durations:
                try:
                    job_id = f"{batch_id}_{style}_{duration}s"
                    
                    # Generate script for this combination
                    script = generate_ad_script(product, style=style, duration=duration)
                    
                    # Generate metadata
                    video_metadata = generate_video_metadata(product, script)
                    
                    output = {
                        "job_id": job_id,
                        "batch_id": batch_id,
                        "title": product.get('title'),
                        "price": product.get('price'),
                        "images": product.get('images'),
                        "script": script,
                        "video_metadata": video_metadata,
                        "style": style,
                        "duration": duration,
                        "generated_at": time.time()
                    }

                    # Save individual file
                    os.makedirs("remotion/input", exist_ok=True)
                    with open(f"remotion/input/{job_id}.json", "w") as f:
                        json.dump(output, f, indent=2)

                    results.append({
                        "job_id": job_id,
                        "style": style,
                        "duration": duration,
                        "status": "success",
                        "script_preview": {
                            "hook": script.get('hook', '')[:100] + "..." if len(script.get('hook', '')) > 100 else script.get('hook', ''),
                            "cta": script.get('cta', '')
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to generate {style}-{duration}s combination: {str(e)}")
                    results.append({
                        "job_id": f"{batch_id}_{style}_{duration}s",
                        "style": style,
                        "duration": duration,
                        "status": "failed",
                        "error": str(e)
                    })

        # Save batch summary
        batch_summary = {
            "batch_id": batch_id,
            "product_title": product.get('title'),
            "total_variations": len(results),
            "successful": len([r for r in results if r['status'] == 'success']),
            "failed": len([r for r in results if r['status'] == 'failed']),
            "results": results,
            "created_at": time.time()
        }

        with open(f"remotion/input/batch_{batch_id}.json", "w") as f:
            json.dump(batch_summary, f, indent=2)

        logger.info(f"Batch generation completed: {batch_summary['successful']}/{batch_summary['total_variations']} successful")

        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "total_variations": len(results),
            "successful": batch_summary['successful'],
            "failed": batch_summary['failed'],
            "results": results
        }), 200

    except Exception as e:
        logger.error(f"Error in batch generation: {str(e)}")
        return jsonify({'error': 'Batch generation failed'}), 500


@app.route('/api/batch-status/<batch_id>', methods=['GET'])
def get_batch_status(batch_id):
    """Get status of batch generation"""
    try:
        batch_path = f"remotion/input/batch_{batch_id}.json"
        
        if not os.path.exists(batch_path):
            return jsonify({'error': 'Batch not found'}), 404

        with open(batch_path, 'r') as f:
            batch_data = json.load(f)

        return jsonify({
            "success": True,
            "batch_data": batch_data
        }), 200

    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        return jsonify({'error': 'Failed to get batch status'}), 500




# Advanced video generation with professional effects
@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    """Generate video using advanced effects engine"""
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        use_advanced = data.get('advanced', True)  # Use advanced by default

        if not job_id:
            return jsonify({'error': 'job_id required'}), 400

        input_path = f"data/jobs/{job_id}.json"
        if not os.path.exists(input_path):
            return jsonify({'error': 'Job not found'}), 404

        with open(input_path, 'r') as f:
            job_data = json.load(f)

        # Use advanced video generator with professional effects
        if use_advanced:
            logger.info(f"Using ADVANCED video generator for job {job_id}")
            video_path = generate_advanced_video(job_data)
        else:
            logger.info(f"Using basic video generator for job {job_id}")
            video_path = generate_ffmpeg_video(job_data)

        if video_path:
            return jsonify({
                "success": True,
                "video_path": video_path,
                "job_id": job_id,
                "advanced": use_advanced
            }), 200
        else:
            return jsonify({'error': 'Video generation failed'}), 500

    except Exception as e:
        logger.error(f"Error in generate_video: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


# Legacy basic video generation (kept for fallback)
@app.route('/api/generate-video-basic', methods=['POST'])
def generate_video_basic():
    """Generate video using basic FFmpeg (legacy fallback)"""
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        if not job_id:
            return jsonify({'error': 'job_id required'}), 400

        input_path = f"data/jobs/{job_id}.json"
        if not os.path.exists(input_path):
            return jsonify({'error': 'Job not found'}), 404

        with open(input_path, 'r') as f:
            job_data = json.load(f)

        video_path = generate_ffmpeg_video(job_data)

        if video_path:
            return jsonify({
                "success": True,
                "video_path": video_path,
                "job_id": job_id
            }), 200
        else:
            return jsonify({'error': 'Video generation failed'}), 500

    except Exception as e:
        logger.error(f"Error in generate_video_basic: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
 
 
def generate_ffmpeg_video(job_data):
    """Generate video using OpenCV and convert to web-compatible format"""
    try:
        job_id = job_data.get('job_id')
        script = job_data.get('script', {})
        product_images = job_data.get('images', [])
        duration = job_data.get('duration', 30)
        product_title = job_data.get('title', 'Amazing Product')

        # Prepare output paths
        os.makedirs("static/videos", exist_ok=True)
        temp_path = f"static/videos/{job_id}_temp.mp4"
        final_path = f"static/videos/{job_id}.mp4"

        # Video parameters
        width, height = 1080, 1920
        fps = 30
        total_frames = duration * fps

        # Use browser-safe codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            logger.error("❌ Failed to open VideoWriter. Check codec or file path.")
            return None

        # Load up to 3 product images
        images = []
        for img_url in product_images[:3]:
            try:
                response = requests.get(img_url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0'
                })
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img = img.resize((800, 800), Image.Resampling.LANCZOS)
                images.append(img)
                logger.info(f"✅ Loaded image: {img_url[:50]}...")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load image {img_url}: {str(e)}")

        if not images:
            images.append(create_placeholder_image(product_title))

        logger.info(f"🎬 Generating {total_frames} frames for {duration}s video...")

        for frame_num in range(total_frames):
            if frame_num % 150 == 0:
                logger.info(f"🧩 {frame_num}/{total_frames} frames done ({frame_num/total_frames:.0%})")

            frame = create_animated_frame(script, images, frame_num, total_frames, width, height, job_data)
            frame_np = np.array(frame)
            frame_cv2 = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_cv2)

        video_writer.release()
        cv2.destroyAllWindows()

        if not os.path.exists(temp_path):
            logger.error("❌ Temp video not created by OpenCV.")
            return None

        success = convert_to_web_format(temp_path, final_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if success and os.path.exists(final_path):
            logger.info(f"✅ Final video generated: {final_path}")
            return f"/static/videos/{job_id}.mp4"
        else:
            logger.error("❌ Final video conversion failed.")
            return None

    except Exception as e:
        logger.error(f"🔥 Exception in video generation: {str(e)}")
        return None


def convert_to_web_format(input_path, output_path):
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error:\n{result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"FFmpeg conversion failed: {str(e)}")
        return False


def create_placeholder_image(title):
    """Create a placeholder product image"""
    img = Image.new('RGB', (800, 800), color=(45, 45, 45))
    draw = ImageDraw.Draw(img)
 
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
 
    # Add gradient background
    for y in range(800):
        color_val = int(45 + (y / 800) * 40)
        draw.line([(0, y), (800, y)], fill=(color_val, color_val, color_val))
 
    # Add text
    wrapped_title = textwrap.fill(title, width=20)
    bbox = draw.multiline_textbbox((0, 0), wrapped_title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
 
    x = (800 - text_width) // 2
    y = (800 - text_height) // 2
 
    draw.multiline_text((x, y), wrapped_title, fill=(255, 255, 255), font=font, align='center')
 
    return img
 
 
def create_animated_frame(script, images, frame_num, total_frames, width, height, job_data):
    """Create individual animated frame"""
    # Calculate progress (0.0 to 1.0)
    progress = frame_num / total_frames
 
    # Determine current scene
    scene_info = get_current_scene(progress, script)
 
    # Create base frame with gradient background
    frame = create_gradient_background(width, height, scene_info['type'])
 
    # Add animated product image
    add_animated_product_image(frame, images, scene_info, frame_num, width, height)
 
    # Add animated text with proper sizing and effects
    add_animated_text(frame, scene_info, frame_num, width, height)
 
    # Add scene-specific effects
    add_advanced_effects(frame, scene_info, frame_num, width, height)
 
    return frame
 
 
def get_current_scene(progress, script):
    """Determine current scene based on progress"""
    if progress < 0.2:
        return {
            'type': 'hook',
            'text': script.get('hook', 'Amazing Product Alert! 🚨'),
            'progress': progress / 0.2,
            'color': (255, 215, 0)  # Gold
        }
    elif progress < 0.5:
        return {
            'type': 'pitch',
            'text': script.get('pitch', 'This product will change your life!'),
            'progress': (progress - 0.2) / 0.3,
            'color': (255, 255, 255)  # White
        }
    elif progress < 0.8:
        return {
            'type': 'features',
            'text': script.get('features', 'Premium quality, amazing features!'),
            'progress': (progress - 0.5) / 0.3,
            'color': (100, 255, 100)  # Green
        }
    else:
        return {
            'type': 'cta',
            'text': script.get('cta', 'Get Yours Now! Limited Time! 🔥'),
            'progress': (progress - 0.8) / 0.2,
            'color': (255, 100, 100)  # Red
        }
 
 
def create_gradient_background(width, height, scene_type):
    """Create animated gradient background"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
 
    # Scene-specific gradients
    gradients = {
        'hook': [(20, 20, 40), (40, 20, 60)],      # Dark blue
        'pitch': [(40, 40, 40), (20, 20, 20)],     # Dark gray
        'features': [(20, 40, 20), (40, 60, 40)],  # Dark green
        'cta': [(60, 20, 20), (40, 20, 40)]        # Dark red
    }
 
    colors = gradients.get(scene_type, gradients['pitch'])
 
    # Create vertical gradient
    for y in range(height):
        ratio = y / height
        r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
        g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
        b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
 
    return img
 
 
def add_animated_product_image(frame, images, scene_info, frame_num, width, height):
    """Add animated product image"""
    if not images:
        return
 
    img = images[0]  # Use first image
    scene_progress = scene_info['progress']
 
    # Image size animation
    base_size = 600
    if scene_info['type'] == 'hook':
        # Zoom in effect
        scale = 0.8 + (scene_progress * 0.4)  # 0.8 to 1.2
    elif scene_info['type'] == 'cta':
        # Pulsing effect
        pulse = math.sin(frame_num * 0.3) * 0.1
        scale = 1.0 + pulse
    else:
        scale = 1.0
 
    # Resize image
    new_size = int(base_size * scale)
    img_resized = img.resize((new_size, new_size), Image.Resampling.LANCZOS)
 
    # Position (upper part of frame)
    img_x = (width - new_size) // 2
    img_y = 150 + int(math.sin(frame_num * 0.1) * 10)  # Subtle floating effect
 
    # Add shadow effect
    shadow = Image.new('RGBA', (new_size + 20, new_size + 20), (0, 0, 0, 80))
    frame.paste(shadow, (img_x + 10, img_y + 10), shadow)
 
    # Paste main image
    if img_resized.mode != 'RGBA':
        img_resized = img_resized.convert('RGBA')
    frame.paste(img_resized, (img_x, img_y), img_resized)
 
 
def add_animated_text(frame, scene_info, frame_num, width, height):
    """Add animated text with proper readability"""
    draw = ImageDraw.Draw(frame)
    text = scene_info['text']
 
    if not text:
        return
 
    # Font sizing based on scene
    font_sizes = {
        'hook': 64,
        'pitch': 48,
        'features': 44,
        'cta': 56
    }
 
    try:
        font_size = font_sizes.get(scene_info['type'], 48)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback for systems without arial.ttf
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
 
    # Text wrapping
    max_chars = 30 if scene_info['type'] == 'hook' else 40
    wrapped_text = textwrap.fill(text, width=max_chars)
 
    # Text position (lower third)
    text_y_base = height - 500
 
    # Animation effects
    scene_progress = scene_info['progress']
 
    if scene_info['type'] == 'hook':
        # Slide up effect
        offset_y = int((1 - scene_progress) * 100)
        alpha = int(scene_progress * 255)
    elif scene_info['type'] == 'cta':
        # Pulsing effect
        pulse = math.sin(frame_num * 0.4) * 0.2 + 1.0
        alpha = 255
        offset_y = 0
    else:
        # Fade in effect
        alpha = min(255, int(scene_progress * 300))
        offset_y = 0
 
    # Draw each line with effects
    lines = wrapped_text.split('\n')
    current_y = text_y_base + offset_y
 
    for i, line in enumerate(lines):
        if not line.strip():
            continue
 
        # Get text dimensions
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
 
        # Center horizontally
        text_x = (width - text_width) // 2
 
        # Add multiple shadow layers for better readability
        shadow_offsets = [(4, 4), (2, 2), (6, 6)]
        shadow_colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0, 100)]
        shadow_alphas = [200, 150, 100]
 
        for j, (sx, sy) in enumerate(shadow_offsets):
            shadow_alpha = min(255, int(shadow_alphas[j] * (alpha / 255)))
            shadow_color = (*shadow_colors[j][:3], shadow_alpha)
 
            # Create shadow layer
            shadow_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_img)
            shadow_draw.text((text_x + sx, current_y + sy), line, 
                           fill=(0, 0, 0, shadow_alpha), font=font)
            frame.paste(shadow_img, (0, 0), shadow_img)
 
        # Main text with scene color
        text_color = (*scene_info['color'], alpha)
        text_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((text_x, current_y), line, fill=text_color, font=font)
        frame.paste(text_img, (0, 0), text_img)
 
        current_y += text_height + 15
 
 
def add_advanced_effects(frame, scene_info, frame_num, width, height):
    """Add advanced visual effects"""
    draw = ImageDraw.Draw(frame)
 
    if scene_info['type'] == 'hook':
        # Add attention-grabbing borders
        border_width = int(10 + math.sin(frame_num * 0.5) * 5)
        color = (255, 215, 0, 150)  # Gold with transparency
 
        # Top and bottom borders
        for i in range(border_width):
            alpha = int(150 * (1 - i / border_width))
            border_color = (*color[:3], alpha)
 
            border_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            border_draw = ImageDraw.Draw(border_img)
 
            # Top border
            border_draw.rectangle([i, i, width-i, i+1], fill=border_color)
            # Bottom border  
            border_draw.rectangle([i, height-i-1, width-i, height-i], fill=border_color)
 
            frame.paste(border_img, (0, 0), border_img)
 
    elif scene_info['type'] == 'cta':
        # Add pulsing corner effects
        pulse = (math.sin(frame_num * 0.6) + 1) / 2  # 0 to 1
        corner_size = int(50 + pulse * 30)
 
        effect_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        effect_draw = ImageDraw.Draw(effect_img)
 
        # Corner highlights
        alpha = int(100 + pulse * 100)
        corner_color = (255, 100, 100, alpha)
 
        # Top corners
        effect_draw.polygon([(0, 0), (corner_size, 0), (0, corner_size)], fill=corner_color)
        effect_draw.polygon([(width, 0), (width-corner_size, 0), (width, corner_size)], fill=corner_color)
 
        # Bottom corners
        effect_draw.polygon([(0, height), (corner_size, height), (0, height-corner_size)], fill=corner_color)
        effect_draw.polygon([(width, height), (width-corner_size, height), (width, height-corner_size)], fill=corner_color)
 
        frame.paste(effect_img, (0, 0), effect_img)
 
 
# Add static file serving for generated videos
@app.route('/static/videos/<filename>')
def serve_video(filename):
    """Serve generated video files"""
    try:
        return send_from_directory('static/videos', filename)
    except Exception as e:
        logger.error(f"Error serving video {filename}: {str(e)}")
        return jsonify({'error': 'Video not found'}), 404
 
 
# Update batch generation to use new FFmpeg approach
def generate_video_batch_ffmpeg(batch_id):
    """Generate videos for all jobs in a batch using FFmpeg"""
    try:
        batch_path = f"data/jobs/batch_{batch_id}.json"
        if not os.path.exists(batch_path):
            return None
 
        with open(batch_path, 'r') as f:
            batch_data = json.load(f)
 
        successful_jobs = [r for r in batch_data['results'] if r['status'] == 'success']
        video_results = []
 
        logger.info(f"Starting FFmpeg video generation for {len(successful_jobs)} jobs in batch {batch_id}")
 
        for job in successful_jobs:
            job_id = job['job_id']
            try:
                # Load job data
                job_path = f"data/jobs/{job_id}.json"
                with open(job_path, 'r') as f:
                    job_data = json.load(f)
 
                # Generate video using FFmpeg
                video_path = generate_ffmpeg_video(job_data)
 
                if video_path:
                    video_results.append({
                        "job_id": job_id,
                        "style": job['style'],
                        "duration": job['duration'],
                        "status": "success",
                        "video_path": video_path
                    })
                    logger.info(f"Successfully generated video for {job_id}")
                else:
                    video_results.append({
                        "job_id": job_id,
                        "style": job['style'],
                        "duration": job['duration'],
                        "status": "failed",
                        "error": "FFmpeg generation failed"
                    })
 
            except Exception as e:
                video_results.append({
                    "job_id": job_id,
                    "style": job['style'],
                    "duration": job['duration'],
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"Video generation error for {job_id}: {str(e)}")
 
        return video_results
 
    except Exception as e:
        logger.error(f"Error in batch video generation: {str(e)}")
        return None
@app.route('/api/job-preview/<job_id>', methods=['GET'])
def preview_job_json(job_id):
    """Return the saved job_id.json for preview/debugging"""
    try:
        path = f"data/jobs/{job_id}.json"
        if not os.path.exists(path):
            return jsonify({'error': 'Job not found'}), 404

        with open(path, 'r') as f:
            data = json.load(f)

        return jsonify({
            "success": True,
            "job_id": job_id,
            "data": data
        }), 200

    except Exception as e:
        logger.error(f"Error previewing job JSON: {str(e)}")
        return jsonify({'error': 'Failed to load job preview'}), 500


@app.route('/api/performance-insights', methods=['POST'])
def get_performance_insights():
    """Get AI-powered insights about ad performance potential"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        job_id = data.get('job_id')
        if not job_id:
            return jsonify({'error': 'job_id required'}), 400

        input_path = f"remotion/input/{job_id}.json"
        if not os.path.exists(input_path):
            return jsonify({'error': 'Job not found'}), 404

        with open(input_path, 'r') as f:
            job_data = json.load(f)

        # Analyze script for performance insights
        script = job_data.get('script', {})
        product = {
            'title': job_data.get('title'),
            'price': job_data.get('price'),
            'features': job_data.get('features', [])
        }
        
        insights = generate_performance_insights(script, product, job_data.get('style', 'energetic'))
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "insights": insights
        }), 200

    except Exception as e:
        logger.error(f"Error generating performance insights: {str(e)}")
        return jsonify({'error': 'Failed to generate insights'}), 500


# Helper functions for the new routes

def get_recommended_styles(analysis):
    """Get recommended ad styles based on product analysis"""
    tone = analysis.get('recommended_tone', 'professional')
    target_audience = analysis.get('target_audience', '').lower()
    
    recommendations = []
    
    # Primary recommendation based on tone
    recommendations.append({
        'style': tone,
        'confidence': 0.9,
        'reason': f'Best match for {tone} tone and target audience'
    })
    
    # Secondary recommendations
    if 'young' in target_audience or 'teen' in target_audience:
        recommendations.append({
            'style': 'energetic',
            'confidence': 0.8,
            'reason': 'Appeals to younger demographics'
        })
    
    if 'professional' in target_audience or 'business' in target_audience:
        recommendations.append({
            'style': 'professional',
            'confidence': 0.85,
            'reason': 'Matches professional audience expectations'
        })
    
    return recommendations[:3]  # Top 3 recommendations


def estimate_ad_performance(product, analysis):
    """Estimate potential ad performance based on product and analysis"""
    score = 70  # Base score
    
    # Adjust based on price point
    price_str = product.get('price', '')
    if price_str:
        import re
        numbers = re.findall(r'\d+\.?\d*', price_str.replace(',', ''))
        if numbers:
            price_value = float(numbers[0])
            if price_value < 50:
                score += 10  # Budget-friendly products often perform well
            elif price_value > 500:
                score -= 5   # Higher-priced items need more convincing
    
    # Adjust based on emotional triggers
    triggers = analysis.get('emotional_triggers', [])
    if len(triggers) >= 3:
        score += 10
    
    # Adjust based on competitive advantage
    if analysis.get('competitive_advantage') and 'unique' in analysis.get('competitive_advantage', '').lower():
        score += 15
    
    # Adjust based on urgency factors
    urgency_factors = analysis.get('urgency_factors', [])
    if len(urgency_factors) >= 2:
        score += 10
    
    # Cap the score
    score = min(95, max(30, score))
    
    return {
        'estimated_score': score,
        'performance_tier': get_performance_tier(score),
        'key_strengths': get_key_strengths(analysis),
        'improvement_suggestions': get_improvement_suggestions(analysis, score)
    }


def get_performance_tier(score):
    """Get performance tier based on score"""
    if score >= 85:
        return 'Excellent'
    elif score >= 75:
        return 'Good'
    elif score >= 60:
        return 'Average'
    else:
        return 'Needs Improvement'


def get_key_strengths(analysis):
    """Extract key strengths from analysis"""
    strengths = []
    
    if analysis.get('competitive_advantage'):
        strengths.append('Strong competitive advantage')
    
    emotional_triggers = analysis.get('emotional_triggers', [])
    if len(emotional_triggers) >= 3:
        strengths.append('Multiple emotional triggers')
    
    urgency_factors = analysis.get('urgency_factors', [])
    if urgency_factors:
        strengths.append('Built-in urgency factors')
    
    trust_signals = analysis.get('trust_signals', [])
    if trust_signals:
        strengths.append('Trust-building elements')
    
    return strengths


def get_improvement_suggestions(analysis, score):
    """Get improvement suggestions based on analysis and score"""
    suggestions = []
    
    if score < 70:
        suggestions.append('Consider highlighting more unique benefits')
        
    emotional_triggers = analysis.get('emotional_triggers', [])
    if len(emotional_triggers) < 2:
        suggestions.append('Add more emotional appeal to connect with audience')
    
    urgency_factors = analysis.get('urgency_factors', [])
    if not urgency_factors:
        suggestions.append('Include urgency or scarcity elements')
    
    trust_signals = analysis.get('trust_signals', [])
    if not trust_signals:
        suggestions.append('Add trust signals like reviews or guarantees')
    
    return suggestions


def generate_optimized_script(product, original_script, feedback, goals, style, duration):
    """Generate an optimized version of the script based on feedback"""
    try:
        optimization_prompt = f"""
        Optimize this video ad script based on the feedback and goals provided.
        
        ORIGINAL SCRIPT:
        Hook: {original_script.get('hook', '')}
        Pitch: {original_script.get('pitch', '')}
        Features: {original_script.get('features', '')}
        CTA: {original_script.get('cta', '')}
        
        PRODUCT: {product.get('title')}
        PRICE: {product.get('price')}
        STYLE: {style}
        DURATION: {duration} seconds
        
        FEEDBACK: {feedback}
        OPTIMIZATION GOALS: {', '.join(goals)}
        
        Create an improved version that addresses the feedback and optimizes for the specified goals.
        
        Respond in this exact JSON format:
        {{
            "hook": "Improved hook",
            "pitch": "Enhanced pitch", 
            "features": "Optimized features section",
            "cta": "Stronger call-to-action",
            "optimization_notes": "What was changed and why",
            "expected_improvements": ["improvement1", "improvement2"]
        }}
        """
        
        from ai_service import call_openai_with_retry
        optimized = call_openai_with_retry(optimization_prompt, "script_optimization")
        
        return optimized
        
    except Exception as e:
        logger.error(f"Error optimizing script: {str(e)}")
        # Return original script with optimization attempt note
        return {
            **original_script,
            "optimization_notes": f"Optimization failed: {str(e)}",
            "expected_improvements": []
        }


def generate_performance_insights(script, product, style):
    """Generate AI-powered performance insights for a script"""
    try:
        insights_prompt = f"""
        Analyze the following product video ad script and provide performance insights.

        SCRIPT:
        Hook: {script.get('hook', '')}
        Pitch: {script.get('pitch', '')}
        Features: {script.get('features', '')}
        CTA: {script.get('cta', '')}

        PRODUCT:
        Title: {product.get('title')}
        Price: {product.get('price')}
        Features: {', '.join(product.get('features', []))}

        STYLE: {style}

        Provide a detailed analysis in this exact JSON format:
        {{
            "overall_score": 85,
            "hook_effectiveness": {{
                "score": 80,
                "analysis": "Analysis of hook strength",
                "suggestions": ["suggestion1", "suggestion2"]
            }},
            "pitch_clarity": {{
                "score": 90,
                "analysis": "Analysis of pitch clarity and value delivery",
                "suggestions": ["suggestion1"]
            }},
            "feature_presentation": {{
                "score": 75,
                "analysis": "How well features are presented",
                "suggestions": ["suggestion1", "suggestion2"]
            }},
            "cta_strength": {{
                "score": 85,
                "analysis": "Call-to-action impact and clarity",
                "suggestions": ["suggestion1"]
            }},
            "target_audience_fit": {{
                "score": 80,
                "analysis": "Audience alignment and tone match",
                "demographic": "primary target demographic"
            }},
            "estimated_engagement": "high|medium|low",
            "estimated_conversion": "high|medium|low",
            "key_strengths": ["strength1", "strength2"],
            "areas_for_improvement": ["area1", "area2"],
            "a_b_test_suggestions": ["test1", "test2"]
        }}
        """

        from ai_service import call_openai_with_retry
        insights = call_openai_with_retry(insights_prompt, "performance_insights")
        return insights

    except Exception as e:
        logger.error(f"Error generating performance insights: {str(e)}")
        return {
            "overall_score": 70,
            "hook_effectiveness": {
                "score": 65,
                "analysis": "Default fallback analysis",
                "suggestions": []
            },
            "pitch_clarity": {
                "score": 70,
                "analysis": "Default fallback analysis",
                "suggestions": []
            },
            "feature_presentation": {
                "score": 65,
                "analysis": "Default fallback analysis",
                "suggestions": []
            },
            "cta_strength": {
                "score": 70,
                "analysis": "Default fallback analysis",
                "suggestions": []
            },
            "target_audience_fit": {
                "score": 70,
                "analysis": "General audience fit",
                "demographic": "general consumers"
            },
            "estimated_engagement": "medium",
            "estimated_conversion": "medium",
            "key_strengths": ["basic clarity"],
            "areas_for_improvement": ["add emotional appeal"],
            "a_b_test_suggestions": ["try multiple hook styles"]
        }

@app.route('/api/analyze-product', methods=['POST'])
def analyze_product():
    """Analyze product for marketing insights without generating full script"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        product = data.get('product')
        if not product or not product.get('title'):
            return jsonify({'error': 'Invalid product data - title required'}), 400

        # Analyze product sentiment and marketing approach
        analysis = analyze_product_sentiment(product)
        
        # Generate multiple hook variations for testing
        hooks = generate_hook_variations(product, count=3)
        
        result = {
            "success": True,
            "product_title": product.get('title'),
            "analysis": analysis,
            "sample_hooks": hooks,
            "recommended_styles": get_recommended_styles(analysis),
            "estimated_performance": estimate_ad_performance(product, analysis)
        }
        
        logger.info(f"Successfully analyzed product: {product.get('title', 'Unknown')[:50]}")
        return jsonify(result), 200

    except AIServiceError as e:
        logger.error(f"AI Service error in analyze_product: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Error in analyze_product: {str(e)}")
        return jsonify({'error': 'Failed to analyze product'}), 500


@app.route('/api/generate-hooks', methods=['POST'])
def generate_hooks():
    """Generate multiple hook variations for A/B testing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        product = data.get('product')
        count = data.get('count', 5)
        style = data.get('style', 'energetic')

        if not product or not product.get('title'):
            return jsonify({'error': 'Invalid product data'}), 400

        if count > 10:
            count = 10  # Limit to prevent overuse

        hooks = generate_hook_variations(product, count=count)
        
        return jsonify({
            "success": True,
            "product_title": product.get('title'),
            "hooks": hooks,
            "count": len(hooks),
            "style": style
        }), 200

    except AIServiceError as e:
        logger.error(f"AI Service error in generate_hooks: {str(e)}")
        return jsonify({'error': f'Hook generation failed: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Error in generate_hooks: {str(e)}")
        return jsonify({'error': 'Failed to generate hooks'}), 500


@app.route('/api/video-metadata/<job_id>', methods=['GET'])
def get_video_metadata(job_id):
    """Get video metadata for a specific job"""
    try:
        input_path = f"remotion/input/{job_id}.json"
        
        if not os.path.exists(input_path):
            return jsonify({'error': 'Job not found'}), 404

        with open(input_path, 'r') as f:
            data = json.load(f)

        metadata = data.get('video_metadata', {})
        if not metadata:
            # Generate metadata if not exists
            product = {
                'title': data.get('title'),
                'price': data.get('price'),
                'features': data.get('features', [])
            }
            script = data.get('script', {})
            metadata = generate_video_metadata(product, script)
            
            # Save updated data
            data['video_metadata'] = metadata
            with open(input_path, 'w') as f:
                json.dump(data, f, indent=2)

        return jsonify({
            "success": True,
            "job_id": job_id,
            "metadata": metadata
        }), 200

    except Exception as e:
        logger.error(f"Error getting video metadata: {str(e)}")
        return jsonify({'error': 'Failed to get video metadata'}), 500


@app.route('/api/optimize-script', methods=['POST'])
def optimize_script():
    """Optimize existing script based on feedback or performance data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        job_id = data.get('job_id')
        feedback = data.get('feedback', '')  # User feedback
        optimization_goals = data.get('goals', ['engagement'])  # engagement, conversion, brand_awareness
        
        if not job_id:
            return jsonify({'error': 'job_id required'}), 400

        input_path = f"remotion/input/{job_id}.json"
        if not os.path.exists(input_path):
            return jsonify({'error': 'Job not found'}), 404

        # Load existing data
        with open(input_path, 'r') as f:
            existing_data = json.load(f)

        product = {
            'title': existing_data.get('title'),
            'price': existing_data.get('price'),
            'features': existing_data.get('features', [])
        }
        
        original_script = existing_data.get('script', {})
        style = existing_data.get('style', 'energetic')
        duration = existing_data.get('duration', 30)

        # Generate optimized script
        optimized_script = generate_optimized_script(
            product, original_script, feedback, optimization_goals, style, duration
        )

        # Save optimized version
        new_job_id = f"{job_id}_optimized_{int(time.time())}"
        optimized_data = {
            **existing_data,
            'job_id': new_job_id,
            'script': optimized_script,
            'optimization_info': {
                'original_job_id': job_id,
                'feedback': feedback,
                'goals': optimization_goals,
                'optimized_at': time.time()
            }
        }

        with open(f"remotion/input/{new_job_id}.json", "w") as f:
            json.dump(optimized_data, f, indent=2)

        return jsonify({
            "success": True,
            "original_job_id": job_id,
            "optimized_job_id": new_job_id,
            "optimized_script": optimized_script,
            "optimization_goals": optimization_goals
        }), 200

    except AIServiceError as e:
        logger.error(f"AI Service error in optimize_script: {str(e)}")
        return jsonify({'error': f'Script optimization failed: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in optimize_script: {str(e)}")
        return jsonify({'error': 'Failed to optimize script'}), 500



@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'cache_size': len(scrape_cache)
    }), 200

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear scraping cache"""
    global scrape_cache
    cache_size = len(scrape_cache)
    scrape_cache.clear()
    return jsonify({
        'message': f'Cache cleared. Removed {cache_size} entries.'
    }), 200

@app.route('/api/supported-platforms', methods=['GET'])
def supported_platforms():
    """Return list of supported e-commerce platforms"""
    return jsonify({
        'platforms': [
            {
                'name': 'Amazon',
                'domains': ['amazon.com', 'amazon.co.uk', 'amazon.ca', 'amazon.de'],
                'example': 'https://www.amazon.com/dp/PRODUCT_ID'
            },
            {
                'name': 'Shopify',
                'domains': ['*.myshopify.com', 'shopify stores'],
                'example': 'https://store.myshopify.com/products/product-name'
            },
            {
                'name': 'Generic E-commerce',
                'domains': ['Any site with /products/ or /product/ in URL'],
                'example': 'https://example.com/products/item'
            }
        ]
    }), 200


# ============= NEW FEATURE ENDPOINTS =============

@app.route('/api/process-image', methods=['POST'])
def process_image():
    """Process product image with various operations."""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        operations = data.get('operations', {})

        if not image_url:
            return jsonify({'error': 'image_url required'}), 400

        # Process image
        processed = process_product_image(image_url, operations)

        if processed is None:
            return jsonify({'error': 'Failed to process image'}), 500

        # Save processed image temporarily
        job_id = f"img_{uuid.uuid4().hex[:8]}"
        output_path = f"static/processed/{job_id}.png"
        os.makedirs("static/processed", exist_ok=True)
        processed.save(output_path, "PNG")

        return jsonify({
            'success': True,
            'processed_image_url': f"/static/processed/{job_id}.png",
            'operations_applied': list(operations.keys())
        }), 200

    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/video-platforms', methods=['GET'])
def get_video_platforms():
    """Get list of supported social media platforms for video export."""
    return jsonify({
        'success': True,
        'platforms': get_all_platforms()
    }), 200


@app.route('/api/generate-variants', methods=['POST'])
def generate_variants():
    """Generate video variants for multiple platforms."""
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        platforms = data.get('platforms', [])

        if not job_id:
            return jsonify({'error': 'job_id required'}), 400

        if not platforms:
            return jsonify({'error': 'platforms list required'}), 400

        # Get base video path
        base_video = f"static/videos/{job_id}.mp4"
        if not os.path.exists(base_video):
            return jsonify({'error': 'Base video not found'}), 404

        # Generate variants
        batch = generate_platform_variants(
            base_video,
            job_id,
            platforms,
            "static/videos"
        )

        return jsonify({
            'success': True,
            'batch_id': batch.batch_id,
            'status': batch.status,
            'results': batch.results,
            'errors': batch.errors
        }), 200

    except Exception as e:
        logger.error(f"Variant generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/audio-tracks', methods=['GET'])
def get_audio_tracks():
    """Get available audio tracks."""
    try:
        engine = AudioEngine()
        style = request.args.get('style')

        tracks = engine.get_available_tracks(style)
        sfx = engine.get_available_sfx()

        return jsonify({
            'success': True,
            'music_tracks': tracks,
            'sound_effects': sfx
        }), 200

    except Exception as e:
        logger.error(f"Audio tracks error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-hooks-ab', methods=['POST'])
def generate_hooks_ab():
    """Generate A/B testable hooks."""
    try:
        data = request.get_json()
        product = data.get('product', {})
        strategies = data.get('strategies')
        count = data.get('count', 5)
        platform = data.get('platform')
        industry = data.get('industry', 'ecommerce')

        if not product.get('title'):
            return jsonify({'error': 'product with title required'}), 400

        # Try AI-generated hooks first
        hooks = generate_ab_hooks_enhanced(product, strategies, count, platform, industry)

        # Supplement with template-based hooks if needed
        if len(hooks) < count:
            manager = HooksManager()
            template_hooks = manager.generate_ab_hooks(product, strategies, count - len(hooks), platform)
            hooks.extend(template_hooks)

        return jsonify({
            'success': True,
            'hooks': hooks[:count]
        }), 200

    except Exception as e:
        logger.error(f"Hooks generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/industries', methods=['GET'])
def get_industries():
    """Get available industry types."""
    return jsonify({
        'success': True,
        'industries': [
            {'id': key, 'name': key.replace('_', ' ').title(), 'config': value}
            for key, value in INDUSTRY_PROMPTS.items()
        ]
    }), 200


@app.route('/api/copy-platforms', methods=['GET'])
def get_copy_platforms():
    """Get platform-specific copy guidelines."""
    return jsonify({
        'success': True,
        'platforms': [
            {'id': key, 'name': key.title(), 'guidelines': value}
            for key, value in PLATFORM_COPY_GUIDELINES.items()
        ]
    }), 200


@app.route('/api/generate-content-v2', methods=['POST'])
def generate_content_v2():
    """Enhanced content generation with industry and platform optimization."""
    try:
        data = request.get_json()
        job_id = data.get('job_id', f"job_{int(time.time() * 1000)}")
        product = data.get('product', {})
        style = data.get('style', 'energetic')
        duration = data.get('duration', 30)
        industry = data.get('industry', 'ecommerce')
        platform = data.get('platform')
        emotional_triggers = data.get('emotional_triggers', [])
        brand_kit_data = data.get('brand_kit')
        audio_settings = data.get('audio_settings')

        # New dynamic content parameters
        dynamic_content_data = data.get('dynamic_content', {})
        industry_template = data.get('industry_template')

        # Advanced video effects
        enable_lens_flare = data.get('enable_lens_flare', False)
        enable_glitch_effects = data.get('enable_glitch_effects', False)
        glitch_intensity = data.get('glitch_intensity', 0.5)

        if not product.get('title'):
            return jsonify({'error': 'Product title required'}), 400

        # Generate enhanced script
        script = generate_ad_script_v2(
            product, style, duration, industry, platform, emotional_triggers
        )

        # Build dynamic content config if provided
        dynamic_content_config = None
        if dynamic_content_data:
            pricing = dynamic_content_data.get('pricing', {})
            countdown = dynamic_content_data.get('countdown', {})
            rating = dynamic_content_data.get('rating', {})
            review = dynamic_content_data.get('review', {})
            cta = dynamic_content_data.get('cta', {})

            dynamic_content_config = {
                'show_price': pricing.get('enabled', False),
                'original_price': pricing.get('original'),
                'sale_price': pricing.get('sale'),
                'price_animation': pricing.get('animation', 'drop'),
                'show_countdown': countdown.get('enabled', False),
                'countdown_seconds': countdown.get('hours', 24) * 3600 if countdown.get('hours') else countdown.get('seconds', 86400),
                'countdown_style': countdown.get('style', 'flip'),
                'show_rating': rating.get('enabled', False),
                'rating': rating.get('value', 4.8),
                'review_count': rating.get('count'),
                'show_review': review.get('enabled', False),
                'review_quote': review.get('quote'),
                'review_author': review.get('author'),
                'cta_text': cta.get('text', 'Shop Now'),
                'cta_style': cta.get('style', 'pulse'),
                'cta_color': tuple(cta.get('color', [255, 87, 51]))
            }

        # Store job data
        job_data = {
            'job_id': job_id,
            'title': product.get('title'),
            'price': product.get('price'),
            'description': product.get('description', ''),
            'features': product.get('features', []),
            'images': product.get('images', []),
            'script': script,
            'style': style,
            'duration': duration,
            'industry': industry,
            'platform': platform,
            'emotional_triggers': emotional_triggers,
            'brand_kit': brand_kit_data,
            'audio_settings': audio_settings,
            'dynamic_content': dynamic_content_config,
            'industry_template': industry_template,
            'enable_lens_flare': enable_lens_flare,
            'enable_glitch_effects': enable_glitch_effects,
            'glitch_intensity': glitch_intensity,
            'generated_at': time.time()
        }

        # Save job data
        os.makedirs('data/jobs', exist_ok=True)
        with open(f'data/jobs/{job_id}.json', 'w') as f:
            json.dump(job_data, f, indent=2)

        return jsonify({
            'success': True,
            'job_id': job_id,
            'script': script,
            'style': style,
            'duration': duration,
            'industry': industry,
            'platform': platform,
            'has_dynamic_content': dynamic_content_config is not None,
            'industry_template': industry_template
        }), 200

    except Exception as e:
        logger.error(f"Content generation v2 error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/industry-templates', methods=['GET'])
def get_industry_templates():
    """Get available industry templates."""
    try:
        industry = request.args.get('industry')

        if industry:
            # Get templates for specific industry
            if industry in INDUSTRY_TEMPLATES:
                templates = []
                for template_id, template_config in INDUSTRY_TEMPLATES[industry].items():
                    templates.append({
                        'id': template_id,
                        'name': template_config.name,
                        'description': template_config.description,
                        'scene_count': len(template_config.scenes),
                        'color_scheme': template_config.color_scheme,
                        'recommended_duration': template_config.recommended_duration
                    })
                return jsonify({
                    'success': True,
                    'industry': industry,
                    'templates': templates
                }), 200
            else:
                return jsonify({'error': f'Industry "{industry}" not found'}), 404
        else:
            # Get all industries with their templates
            all_templates = {}
            for ind, templates in INDUSTRY_TEMPLATES.items():
                all_templates[ind] = []
                for template_id, template_config in templates.items():
                    all_templates[ind].append({
                        'id': template_id,
                        'name': template_config.name,
                        'description': template_config.description
                    })

            return jsonify({
                'success': True,
                'industries': list(INDUSTRY_TEMPLATES.keys()),
                'templates': all_templates
            }), 200

    except Exception as e:
        logger.error(f"Industry templates error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/color-schemes', methods=['GET'])
def get_color_schemes():
    """Get available color schemes for video generation."""
    try:
        schemes = []
        for scheme_id, scheme_config in COLOR_SCHEMES.items():
            schemes.append({
                'id': scheme_id,
                'name': scheme_id.replace('_', ' ').title(),
                'primary': list(scheme_config.get('primary', (255, 255, 255))),
                'secondary': list(scheme_config.get('secondary', (200, 200, 200))),
                'accent': list(scheme_config.get('accent', (255, 87, 51))),
                'background': list(scheme_config.get('background', (0, 0, 0)))
            })

        return jsonify({
            'success': True,
            'color_schemes': schemes
        }), 200

    except Exception as e:
        logger.error(f"Color schemes error: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def log_request_info():
    """Log request information"""
    if request.endpoint:
        logger.info(f"Request: {request.method} {request.path}")

if __name__ == '__main__':
    print("🚀 Starting AI Video Generator API Server...")
    print("📊 Available endpoints:")
    print("   POST /api/analyze-url - Scrape product data")
    print("   GET  /api/health - Health check")
    print("   POST /api/clear-cache - Clear cache")
    print("   GET  /api/supported-platforms - List supported platforms")
    print("\n🔍 Testing with curl:")
    print('   curl -X POST http://localhost:5000/api/analyze-url \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"url": "https://www.amazon.com/dp/B08N5WRWNW"}\'')
    print("\n" + "="*50)
    
    app.run(debug=True, port=5000, host='0.0.0.0')