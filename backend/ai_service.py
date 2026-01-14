import openai
import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client (modern approach)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1
DEFAULT_MODEL = "gpt-4o-mini"  # More cost-effective for most use cases
PREMIUM_MODEL = "gpt-4o"      # For complex requests

# Industry-specific prompt configurations
INDUSTRY_PROMPTS = {
    "ecommerce": {
        "tone": "exciting, benefit-focused, urgency-driven",
        "cta_style": "Shop Now, Add to Cart, Get Yours Today",
        "emotional_focus": "desire, FOMO, instant gratification",
        "hooks": ["problem-solution", "curiosity", "social-proof"],
        "power_words": ["exclusive", "limited", "bestselling", "trending", "must-have"]
    },
    "saas": {
        "tone": "professional, solution-oriented, ROI-focused",
        "cta_style": "Start Free Trial, Book Demo, Get Started Free",
        "emotional_focus": "efficiency, growth, competitive advantage",
        "hooks": ["pain-point", "statistics", "transformation"],
        "power_words": ["automate", "streamline", "scale", "boost", "optimize"]
    },
    "services": {
        "tone": "trustworthy, expert, results-driven",
        "cta_style": "Book Now, Get Quote, Schedule Consultation",
        "emotional_focus": "trust, expertise, peace of mind",
        "hooks": ["testimonial", "expertise", "guarantee"],
        "power_words": ["certified", "guaranteed", "proven", "expert", "trusted"]
    },
    "local_business": {
        "tone": "friendly, community-focused, personal",
        "cta_style": "Visit Us Today, Call Now, Book Appointment",
        "emotional_focus": "community, convenience, personal touch",
        "hooks": ["local-pride", "personal-story", "special-offer"],
        "power_words": ["local", "family-owned", "neighborhood", "community", "nearby"]
    },
    "health": {
        "tone": "caring, science-backed, transformative",
        "cta_style": "Start Your Journey, Try Risk-Free, Get Healthier",
        "emotional_focus": "transformation, wellness, self-care",
        "hooks": ["transformation", "natural", "results"],
        "power_words": ["natural", "clinically-proven", "doctor-recommended", "pure", "vital"]
    },
    "finance": {
        "tone": "confident, trustworthy, empowering",
        "cta_style": "Get Started, Calculate Savings, Secure Your Future",
        "emotional_focus": "security, growth, financial freedom",
        "hooks": ["statistics", "transformation", "fear-prevention"],
        "power_words": ["secure", "grow", "save", "protect", "invest"]
    }
}

# Platform-specific copy guidelines
PLATFORM_COPY_GUIDELINES = {
    "tiktok": {
        "max_hook_words": 8,
        "max_total_words": 50,
        "tone": "casual, trendy, authentic, Gen-Z friendly",
        "emoji_usage": "high",
        "cta_style": "short action verbs (Link in bio, Try it, Get yours)",
        "avoid": ["formal language", "long sentences", "corporate speak"]
    },
    "instagram": {
        "max_hook_words": 12,
        "max_total_words": 75,
        "tone": "aspirational, visual-focused, lifestyle",
        "emoji_usage": "medium",
        "cta_style": "lifestyle-oriented (Transform your routine, Elevate your style)",
        "avoid": ["overly salesy", "clickbait"]
    },
    "youtube": {
        "max_hook_words": 15,
        "max_total_words": 150,
        "tone": "educational, engaging, detailed",
        "emoji_usage": "low",
        "cta_style": "subscribe, like, learn more",
        "avoid": ["rushing", "skipping details"]
    },
    "linkedin": {
        "max_hook_words": 15,
        "max_total_words": 100,
        "tone": "professional, data-driven, insightful",
        "emoji_usage": "minimal",
        "cta_style": "business-value focused (Boost ROI, Scale operations)",
        "avoid": ["casual slang", "excessive emojis", "entertainment focus"]
    },
    "facebook": {
        "max_hook_words": 12,
        "max_total_words": 80,
        "tone": "friendly, community-focused, relatable",
        "emoji_usage": "medium",
        "cta_style": "clear action (Shop Now, Learn More, Sign Up)",
        "avoid": ["youth slang", "overly trendy"]
    },
    "twitter": {
        "max_hook_words": 10,
        "max_total_words": 60,
        "tone": "witty, concise, conversational",
        "emoji_usage": "medium",
        "cta_style": "click, engage, share",
        "avoid": ["long sentences", "multiple CTAs"]
    }
}

class AIServiceError(Exception):
    """Custom exception for AI service errors"""
    pass

def generate_ad_script(product: Dict[str, Any], style: str = "energetic", duration: int = 30) -> Dict[str, Any]:
    """
    Generate an engaging video ad script for a product
    
    Args:
        product: Dictionary containing product information
        style: Ad style ('energetic', 'professional', 'casual', 'luxury')
        duration: Target video duration in seconds (15, 30, 60)
    
    Returns:
        Dictionary with script sections and metadata
    """
    try:
        # Validate input
        if not product or not product.get('title'):
            raise AIServiceError("Product title is required")

        # Determine timing based on duration
        timing = get_timing_breakdown(duration)
        
        # Build enhanced prompt
        prompt = build_script_prompt(product, style, timing)
        
        # Generate script with retries
        script_data = call_openai_with_retry(prompt, "script")
        
        # Enhance script with additional metadata
        enhanced_script = enhance_script_output(script_data, product, style, duration)
        
        logger.info(f"Successfully generated ad script for: {product.get('title', 'Unknown')[:50]}")
        return enhanced_script
        
    except Exception as e:
        logger.error(f"Error generating ad script: {str(e)}")
        raise AIServiceError(f"Failed to generate ad script: {str(e)}")


def generate_video_metadata(product: Dict[str, Any], script: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate metadata for video production (colors, fonts, animations, etc.)
    
    Args:
        product: Product information
        script: Generated script data
    
    Returns:
        Dictionary with video production metadata
    """
    try:
        prompt = f"""
        You are a video production expert. Based on this product and script, suggest video production elements.
        
        Product: {product.get('title')}
        Price: {product.get('price')}
        Category: {classify_product_category(product)}
        
        Script Hook: {script.get('hook', '')}
        
        Provide production suggestions in this exact JSON format:
        {{
            "color_scheme": {{
                "primary": "#hex_color",
                "secondary": "#hex_color",
                "accent": "#hex_color",
                "background": "#hex_color"
            }},
            "typography": {{
                "primary_font": "font_name",
                "secondary_font": "font_name",
                "font_weight": "normal|bold|light"
            }},
            "animations": [
                "fade_in",
                "slide_up",
                "zoom_in",
                "bounce"
            ],
            "music_style": "upbeat|calm|dramatic|electronic|corporate",
            "visual_effects": [
                "particle_effects",
                "gradient_overlays",
                "product_highlights"
            ],
            "layout_suggestions": {{
                "product_position": "center|left|right",
                "text_alignment": "center|left|right",
                "logo_placement": "top_right|bottom_right|bottom_center"
            }}
        }}
        """
        
        metadata = call_openai_with_retry(prompt, "metadata", model=DEFAULT_MODEL)
        logger.info("Successfully generated video metadata")
        return metadata
        
    except Exception as e:
        logger.error(f"Error generating video metadata: {str(e)}")
        return get_default_video_metadata()


def generate_hook_variations(product: Dict[str, Any], count: int = 5) -> List[str]:
    """
    Generate multiple hook variations for A/B testing
    
    Args:
        product: Product information
        count: Number of variations to generate
    
    Returns:
        List of hook variations
    """
    try:
        prompt = f"""
        Create {count} different engaging hooks (2-3 seconds each) for this product ad.
        Make each hook unique in approach and style.
        
        Product: {product.get('title')}
        Price: {product.get('price')}
        Key benefit: {get_main_benefit(product)}
        
        Return as JSON array of strings:
        ["Hook 1", "Hook 2", "Hook 3", ...]
        
        Hook styles to vary:
        - Question-based
        - Problem-solution
        - Benefit-focused
        - Curiosity-driven
        - Social proof
        """
        
        response = call_openai_with_retry(prompt, "hooks", model=DEFAULT_MODEL)
        
        if isinstance(response, list):
            return response[:count]
        else:
            logger.warning("Unexpected hook response format, returning default")
            return [f"Discover the {product.get('title', 'amazing product')}!"]
            
    except Exception as e:
        logger.error(f"Error generating hook variations: {str(e)}")
        return [f"Check out this {product.get('title', 'amazing product')}!"]


def analyze_product_sentiment(product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze product sentiment and suggest marketing approach
    
    Args:
        product: Product information
    
    Returns:
        Dictionary with sentiment analysis and recommendations
    """
    try:
        prompt = f"""
        Analyze this product and suggest the best marketing approach:
        
        Product: {product.get('title')}
        Description: {product.get('description', '')}
        Price: {product.get('price')}
        Features: {', '.join(product.get('features', []))}
        
        Provide analysis in this JSON format:
        {{
            "target_audience": "primary demographic",
            "emotional_triggers": ["trigger1", "trigger2", "trigger3"],
            "value_proposition": "main selling point",
            "competitive_advantage": "what makes it special",
            "urgency_factors": ["factor1", "factor2"],
            "trust_signals": ["signal1", "signal2"],
            "recommended_tone": "energetic|professional|casual|luxury|playful"
        }}
        """
        
        analysis = call_openai_with_retry(prompt, "sentiment", model=DEFAULT_MODEL)
        logger.info("Successfully analyzed product sentiment")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing product sentiment: {str(e)}")
        return get_default_sentiment_analysis()


def call_openai_with_retry(prompt: str, request_type: str, model: str = None) -> Dict[str, Any]:
    """
    Call OpenAI API with retry logic and error handling
    
    Args:
        prompt: The prompt to send
        request_type: Type of request for logging
        model: Model to use (defaults to DEFAULT_MODEL)
    
    Returns:
        Parsed response data
    """
    if model is None:
        model = DEFAULT_MODEL
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"OpenAI API call attempt {attempt + 1} for {request_type}")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert marketing copywriter and video production specialist. Always respond with valid JSON only, no additional text or formatting."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=30
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean and parse JSON response
            content = clean_json_response(content)
            parsed_content = json.loads(content)
            
            logger.info(f"Successfully completed {request_type} request")
            return parsed_content
            
        except json.JSONDecodeError as e:
            last_error = f"JSON parsing error: {str(e)}"
            logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {last_error}")
            
        except openai.RateLimitError as e:
            last_error = f"Rate limit exceeded: {str(e)}"
            logger.warning(f"Rate limit hit on attempt {attempt + 1}, waiting...")
            time.sleep(RETRY_DELAY * (attempt + 1))
            
        except openai.APIError as e:
            last_error = f"OpenAI API error: {str(e)}"
            logger.error(f"API error on attempt {attempt + 1}: {last_error}")
            
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error on attempt {attempt + 1}: {last_error}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    # All retries failed
    logger.error(f"All {MAX_RETRIES} attempts failed for {request_type}: {last_error}")
    raise AIServiceError(f"Failed to complete {request_type} after {MAX_RETRIES} attempts: {last_error}")


def build_script_prompt(product: Dict[str, Any], style: str, timing: Dict[str, int]) -> str:
    """Build enhanced prompt for script generation"""
    
    features_text = format_features(product.get('features', []))
    category = classify_product_category(product)
    price_context = analyze_price_point(product.get('price', ''))
    
    return f"""
    Create a compelling {timing['total']}-second video ad script for this {category} product.
    
    PRODUCT DETAILS:
    Title: {product.get('title')}
    Description: {product.get('description', 'N/A')}
    Price: {product.get('price')} {price_context}
    Features: {features_text}
    
    STYLE REQUIREMENTS:
    - Tone: {style}
    - Target audience: {get_target_audience(product, style)}
    - Emotional approach: {get_emotional_approach(style)}
    
    TIMING BREAKDOWN:
    - Hook: {timing['hook']} seconds (grab attention immediately)
    - Pitch: {timing['pitch']} seconds (present the solution)
    - Features: {timing['features']} seconds (highlight key benefits)
    - CTA: {timing['cta']} seconds (drive action)
    
    SCRIPT REQUIREMENTS:
    - Use power words and emotional triggers
    - Include specific benefits, not just features
    - Create urgency without being pushy
    - Make it conversational and natural
    - Include visual cues for video production
    
    Respond in this exact JSON format:
    {{
        "hook": "Attention-grabbing opening that addresses a pain point or desire",
        "pitch": "Clear value proposition with emotional appeal",
        "features": "Top 2-3 benefits with social proof elements",
        "cta": "Compelling call-to-action with urgency",
        "visual_cues": [
            "Scene 1: Product showcase",
            "Scene 2: Feature demonstration", 
            "Scene 3: Happy customer",
            "Scene 4: Strong CTA with product"
        ],
        "word_count": {timing['total'] * 3},
        "key_messages": ["message1", "message2", "message3"]
    }}
    """


# Helper functions
def get_timing_breakdown(duration: int) -> Dict[str, int]:
    """Get timing breakdown based on total duration"""
    if duration <= 15:
        return {"total": 15, "hook": 3, "pitch": 5, "features": 4, "cta": 3}
    elif duration <= 30:
        return {"total": 30, "hook": 5, "pitch": 10, "features": 10, "cta": 5}
    else:  # 60 seconds
        return {"total": 60, "hook": 8, "pitch": 20, "features": 25, "cta": 7}


def classify_product_category(product: Dict[str, Any]) -> str:
    """Classify product into broad category"""
    title = product.get('title', '').lower()
    description = product.get('description', '').lower()
    text = f"{title} {description}"
    
    categories = {
        'electronics': ['phone', 'computer', 'laptop', 'tablet', 'headphone', 'camera', 'tech'],
        'fashion': ['clothing', 'shirt', 'dress', 'shoes', 'jewelry', 'watch', 'fashion'],
        'health': ['vitamin', 'supplement', 'fitness', 'health', 'wellness', 'medical'],
        'home': ['furniture', 'kitchen', 'home', 'decor', 'appliance', 'tool'],
        'beauty': ['makeup', 'skincare', 'beauty', 'cosmetic', 'hair', 'fragrance']
    }
    
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    
    return 'general'


def analyze_price_point(price: str) -> str:
    """Analyze price point and return context"""
    if not price:
        return ""
    
    # Extract numeric value from price string
    import re
    numbers = re.findall(r'\d+\.?\d*', price.replace(',', ''))
    if not numbers:
        return ""
    
    price_value = float(numbers[0])
    
    if price_value < 20:
        return "(budget-friendly)"
    elif price_value < 100:
        return "(affordable)"
    elif price_value < 500:
        return "(mid-range)"
    else:
        return "(premium)"


def format_features(features: List[str]) -> str:
    """Format features list for prompt"""
    if not features:
        return "No specific features listed"
    
    return "\n".join([f"• {feature}" for feature in features[:5]])  # Limit to top 5


def get_target_audience(product: Dict[str, Any], style: str) -> str:
    """Determine target audience based on product and style"""
    category = classify_product_category(product)
    
    audiences = {
        'electronics': 'tech enthusiasts and professionals',
        'fashion': 'style-conscious consumers',
        'health': 'health and wellness focused individuals',
        'home': 'homeowners and apartment dwellers',
        'beauty': 'beauty and self-care enthusiasts',
        'general': 'general consumers'
    }
    
    base_audience = audiences.get(category, 'general consumers')
    
    if style == 'luxury':
        return f"affluent {base_audience}"
    elif style == 'casual':
        return f"everyday {base_audience}"
    else:
        return base_audience


def get_emotional_approach(style: str) -> str:
    """Get emotional approach based on style"""
    approaches = {
        'energetic': 'excitement and enthusiasm',
        'professional': 'trust and reliability',
        'casual': 'relatability and friendliness',
        'luxury': 'exclusivity and aspiration',
        'playful': 'fun and joy'
    }
    return approaches.get(style, 'positive and engaging')


def get_main_benefit(product: Dict[str, Any]) -> str:
    """Extract main benefit from product"""
    features = product.get('features', [])
    if features:
        return features[0]
    
    description = product.get('description', '')
    if description:
        return description[:100] + "..." if len(description) > 100 else description
    
    return f"Great value at {product.get('price', 'affordable price')}"


def enhance_script_output(script_data: Dict[str, Any], product: Dict[str, Any], style: str, duration: int) -> Dict[str, Any]:
    """Enhance script output with additional metadata"""
    enhanced = {
        **script_data,
        'metadata': {
            'product_title': product.get('title'),
            'style': style,
            'duration': duration,
            'generated_at': time.time(),
            'category': classify_product_category(product),
            'price_point': analyze_price_point(product.get('price', '')),
            'target_audience': get_target_audience(product, style)
        },
        'production_notes': {
            'recommended_music': get_music_recommendation(style),
            'color_mood': get_color_mood(style),
            'pacing': get_pacing_recommendation(duration),
            'visual_style': get_visual_style(style)
        }
    }
    
    return enhanced


def get_music_recommendation(style: str) -> str:
    """Get music recommendation based on style"""
    music_styles = {
        'energetic': 'upbeat electronic or pop',
        'professional': 'corporate or ambient',
        'casual': 'indie or acoustic',
        'luxury': 'sophisticated or classical',
        'playful': 'fun and bouncy'
    }
    return music_styles.get(style, 'upbeat and positive')


def get_color_mood(style: str) -> str:
    """Get color mood based on style"""
    color_moods = {
        'energetic': 'bright and vibrant',
        'professional': 'clean and minimal',
        'casual': 'warm and friendly',
        'luxury': 'elegant and sophisticated',
        'playful': 'colorful and fun'
    }
    return color_moods.get(style, 'positive and engaging')


def get_pacing_recommendation(duration: int) -> str:
    """Get pacing recommendation based on duration"""
    if duration <= 15:
        return "fast-paced and punchy"
    elif duration <= 30:
        return "moderate with clear sections"
    else:
        return "relaxed with detailed explanations"


def get_visual_style(style: str) -> str:
    """Get visual style recommendation"""
    visual_styles = {
        'energetic': 'dynamic with quick cuts',
        'professional': 'clean and structured',
        'casual': 'natural and authentic',
        'luxury': 'premium and polished',
        'playful': 'creative and animated'
    }
    return visual_styles.get(style, 'engaging and clear')


def clean_json_response(content: str) -> str:
    """Clean JSON response from potential formatting issues"""
    # Remove any markdown code blocks
    content = content.replace('```json', '').replace('```', '')
    
    # Remove any leading/trailing whitespace
    content = content.strip()
    
    # Find JSON object bounds
    start = content.find('{')
    end = content.rfind('}') + 1
    
    if start != -1 and end != 0:
        content = content[start:end]
    
    return content


def get_default_video_metadata() -> Dict[str, Any]:
    """Return default video metadata if generation fails"""
    return {
        "color_scheme": {
            "primary": "#2563eb",
            "secondary": "#64748b",
            "accent": "#f59e0b",
            "background": "#ffffff"
        },
        "typography": {
            "primary_font": "Inter",
            "secondary_font": "Inter",
            "font_weight": "bold"
        },
        "animations": ["fade_in", "slide_up"],
        "music_style": "upbeat",
        "visual_effects": ["gradient_overlays"],
        "layout_suggestions": {
            "product_position": "center",
            "text_alignment": "center",
            "logo_placement": "bottom_right"
        }
    }


def get_default_sentiment_analysis() -> Dict[str, Any]:
    """Return default sentiment analysis if generation fails"""
    return {
        "target_audience": "general consumers",
        "emotional_triggers": ["convenience", "value", "quality"],
        "value_proposition": "great product at a fair price",
        "competitive_advantage": "unique features and benefits",
        "urgency_factors": ["limited availability"],
        "trust_signals": ["customer reviews"],
        "recommended_tone": "professional"
    }


def generate_ad_script_v2(
    product: Dict[str, Any],
    style: str = "energetic",
    duration: int = 30,
    industry: str = "ecommerce",
    platform: str = None,
    emotional_triggers: List[str] = None
) -> Dict[str, Any]:
    """
    Enhanced script generation with industry and platform optimization.

    Args:
        product: Product information dictionary
        style: Ad style (energetic, professional, casual, luxury)
        duration: Target duration in seconds
        industry: Industry type (ecommerce, saas, services, health, finance, local_business)
        platform: Target platform (tiktok, instagram, youtube, linkedin, facebook, twitter)
        emotional_triggers: List of emotional triggers to emphasize

    Returns:
        Enhanced script dictionary with platform-optimized copy
    """
    try:
        # Get industry-specific configuration
        industry_config = INDUSTRY_PROMPTS.get(industry, INDUSTRY_PROMPTS["ecommerce"])

        # Get platform guidelines if specified
        platform_config = None
        if platform:
            platform_config = PLATFORM_COPY_GUIDELINES.get(platform)

        # Build enhanced prompt with industry and platform context
        timing = get_timing_breakdown(duration)
        prompt = build_enhanced_script_prompt(
            product, style, timing, industry_config, platform_config, emotional_triggers
        )

        # Generate script
        script_data = call_openai_with_retry(prompt, "enhanced_script")

        # Add metadata
        script_data['metadata'] = {
            'product_title': product.get('title'),
            'style': style,
            'duration': duration,
            'industry': industry,
            'platform': platform,
            'emotional_triggers': emotional_triggers or [],
            'generated_at': time.time()
        }

        logger.info(f"Generated enhanced script for {industry} / {platform or 'general'}")
        return script_data

    except Exception as e:
        logger.error(f"Enhanced script generation failed: {e}")
        # Fallback to basic script generation
        return generate_ad_script(product, style, duration)


def build_enhanced_script_prompt(
    product: Dict[str, Any],
    style: str,
    timing: Dict[str, int],
    industry_config: Dict[str, Any],
    platform_config: Optional[Dict[str, Any]],
    emotional_triggers: Optional[List[str]]
) -> str:
    """Build enhanced prompt with industry and platform specifics."""

    features_text = format_features(product.get('features', []))
    category = classify_product_category(product)

    # Platform-specific constraints
    word_limit = ""
    platform_guidance = ""
    if platform_config:
        word_limit = f"""
    WORD LIMITS (CRITICAL):
    - Hook: Maximum {platform_config['max_hook_words']} words
    - Total script: Maximum {platform_config['max_total_words']} words
    """
        platform_guidance = f"""
    PLATFORM OPTIMIZATION:
    - Tone: {platform_config['tone']}
    - CTA Style: {platform_config['cta_style']}
    - Emoji usage: {platform_config['emoji_usage']}
    - AVOID: {', '.join(platform_config.get('avoid', []))}
    """

    # Emotional triggers
    triggers_text = ""
    if emotional_triggers:
        triggers_text = f"""
    EMOTIONAL TRIGGERS TO EMPHASIZE:
    - {', '.join(emotional_triggers)}
    """

    return f"""
    Create a {timing['total']}-second video ad script optimized for {industry_config.get('tone', 'engaging')} marketing.

    PRODUCT:
    Title: {product.get('title')}
    Description: {product.get('description', 'N/A')}
    Price: {product.get('price')}
    Features: {features_text}
    Category: {category}

    INDUSTRY CONTEXT ({category}):
    - Tone: {industry_config['tone']}
    - Emotional focus: {industry_config['emotional_focus']}
    - CTA examples: {industry_config['cta_style']}
    - Power words to use: {', '.join(industry_config['power_words'])}
    {platform_guidance}
    {word_limit}
    {triggers_text}

    TIMING:
    - Hook: {timing['hook']}s (MUST grab attention instantly)
    - Pitch: {timing['pitch']}s (present the solution)
    - Features: {timing['features']}s (highlight benefits)
    - CTA: {timing['cta']}s (drive action)

    REQUIREMENTS:
    1. Hook MUST be punchy and attention-grabbing
    2. Use industry-appropriate power words
    3. Include social proof or credibility elements
    4. Create urgency without being pushy
    5. CTA should match industry conventions

    Return JSON:
    {{
        "hook": "Attention-grabbing opening",
        "pitch": "Clear value proposition",
        "features": "Key benefits with proof",
        "cta": "Compelling call-to-action",
        "visual_cues": ["Scene descriptions"],
        "key_messages": ["message1", "message2", "message3"],
        "ab_hooks": ["Alternative hook 1", "Alternative hook 2", "Alternative hook 3"]
    }}
    """


def generate_ab_hooks_enhanced(
    product: Dict[str, Any],
    strategies: List[str] = None,
    count: int = 5,
    platform: str = None,
    industry: str = "ecommerce"
) -> List[Dict[str, Any]]:
    """
    Generate A/B testable hooks with strategy labels and predicted scores.

    Args:
        product: Product information
        strategies: Hook strategies to use
        count: Number of hooks to generate
        platform: Target platform
        industry: Industry type

    Returns:
        List of hook dicts with text, strategy, and predicted_score
    """
    if strategies is None:
        strategies = ["question", "social_proof", "curiosity", "urgency", "transformation"]

    try:
        # Get configurations
        industry_config = INDUSTRY_PROMPTS.get(industry, INDUSTRY_PROMPTS["ecommerce"])
        platform_config = PLATFORM_COPY_GUIDELINES.get(platform) if platform else None

        max_words = platform_config['max_hook_words'] if platform_config else 12

        prompt = f"""
        Generate {count} unique, high-converting hooks for this product.

        Product: {product.get('title')}
        Price: {product.get('price')}
        Key benefit: {get_main_benefit(product)}

        Industry: {industry}
        Platform: {platform or 'general'}
        Max words per hook: {max_words}

        Strategies to use: {', '.join(strategies)}
        Power words: {', '.join(industry_config['power_words'])}

        Return JSON array with EXACTLY this format:
        [
            {{"text": "hook text", "strategy": "strategy_name", "predicted_score": 0.85}},
            ...
        ]

        Score based on:
        - Clarity (0-0.3)
        - Emotional impact (0-0.3)
        - Platform fit (0-0.2)
        - Uniqueness (0-0.2)
        """

        response = call_openai_with_retry(prompt, "ab_hooks", model=DEFAULT_MODEL)

        if isinstance(response, list):
            return response[:count]
        else:
            logger.warning("Unexpected AB hooks response format")
            return [{"text": f"Discover {product.get('title', 'this product')}!", "strategy": "general", "predicted_score": 0.7}]

    except Exception as e:
        logger.error(f"AB hooks generation failed: {e}")
        return [{"text": f"Check out {product.get('title', 'this')}!", "strategy": "general", "predicted_score": 0.6}]


def get_industry_config(industry: str) -> Dict[str, Any]:
    """Get industry configuration."""
    return INDUSTRY_PROMPTS.get(industry, INDUSTRY_PROMPTS["ecommerce"])


def get_platform_config(platform: str) -> Optional[Dict[str, Any]]:
    """Get platform configuration."""
    return PLATFORM_COPY_GUIDELINES.get(platform)


# Export functions for use in main application
__all__ = [
    'generate_ad_script',
    'generate_ad_script_v2',
    'generate_video_metadata',
    'generate_hook_variations',
    'generate_ab_hooks_enhanced',
    'analyze_product_sentiment',
    'get_industry_config',
    'get_platform_config',
    'INDUSTRY_PROMPTS',
    'PLATFORM_COPY_GUIDELINES',
    'AIServiceError'
]