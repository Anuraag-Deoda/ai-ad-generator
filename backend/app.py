from flask import Flask, request, jsonify
from flask_cors import CORS
import validators
import logging
import time
import uuid
import os
import json
import subprocess
from dotenv import load_dotenv
from scraper import scrape_product_data
from ai_service import (
    generate_ad_script, 
    generate_video_metadata, 
    generate_hook_variations,
    analyze_product_sentiment,
    AIServiceError
)


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
        os.makedirs("remotion/input", exist_ok=True)
        with open(f"remotion/input/{job_id}.json", "w") as f:
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


@app.route('/api/generate-video-batch', methods=['POST'])
def generate_video_batch():
    """Generate videos for all jobs in a batch"""
    try:
        data = request.get_json()
        batch_id = data.get('batch_id')
        
        if not batch_id:
            return jsonify({'error': 'batch_id required'}), 400

        batch_path = f"remotion/input/batch_{batch_id}.json"
        if not os.path.exists(batch_path):
            return jsonify({'error': 'Batch not found'}), 404

        with open(batch_path, 'r') as f:
            batch_data = json.load(f)

        successful_jobs = [r for r in batch_data['results'] if r['status'] == 'success']
        video_results = []

        logger.info(f"Starting video generation for {len(successful_jobs)} jobs in batch {batch_id}")

        for job in successful_jobs:
            job_id = job['job_id']
            try:
                # Generate video using existing video generation logic
                output_path = f"remotion/public/out/{job_id}.mp4"
                
                cmd = [
                    "npx", "remotion", "render",
                    "src/index.tsx", "MyAdVideo",
                    "--props", f'{{"jobId":"{job_id}"}}',
                    "--output", output_path
                ]

                process = subprocess.run(cmd, cwd="remotion", capture_output=True, text=True, timeout=300)

                if process.returncode == 0:
                    video_results.append({
                        "job_id": job_id,
                        "style": job['style'],
                        "duration": job['duration'],
                        "status": "success",
                        "video_path": f"/out/{job_id}.mp4"
                    })
                    logger.info(f"Successfully generated video for {job_id}")
                else:
                    video_results.append({
                        "job_id": job_id,
                        "style": job['style'],
                        "duration": job['duration'],
                        "status": "failed",
                        "error": process.stderr
                    })
                    logger.error(f"Video generation failed for {job_id}: {process.stderr}")

            except subprocess.TimeoutExpired:
                video_results.append({
                    "job_id": job_id,
                    "style": job['style'],
                    "duration": job['duration'],
                    "status": "failed",
                    "error": "Video generation timeout"
                })
                logger.error(f"Video generation timeout for {job_id}")
            except Exception as e:
                video_results.append({
                    "job_id": job_id,
                    "style": job['style'],
                    "duration": job['duration'],
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"Video generation error for {job_id}: {str(e)}")

        # Update batch data with video results
        batch_data['video_results'] = video_results
        batch_data['videos_generated_at'] = time.time()
        
        with open(batch_path, 'w') as f:
            json.dump(batch_data, f, indent=2)

        successful_videos = len([r for r in video_results if r['status'] == 'success'])
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "total_videos": len(video_results),
            "successful_videos": successful_videos,
            "failed_videos": len(video_results) - successful_videos,
            "video_results": video_results
        }), 200

    except Exception as e:
        logger.error(f"Error in batch video generation: {str(e)}")
        return jsonify({'error': 'Batch video generation failed'}), 500


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



@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    try:
        data = request.get_json()
        job_id = data.get('job_id')

        if not job_id:
            return jsonify({'error': 'job_id required'}), 400

        input_path = f"remotion/input/{job_id}.json"
        output_path = f"remotion/public/out/{job_id}.mp4"

        cmd = [
            "npx", "remotion", "render",
            "src/index.tsx", "MyAdVideo",
            "--props", f'{{"jobId":"{job_id}"}}',
            "--output", output_path
        ]

        process = subprocess.run(cmd, cwd="remotion", capture_output=True, text=True)

        if process.returncode != 0:
            logger.error(f"Remotion error: {process.stderr}")
            return jsonify({'error': 'Video generation failed', 'details': process.stderr}), 500

        return jsonify({"success": True, "video_path": f"/out/{job_id}.mp4"}), 200

    except Exception as e:
        logger.error(f"Error in generate_video: {str(e)}")
        return jsonify({'error': 'Internal error generating video'}), 500



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