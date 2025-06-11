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
from ai_service import generate_ad_script

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
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        job_id = data.get('job_id') or str(uuid.uuid4())
        product = data.get('product')

        if not product or not product.get('title'):
            return jsonify({'error': 'Invalid product data'}), 400

        script = generate_ad_script(product)

        output = {
            "job_id": job_id,
            "title": product.get('title'),
            "price": product.get('price'),
            "images": product.get('images'),
            "script": script
        }

        os.makedirs("remotion/input", exist_ok=True)
        with open(f"remotion/input/{job_id}.json", "w") as f:
            json.dump(output, f, indent=2)

        return jsonify({"success": True, "job_id": job_id, "script": script}), 200

    except Exception as e:
        logger.error(f"Error in generate_content: {str(e)}")
        return jsonify({'error': 'Failed to generate script'}), 500



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