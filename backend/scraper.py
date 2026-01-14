import requests
from bs4 import BeautifulSoup
import time
import random
import json
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedProductScraper:
    # Pool of realistic user agents (updated browsers)
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    ]

    def __init__(self):
        self.session = requests.Session()
        self._rotate_headers()

    def _rotate_headers(self):
        """Rotate user agent and headers to avoid detection"""
        user_agent = random.choice(self.USER_AGENTS)
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"' if "Windows" in user_agent else '"macOS"',
        }
        self.session.headers.update(self.headers)

    def get_page_content(self, url, retries=5):
        """Get page content with retry logic, header rotation, and random delays"""
        domain = urlparse(url).netloc

        for attempt in range(retries):
            try:
                # Rotate headers on each attempt
                self._rotate_headers()

                # Add referer to look more legitimate
                self.session.headers["Referer"] = f"https://www.google.com/search?q={domain}"

                # Random delay - longer delays for retries
                delay = random.uniform(2, 4) + (attempt * random.uniform(1, 3))
                time.sleep(delay)

                # For Amazon, try mobile URL as fallback on later attempts
                request_url = url
                if attempt >= 2 and 'amazon' in domain:
                    # Try mobile version which may have less strict bot detection
                    request_url = url.replace('www.amazon', 'm.amazon')
                    logger.info(f"Trying mobile URL: {request_url}")

                response = self.session.get(request_url, timeout=20, allow_redirects=True)
                response.raise_for_status()

                # Check if we got blocked (common Amazon response)
                content_lower = response.text.lower()
                if "robot" in content_lower or "captcha" in content_lower or "automated access" in content_lower:
                    logger.warning(f"Potential bot detection on attempt {attempt + 1}")
                    # Clear session cookies and create new session
                    self.session = requests.Session()
                    if attempt < retries - 1:
                        time.sleep(random.uniform(5, 10))
                        continue

                # Additional check for empty/minimal response
                if len(response.text) < 1000:
                    logger.warning(f"Suspiciously short response on attempt {attempt + 1}")
                    if attempt < retries - 1:
                        continue

                return response.text

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {str(e)}")
                # Reset session on connection errors
                self.session = requests.Session()
                if attempt < retries - 1:
                    time.sleep(random.uniform(3, 6))
                    continue

        return None

    def extract_json_ld(self, soup):
        """Extract structured data from JSON-LD"""
        try:
            json_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        data = data[0]
                    if data.get('@type') == 'Product':
                        return data
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.warning(f"JSON-LD extraction failed: {str(e)}")
        return None

    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-\.\,\!\?\$\%\(\)]', '', text)
        return text

    def extract_price(self, text):
        """Extract price from text using regex"""
        if not text:
            return ""
        # Look for price patterns like $19.99, £25.50, €30.00, etc.
        price_patterns = [
            r'[\$£€¥₹]\s*[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*[\$£€¥₹]',
            r'USD\s*[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*USD'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
        return text.strip()

    def _extract_asin_from_url(self, url):
        """Extract ASIN from Amazon URL"""
        patterns = [
            r'/dp/([A-Z0-9]{10})',
            r'/gp/product/([A-Z0-9]{10})',
            r'/product/([A-Z0-9]{10})',
            r'asin=([A-Z0-9]{10})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def scrape_amazon_product(self, url):
        """Advanced Amazon product scraping with multiple fallback strategies"""
        try:
            # Extract ASIN for potential fallback
            asin = self._extract_asin_from_url(url)

            html_content = self.get_page_content(url)
            if not html_content:
                if asin:
                    return {
                        "error": "Amazon blocked the request. Please try manually entering product details.",
                        "asin": asin,
                        "suggestion": "You can try again in a few minutes or enter the product information manually."
                    }
                return {"error": "Failed to fetch Amazon page - please try again later"}

            soup = BeautifulSoup(html_content, 'html.parser')

            # Check for bot detection
            if soup.find('form', {'action': '/errors/validateCaptcha'}):
                return {
                    "error": "Amazon is showing a CAPTCHA. Please try again in a few minutes.",
                    "asin": asin,
                    "suggestion": "Amazon's bot protection triggered. Wait 2-3 minutes and try again."
                }

            result = {
                "platform": "amazon",
                "title": "",
                "price": "",
                "features": [],
                "description": "",
                "images": [],
                "rating": "",
                "availability": ""
            }

            # Try JSON-LD first (most reliable)
            json_data = self.extract_json_ld(soup)
            if json_data:
                result["title"] = self.clean_text(json_data.get('name', ''))
                if 'offers' in json_data:
                    offers = json_data['offers']
                    if isinstance(offers, list):
                        offers = offers[0]
                    result["price"] = self.extract_price(str(offers.get('price', '')))
                if 'image' in json_data:
                    images = json_data['image']
                    if isinstance(images, list):
                        result["images"] = images[:3]  # Limit to 3 images
                    else:
                        result["images"] = [images]

            # Title extraction with multiple selectors
            if not result["title"]:
                title_selectors = [
                    '#productTitle',
                    '.product-title',
                    '[data-automation-id="product-title"]',
                    'h1.a-size-large',
                    'h1 span',
                    '.it-ttl'
                ]
                
                for selector in title_selectors:
                    title_elem = soup.select_one(selector)
                    if title_elem and title_elem.get_text().strip():
                        result["title"] = self.clean_text(title_elem.get_text())
                        break

            # Price extraction with multiple selectors
            if not result["price"]:
                price_selectors = [
                    '.a-price .a-offscreen',
                    '#priceblock_ourprice',
                    '#priceblock_dealprice',
                    '.a-price-current .a-offscreen',
                    '.a-price-to-pay .a-offscreen',
                    '.a-price-range .a-offscreen',
                    '[data-automation-id="product-price"]',
                    '.a-price .a-price-whole',
                    '#apex_desktop .a-price .a-offscreen'
                ]
                
                for selector in price_selectors:
                    price_elem = soup.select_one(selector)
                    if price_elem:
                        price_text = price_elem.get_text().strip()
                        if price_text and ('$' in price_text or '£' in price_text or '€' in price_text or '₹' in price_text):
                            result["price"] = self.extract_price(price_text)
                            break

            # Image extraction
            if not result["images"]:
                image_selectors = [
                    '#landingImage',
                    '#imgTagWrapperId img',
                    '.a-dynamic-image',
                    '[data-automation-id="product-image"]',
                    '.imgTagWrapper img'
                ]
                
                for selector in image_selectors:
                    img_elem = soup.select_one(selector)
                    if img_elem:
                        src = img_elem.get('src') or img_elem.get('data-src')
                        if src and 'http' in src:
                            result["images"] = [src]
                            break

            # Features extraction
            feature_selectors = [
                '#feature-bullets ul li span.a-list-item',
                '#feature-bullets ul li',
                '.a-unordered-list .a-list-item',
                '[data-automation-id="product-highlights"] li',
                '.feature .a-list-item'
            ]
            
            for selector in feature_selectors:
                features = soup.select(selector)
                if features:
                    for feature in features[:5]:  # Limit to 5 features
                        text = self.clean_text(feature.get_text())
                        if text and len(text) > 10 and text not in result["features"]:
                            result["features"].append(text)
                    if result["features"]:
                        break

            # Description extraction
            desc_selectors = [
                '#productDescription p',
                '#aplus_feature_div',
                '[data-automation-id="product-overview"]',
                '.a-expander-content p'
            ]
            
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    desc_text = self.clean_text(desc_elem.get_text())
                    if desc_text and len(desc_text) > 20:
                        result["description"] = desc_text[:500]  # Limit description length
                        break

            # Rating extraction
            rating_selectors = [
                '.a-icon-alt',
                '[data-automation-id="product-rating"]',
                '.a-star-medium .a-icon-alt'
            ]
            
            for selector in rating_selectors:
                rating_elem = soup.select_one(selector)
                if rating_elem:
                    rating_text = rating_elem.get('alt') or rating_elem.get_text()
                    if rating_text and 'out of' in rating_text:
                        result["rating"] = rating_text
                        break

            # Availability
            availability_selectors = [
                '#availability span',
                '[data-automation-id="product-availability"]',
                '.a-size-medium .a-color-success'
            ]
            
            for selector in availability_selectors:
                avail_elem = soup.select_one(selector)
                if avail_elem:
                    avail_text = self.clean_text(avail_elem.get_text())
                    if avail_text:
                        result["availability"] = avail_text
                        break

            # Validate we got minimum required data
            if not result["title"] and not result["price"]:
                return {
                    "error": "Could not extract product information. Amazon may have blocked the request.",
                    "asin": asin,
                    "suggestion": "Try again in 2-3 minutes, or manually enter the product details."
                }

            # Remove empty fields
            result = {k: v for k, v in result.items() if v}
            
            logger.info(f"Successfully scraped Amazon product: {result.get('title', 'Unknown')[:50]}")
            return result

        except Exception as e:
            logger.error(f"Amazon scraping failed: {str(e)}")
            return {"error": f"Amazon scrape failed: {str(e)}"}

    def scrape_shopify_product(self, url):
        """Advanced Shopify product scraping"""
        try:
            html_content = self.get_page_content(url)
            if not html_content:
                return {"error": "Failed to fetch Shopify page"}

            soup = BeautifulSoup(html_content, 'html.parser')
            
            result = {
                "platform": "shopify",
                "title": "",
                "price": "",
                "features": [],
                "description": "",
                "images": [],
                "rating": "",
                "availability": ""
            }

            # Try JSON-LD first
            json_data = self.extract_json_ld(soup)
            if json_data:
                result["title"] = self.clean_text(json_data.get('name', ''))
                if 'offers' in json_data:
                    offers = json_data['offers']
                    if isinstance(offers, list):
                        offers = offers[0]
                    result["price"] = self.extract_price(str(offers.get('price', '')))

            # Title extraction
            if not result["title"]:
                title_selectors = [
                    'h1.product-single__title',
                    'h1.product__title',
                    '.product-title h1',
                    'h1',
                    '.product-single__meta h1'
                ]
                
                for selector in title_selectors:
                    title_elem = soup.select_one(selector)
                    if title_elem:
                        result["title"] = self.clean_text(title_elem.get_text())
                        break

            # Price extraction
            if not result["price"]:
                price_selectors = [
                    '.product-single__price',
                    '.product__price',
                    '.price',
                    '[data-product-price]',
                    '.money'
                ]
                
                for selector in price_selectors:
                    price_elem = soup.select_one(selector)
                    if price_elem:
                        result["price"] = self.extract_price(price_elem.get_text())
                        break

            # Images extraction
            img_selectors = [
                '.product-single__photo img',
                '.product__photo img',
                '.product-image-main img',
                'meta[property="og:image"]'
            ]
            
            for selector in img_selectors:
                if 'meta' in selector:
                    img_elem = soup.select_one(selector)
                    if img_elem and img_elem.get('content'):
                        result["images"] = [img_elem.get('content')]
                        break
                else:
                    img_elem = soup.select_one(selector)
                    if img_elem:
                        src = img_elem.get('src') or img_elem.get('data-src')
                        if src:
                            # Convert relative URLs to absolute
                            if src.startswith('//'):
                                src = 'https:' + src
                            elif src.startswith('/'):
                                src = urljoin(url, src)
                            result["images"] = [src]
                            break

            # Description extraction
            desc_selectors = [
                '.product-single__description',
                '.product__description',
                '.product-description',
                'meta[name="description"]'
            ]
            
            for selector in desc_selectors:
                if 'meta' in selector:
                    desc_elem = soup.select_one(selector)
                    if desc_elem and desc_elem.get('content'):
                        result["description"] = self.clean_text(desc_elem.get('content'))
                        break
                else:
                    desc_elem = soup.select_one(selector)
                    if desc_elem:
                        result["description"] = self.clean_text(desc_elem.get_text())[:500]
                        break

            # Features/bullet points
            feature_selectors = [
                '.product-single__description ul li',
                '.product__description ul li',
                '.product-features li'
            ]
            
            for selector in feature_selectors:
                features = soup.select(selector)
                if features:
                    for feature in features[:5]:
                        text = self.clean_text(feature.get_text())
                        if text and len(text) > 5:
                            result["features"].append(text)
                    if result["features"]:
                        break

            # Remove empty fields
            result = {k: v for k, v in result.items() if v}
            
            if not result["title"]:
                return {"error": "Could not extract product title from Shopify store"}
                
            logger.info(f"Successfully scraped Shopify product: {result.get('title', 'Unknown')[:50]}")
            return result

        except Exception as e:
            logger.error(f"Shopify scraping failed: {str(e)}")
            return {"error": f"Shopify scrape failed: {str(e)}"}

    def scrape_generic_product(self, url):
        """Generic e-commerce scraping for other platforms"""
        try:
            html_content = self.get_page_content(url)
            if not html_content:
                return {"error": "Failed to fetch page"}

            soup = BeautifulSoup(html_content, 'html.parser')
            
            result = {
                "platform": "generic",
                "title": "",
                "price": "",
                "features": [],
                "description": "",
                "images": []
            }

            # Try JSON-LD first
            json_data = self.extract_json_ld(soup)
            if json_data:
                result["title"] = self.clean_text(json_data.get('name', ''))
                if 'offers' in json_data:
                    offers = json_data['offers']
                    if isinstance(offers, list):
                        offers = offers[0]
                    result["price"] = self.extract_price(str(offers.get('price', '')))

            # Fallback to meta tags and common selectors
            if not result["title"]:
                # Try Open Graph and meta tags first
                og_title = soup.select_one('meta[property="og:title"]')
                if og_title:
                    result["title"] = self.clean_text(og_title.get('content', ''))
                else:
                    # Generic title selectors
                    title_elem = soup.select_one('h1') or soup.find('title')
                    if title_elem:
                        result["title"] = self.clean_text(title_elem.get_text())

            # Generic price detection
            if not result["price"]:
                price_text = soup.get_text()
                price_match = re.search(r'[\$£€¥₹]\s*[\d,]+\.?\d*', price_text)
                if price_match:
                    result["price"] = price_match.group(0)

            # Generic image
            og_image = soup.select_one('meta[property="og:image"]')
            if og_image:
                result["images"] = [og_image.get('content')]

            # Generic description
            og_desc = soup.select_one('meta[property="og:description"]')
            meta_desc = soup.select_one('meta[name="description"]')
            if og_desc:
                result["description"] = self.clean_text(og_desc.get('content', ''))
            elif meta_desc:
                result["description"] = self.clean_text(meta_desc.get('content', ''))

            # Remove empty fields
            result = {k: v for k, v in result.items() if v}
            
            if not result["title"]:
                return {"error": "Could not extract product information from this website"}
                
            return result

        except Exception as e:
            logger.error(f"Generic scraping failed: {str(e)}")
            return {"error": f"Generic scrape failed: {str(e)}"}


# Main scraping function
def scrape_product_data(url):
    scraper = AdvancedProductScraper()
    
    try:
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        domain = urlparse(url).netloc.lower()
        
        # Platform detection
        if 'amazon' in domain:
            logger.info(f"Detected Amazon URL: {domain}")
            return scraper.scrape_amazon_product(url)
        elif any(platform in domain for platform in ['shopify', 'myshopify']):
            logger.info(f"Detected Shopify URL: {domain}")
            return scraper.scrape_shopify_product(url)
        elif 'products/' in url or 'product/' in url:
            logger.info(f"Detected potential e-commerce URL: {domain}")
            # Try Shopify first, then generic
            result = scraper.scrape_shopify_product(url)
            if 'error' in result:
                return scraper.scrape_generic_product(url)
            return result
        else:
            logger.info(f"Using generic scraper for: {domain}")
            return scraper.scrape_generic_product(url)
            
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {str(e)}")
        return {"error": f"Failed to scrape product data: {str(e)}"}


# Test function
if __name__ == "__main__":
    test_urls = [
        "https://www.amazon.com/dp/B08N5WRWNW", 
        "https://example.myshopify.com/products/test-product"  
    ]
    
    for test_url in test_urls:
        print(f"\nTesting: {test_url}")
        result = scrape_product_data(test_url)
        print(json.dumps(result, indent=2))