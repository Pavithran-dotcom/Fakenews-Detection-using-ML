from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import requests
import re
import time
import os
from urllib.parse import quote, urlparse
from bs4 import BeautifulSoup
import google.generativeai as genai


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication


# Load your trained model
try:
    model = joblib.load('models/fake_news_model.pkl')
    print("✓ ML Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None


# ========== API CONFIGURATION ==========
# Add your actual API keys here or set as environment variables
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY', 'your gnews apikey')
GEMINI_API_KEY = ''

# Configure Gemini
if GEMINI_API_KEY and GEMINI_API_KEY != '':
    
    
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
    print("✓ Gemini API configured")
else:
    gemini_model = None
    print("⚠ Gemini API key not set")


TRUSTED_DOMAINS = [
    'reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com',
    'washingtonpost.com', 'cnn.com', 'theguardian.com',
    'wsj.com', 'bloomberg.com', 'npr.org', 'aljazeera.com',
    'indianexpress.com', 'thehindu.com', 'ndtv.com',
    'forbes.com', 'time.com', 'newsweek.com', 'espn.com',
    'medicalnewstoday.com', 'health.harvard.edu', 'who.int'
]


KNOWN_FAKE_DOMAINS = [
    'naturalnews.com', 'infowars.com', 'beforeitsnews.com',
    'yournewswire.com', 'activistpost.com', 'globalresearch.ca'
]


# ========== URL SCRAPING FUNCTION ==========
def scrape_article_from_url(url):
    """
    Scrape article text from any news URL
    """
    try:
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        print(f"\nScraping URL: {url}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
        }

        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'noscript', 'iframe', 'ads', 'advertisement']):
            tag.decompose()

        # Extract title
        title = ""
        title_selectors = [
            ('h1', {'class': 'headline'}),
            ('h1', {'class': 'article-title'}),
            ('h1', {'class': 'entry-title'}),
            ('h1', {}),
            ('title', {})
        ]

        for tag, attrs in title_selectors:
            elem = soup.find(tag, attrs)
            if elem:
                title = elem.get_text(strip=True)
                if len(title) > 10:
                    break

        # Extract article text
        article_text = ""

        # Try article tag first
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
            article_text = ' '.join(
                [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]
            )

        # Try common content selectors
        if len(article_text) < 200:
            selectors = [
                '[itemprop="articleBody"]',
                '.article-content',
                '.story-content',
                '.news-content',
                '.entry-content',
                '.post-content',
                '.content__article-body',
                '.article__body',
                '.story-body',
                '#article-body',
                '.field-body'
            ]

            for selector in selectors:
                container = soup.select_one(selector)
                if container:
                    paragraphs = container.find_all('p')
                    article_text = ' '.join(
                        [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]
                    )
                    if len(article_text) > 200:
                        break

        # Fallback: get all paragraphs from main
        if len(article_text) < 200:
            main = soup.find('main') or soup.find('body')
            if main:
                paragraphs = main.find_all('p')
                article_text = ' '.join(
                    [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]
                )

        # Clean text
        full_text = f"{title} {article_text}".strip()
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = re.sub(r'\n+', ' ', full_text)

        if len(full_text) < 100:
            return None, "Could not extract enough text from this URL. The site may use JavaScript to load content or block scraping."

        domain = urlparse(url).netloc.replace('www.', '')

        print(f"✓ Extracted {len(full_text)} characters from {domain}")

        return {
            'text': full_text[:8000],
            'title': title,
            'domain': domain,
            'url': url
        }, None

    except requests.exceptions.Timeout:
        return None, "Request timed out. The website is slow or unresponsive."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to this website. It may be blocking automated access."
    except requests.exceptions.RequestException as e:
        return None, f"Cannot access URL: {str(e)}"
    except Exception as e:
        return None, f"Error scraping: {str(e)}"


# ========== GEMINI AI ANALYSIS ==========
def analyze_with_gemini(text, title=""):
    """
    Use Google's Gemini AI to analyze the news article
    """
    if not gemini_model:
        return None

    try:
        prompt = f"""Analyze this news article and determine if it's likely to be REAL or FAKE news.

Title: {title}

Article Text: {text[:3000]}

Please analyze:
1. Writing style and tone
2. Presence of sensational language
3. Credibility indicators
4. Factual consistency
5. Source references

Return your response in this exact JSON format:
{{
    "verdict": "REAL" or "FAKE" or "UNCERTAIN",
    "confidence": 0-100,
    "reasoning": "brief explanation",
    "red_flags": ["list", "of", "concerning", "elements"],
    "positive_indicators": ["list", "of", "credible", "elements"]
}}"""

        response = gemini_model.generate_content(prompt)

        # Parse the response
        response_text = response.text

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            import json
            result = json.loads(json_match.group())
            return result

        return None

    except Exception as e:
        print(f"Gemini API error: {e}")
        return None


# ========== GNEWS FACT CHECK ==========
def check_gnews_api(text):
    """
    Check GNews API for similar articles
    """
    if not GNEWS_API_KEY or GNEWS_API_KEY == '01818c115243c12cf11afae1275b7d11':
        return None

    try:
        # Extract key terms (first 5 words)
        words = text.split()
        stopwords = {'the','a','an','is','was','are','were','in','on','at','of','to','and','that','this','it','for','with','has'}
        meaningful = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', text) if w.lower() not in stopwords]
        key_terms = ' '.join(meaningful[:5]) if meaningful else ' '.join(words[:5])

        url = f"https://gnews.io/api/v4/search?q={quote(key_terms)}&apikey={GNEWS_API_KEY}&max=10&lang=en&country=us"

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            if articles:
                sources = list(set([article['source']['name'] for article in articles]))
                titles = [article['title'] for article in articles[:3]]

                # Check if trusted sources are covering this
                trusted_found = [
                    s for s in sources
                    if any(t in s.lower() for t in ['reuters', 'ap', 'bbc', 'cnn', 'guardian', 'times'])
                ]

                return {
                    'found': True,
                    'total_articles': len(articles),
                    'sources': sources[:5],
                    'trusted_sources': trusted_found,
                    'titles': titles,
                    'api_used': 'GNews API'
                }
            else:
                return {
                    'found': False,
                    'sources': [],
                    'api_used': 'GNews API'
                }
        else:
            return None

    except Exception as e:
        print(f"GNews API error: {e}")
        return None


# ========== FALLBACK SOURCE CHECK ==========
def check_google_news_rss(text):
    """
    Fallback using Google News RSS if GNews fails
    """
    try:
        stopwords = {'the','a','an','is','was','are','were','in','on','at','of','to','and','that','this','it','for','with','has'}
        meaningful = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', text) if w.lower() not in stopwords]
        key_terms = quote(' '.join(meaningful[:5]) if meaningful else ' '.join(text.split()[:5]))
        url = f"https://news.google.com/rss/search?q={key_terms}"

        response = requests.get(url, timeout=10)

        found_sources = []
        for domain in TRUSTED_DOMAINS:
            if domain.replace('.com', '').replace('.org', '') in response.text.lower():
                found_sources.append(domain)

        return {
            'found': len(found_sources) > 0,
            'sources': found_sources[:5],
            'count': len(found_sources),
            'api_used': 'Google News RSS (fallback)'
        }
    except Exception:
        return None


# ========== WRITING STYLE ANALYSIS ==========
def analyze_writing_style(text):
    indicators = {
        'exclamation_marks': text.count('!'),
        'caps_words': len([w for w in text.split() if w.isupper() and len(w) > 4]),
        'sensational_words': 0,
        'word_count': len(text.split()),
        'question_marks': text.count('?'),
        'all_caps_sentences': len(
            [s for s in text.split('.') if s.strip().isupper() and len(s.strip()) > 5]
        )
    }

    sensational = [
        'shocking', 'unbelievable', 'secret', 'conspiracy',
        'dont want you to know', 'miracle', 'cure', 'urgent',
        'banned', 'censored', 'what they dont tell you',
        'mainstream media hides', 'wake up', 'sheeple',
        'microchip', 'deep state', 'new world order',
        'share before', 'gets deleted', 'delete this',
        'hiding the truth', 'governments hiding',
        'doctors banned', 'bombshell', 'exposes',
        'shocking truth', 'big pharma', 'plandemic', 'hoax',
        'they dont want you', 'wake up people'
    ]

    # Remove 'breaking' and 'exclusive' and 'warning'
    # as these appear in legitimate news too

    text_lower = text.lower()
    indicators['sensational_words'] = sum(1 for word in sensational if word in text_lower)

    # Calculate sensationalism score
    score = 0

    # Only penalize excessive exclamation marks (more than 5)
    if indicators['exclamation_marks'] > 5:
        score += 15
    elif indicators['exclamation_marks'] > 3:
        score += 5

    # Sensational words — strong signal
    if indicators['sensational_words'] >= 4:
        score += indicators['sensational_words'] * 12
    elif indicators['sensational_words'] >= 2:
        score += indicators['sensational_words'] * 6
    elif indicators['sensational_words'] > 0:
        score += 3

    # Only penalize caps if excessive (quoted speech has some caps)
    if indicators['caps_words'] > 10:
        score += 10

    if indicators['all_caps_sentences'] > 0:
        score += 15

    indicators['sensationalism_score'] = min(score, 100)

    return indicators

# ========== DOMAIN CREDIBILITY CHECK ==========
def check_domain_credibility(domain):
    domain_lower = domain.lower()

    # Check trusted domains
    for trusted in TRUSTED_DOMAINS:
        if trusted in domain_lower:
            return 'trusted', trusted

    # Check known fake domains
    for fake in KNOWN_FAKE_DOMAINS:
        if fake in domain_lower:
            return 'fake', fake

    # Check for suspicious patterns
    suspicious_patterns = ['click', 'viral', 'shock', 'truth', 'real-news', 'worldtruth']
    for pattern in suspicious_patterns:
        if pattern in domain_lower:
            return 'suspicious', f"Contains '{pattern}'"

    return 'unknown', domain


# ========== FLASK ROUTES ==========
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data received'}), 400

        text_input = data.get('text', '').strip()
        url_input = data.get('url', '').strip()

        scraped_info = None
        text_to_analyze = ""
        source_domain = None

        # Process input
        if url_input:
            scraped_info, error = scrape_article_from_url(url_input)
            if error:
                return jsonify({'error': error}), 400

            text_to_analyze = scraped_info['text']
            source_domain = scraped_info['domain']
        elif text_input:
            text_to_analyze = text_input
            if len(text_to_analyze) < 50:
                return jsonify({'error': 'Text too short. Please provide at least 50 characters.'}), 400
        else:
            return jsonify({'error': 'Please provide either text or URL'}), 400

        # Limit text length
        if len(text_to_analyze) > 8000:
            text_to_analyze = text_to_analyze[:8000]

        results = {
            'ml_analysis': {},
            'gemini_analysis': {},
            'source_verification': {},
            'style_analysis': {},
            'domain_check': {},
            'final_verdict': {}
        }

        # 1. ML Model Prediction
        if model:
            prediction = model.predict([text_to_analyze])[0]
            probabilities = model.predict_proba([text_to_analyze])[0]

            ml_confidence = float(max(probabilities) * 100)
            ml_label = "REAL" if prediction == 1 else "FAKE"

            results['ml_analysis'] = {
                'prediction': ml_label,
                'confidence': round(ml_confidence, 2),
                'model_used': 'Custom Trained Model'
            }
        else:
            results['ml_analysis'] = {
                'prediction': 'N/A',
                'confidence': 0,
                'error': 'Model not loaded'
            }

        # 2. Gemini AI Analysis
        gemini_result = analyze_with_gemini(
            text_to_analyze,
            scraped_info['title'] if scraped_info else ""
        )
        if gemini_result:
            results['gemini_analysis'] = gemini_result
        else:
            results['gemini_analysis'] = {'error': 'Gemini API not available'}

        # 3. Source Verification (GNews or fallback)
        gnews_result = check_gnews_api(text_to_analyze)
        if gnews_result:
            results['source_verification'] = gnews_result
        else:
            results['source_verification'] = {'found': False, 'error': 'Source check unavailable'}

        # 4. Writing Style Analysis
        style = analyze_writing_style(text_to_analyze)
        results['style_analysis'] = style

        # 5. Domain Check
        if source_domain:
            domain_status, domain_info = check_domain_credibility(source_domain)
            results['domain_check'] = {
                'domain': source_domain,
                'status': domain_status,
                'info': domain_info
            }

        # 6. Calculate Final Verdict
        score = 0
        reasons = []
        warnings = []
        text_to_check = results.get('text_analyzed', '')
        word_count = len(text_to_check.split())
        if word_count > 200:
            score += 20  # longer articles tend to be real journalism
        elif word_count > 100:
            score += 15

        # ML contribution
        if model and results['ml_analysis']['prediction'] != 'N/A':
            ml_conf = results['ml_analysis'].get('confidence', 0)
            ml_pred = results['ml_analysis'].get('prediction')
            has_conspiracy = results['style_analysis'].get('sensational_words', 0) >= 2
            
            if ml_pred == 'REAL':
                if ml_conf >= 90:
                    score += 30
                    reasons.append(f"ML model confident real news ({ml_conf:.1f}%)")
                elif ml_conf >= 60:
                    score += 10
                    reasons.append(f"ML model suggests real news ({ml_conf:.1f}%)")
    
            elif ml_pred == 'FAKE':
                if has_conspiracy and ml_conf >= 70:
                    score -= 40
                    warnings.append(f"ML model suggests fake news ({ml_conf:.1f}%)")
                elif has_conspiracy:
                    score -= 20
                    warnings.append(f"Suspicious content detected")
                else:
                    score -= 5
                    warnings.append(f"ML model uncertain about this content")
                
        # Gemini contribution
        if 'verdict' in results['gemini_analysis']:
            gemini = results['gemini_analysis']
            if gemini['verdict'] == 'REAL':
                score += gemini['confidence'] * 0.2
                reasons.append(f"AI analysis: {gemini['reasoning']}")
            elif gemini['verdict'] == 'FAKE':
                score -= gemini['confidence'] * 0.25
                warnings.append(f"AI analysis: {gemini['reasoning']}")

            if 'red_flags' in gemini and gemini['red_flags']:
                warnings.extend(gemini['red_flags'])

        # Source verification contribution
        src = results['source_verification']
        if src.get('found'):
            trusted_count = len(src.get('trusted_sources', []))
            total_count = src.get('total_articles', 0) or src.get('count', 0)

            if trusted_count >= 2:
                score += 15
                reasons.append(f"Verified by {trusted_count} trusted sources")
            elif total_count >= 3:
                score += 5
                reasons.append(f"Reported by {total_count} news outlets")
            else:
                score -= 10
                warnings.append("Only weak or partial source matches found")
        else:
            score += 0   # reduced penalty - don't punish heavily for missing source
            warnings.append("No trusted news source verification found")
        # Domain contribution
        if results['domain_check']:
            domain_status = results['domain_check']['status']
            if domain_status == 'trusted':
                score += 20
                reasons.append(f"Trusted domain: {results['domain_check']['domain']}")
            elif domain_status == 'fake':
                score -= 40
                warnings.append(f"Known unreliable source: {results['domain_check']['info']}")
            elif domain_status == 'unknown':
                score -= 10
                warnings.append("Unknown or unverified news source domain")
            elif domain_status == 'suspicious':
                score -= 15
                warnings.append(f"Suspicious domain pattern: {results['domain_check']['info']}")

        # Style analysis contribution
        style_score = results['style_analysis'].get('sensationalism_score', 0)
        if style_score > 60:
            score -= 25
            warnings.append(f"Extremely sensational writing style (score: {style_score})")
        elif style_score > 40:
            score -= 15
            warnings.append(f"Sensational writing style detected (score: {style_score})")
        elif style_score > 20:
            warnings.append(f"Somewhat sensational writing (score: {style_score})")

        # Normalize score to 0-100
        final_score = max(0, min(100, 50 + score))
        
        # Only apply ML cap if domain is NOT trusted
        domain_trusted = (results.get('domain_check', {}).get('status') == 'trusted')

        ml_conf_check = results['ml_analysis'].get('confidence', 0)
        if results['ml_analysis'].get('prediction') == 'FAKE' and not domain_trusted and ml_conf_check > 85:
            ml_conf = results['ml_analysis'].get('confidence', 0)
            if ml_conf > 70:
                score = min(score, 10)
                final_score = max(0, min(100, 50 + score))
                warnings.append(f"ML model strongly indicates fake news ({ml_conf:.1f}% confidence)")
            elif ml_conf > 50:
                score = min(score, 30)
                final_score = max(0, min(100, 50 + score))

        # Conflict resolution: ML + Gemini say FAKE but sources exist
        ml_fake = results['ml_analysis'].get('prediction') == 'FAKE'
        gemini_fake = results['gemini_analysis'].get('verdict') == 'FAKE'
        sources_found = results['source_verification'].get('found', False)

        if ml_fake and gemini_fake and sources_found:
            score = min(score, 55)  # force UNCERTAIN
            final_score = max(0, min(100, 50 + score))
            warnings.append(
                "Conflicting evidence detected between content analysis and source verification"
            )

        # TRUSTED DOMAIN OVERRIDE
        # TO:
        if results.get('domain_check'):
            domain_status = results['domain_check'].get('status')
            print(f"DEBUG domain:{domain_status} score:{final_score}")
            if domain_status == 'trusted':
                if final_score < 80:
                    final_score = 80
                    reasons.append(f"Trusted domain override: {results['domain_check']['domain']}")
            elif domain_status == 'fake':
                if final_score > 50:
                    final_score = 25
                    warnings.append(f"Fake domain override: {results['domain_check']['domain']}")

        if final_score >= 80:
            verdict = "✅ VERIFIED REAL"
            verdict_class = "real"
            verdict_icon = "✓"
        elif final_score >= 60:
            verdict = "LIKELY REAL"
            verdict_class = "real"
            verdict_icon = "✓"
        elif final_score >= 45:
            verdict = "⚠️ UNCERTAIN"
            verdict_class = "uncertain"
            verdict_icon = "?"
        elif final_score >= 30:
            verdict = "⚠️ LIKELY FAKE"
            verdict_class = "fake"
            verdict_icon = "!"
        else:
            verdict = "❌ FAKE"
            verdict_class = "fake"
            verdict_icon = "✗"

        results['final_verdict'] = {
            'verdict': verdict,
            'verdict_class': verdict_class,
            'verdict_icon': verdict_icon,
            'credibility_score': round(final_score, 1),
            'reasons': reasons,
            'warnings': warnings,
            'input_type': 'url' if url_input else 'text',
            'article_title': scraped_info['title'] if scraped_info else "",
            'article_excerpt': text_to_analyze[:300] + "..." if len(text_to_analyze) > 300 else text_to_analyze
        }

        return jsonify(results)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

