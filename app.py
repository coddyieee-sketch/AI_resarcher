"""
Flask Backend for Agentic AI Research Assistant
Uses Perplexity AI for LLM and arXiv for paper search
COMPLETE FIXED VERSION - With timeout configuration and improved error handling
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx  # CRITICAL: Import for timeout configuration
import arxiv
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure logging with better format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================================================
# CRITICAL FIX: Initialize Perplexity client with PROPER TIMEOUT SETTINGS
# ============================================================================
perplexity_client = OpenAI(
    api_key=os.getenv('PERPLEXITY_API_KEY'),
    base_url="https://api.perplexity.ai",
    timeout=httpx.Timeout(
        connect=10.0,      # Connection timeout (10 seconds)
        read=120.0,        # Read timeout (2 minutes) - CRITICAL FOR LLM RESPONSES
        write=10.0,        # Write timeout (10 seconds)
        pool=10.0          # Pool timeout (10 seconds)
    ),
    max_retries=2          # Retry failed requests up to 2 times
)


class ResearchAssistant:
    """Main class for AI Research Assistant functionality"""

    def __init__(self):
        self.perplexity_model = "sonar-pro"

    def parse_problem_statement(self, problem: str) -> Dict[str, Any]:
        """Parse research problem using Perplexity AI - IMPROVED WITH BETTER ERROR HANDLING"""
        try:
            prompt = f"""Analyze this research problem statement and provide:
1. Key concepts and keywords (list of 5-8 keywords)
2. Research domains involved (e.g., computer vision, NLP, ML)
3. Specific clarifying questions to refine the problem
4. Suggested refinements to make it more specific

Research Problem: {problem}

Provide response in JSON format with keys: keywords, domains, clarifying_questions, suggestions"""

            logger.info(f"[PARSE_PROBLEM] Calling Perplexity API for: {problem[:50]}...")
            
            response = perplexity_client.chat.completions.create(
                model=self.perplexity_model,
                messages=[
                    {"role": "system", "content": "You are an AI research assistant expert in analyzing research problems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )

            logger.info("[PARSE_PROBLEM] Perplexity API call completed successfully")
            content = response.choices[0].message.content

            # Parse the response - IMPROVED FALLBACK
            try:
                parsed_data = json.loads(content)
                logger.info("[PARSE_PROBLEM] JSON parsing successful")
            except json.JSONDecodeError as je:
                logger.warning(f"[PARSE_PROBLEM] JSON parsing failed: {je}. Using fallback.")
                # Fallback if JSON parsing fails
                parsed_data = {
                    "keywords": self._extract_keywords(problem),
                    "domains": self._identify_domains(problem),
                    "clarifying_questions": [
                        "What is the specific application domain?",
                        "What are the evaluation metrics?",
                        "What datasets will be used?"
                    ],
                    "suggestions": [content]  # Include raw response as suggestion
                }

            return {
                "original_problem": problem,
                "parsed_data": parsed_data,
                "ai_response": content,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"[PARSE_PROBLEM] Error: {str(e)}", exc_info=True)
            # Return usable fallback data instead of just error
            return {
                "original_problem": problem,
                "error": str(e),
                "error_type": type(e).__name__,
                "parsed_data": {
                    "keywords": self._extract_keywords(problem),
                    "domains": self._identify_domains(problem),
                    "clarifying_questions": [
                        "What is the specific application domain?",
                        "What are the evaluation metrics?"
                    ],
                    "suggestions": [
                        "Please try again or rephrase your problem.",
                        "If the error persists, check your API connection."
                    ]
                },
                "timestamp": datetime.now().isoformat(),
                "status": "error_with_fallback"
            }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using simple NLP"""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'can', 'we', 'how', 'what', 'is', 'this', 'that'}
        words = text.lower().split()
        keywords = [w.strip('.,?!;:') for w in words if w not in stopwords and len(w) > 3]
        return list(set(keywords))[:8]

    def _identify_domains(self, text: str) -> List[str]:
        """Identify research domains from problem statement"""
        domains_map = {
            'computer_vision': ['vision', 'image', 'visual', 'cnn', 'object detection', 'detection', 'segmentation'],
            'nlp': ['language', 'text', 'nlp', 'bert', 'gpt', 'transformer', 'sentence', 'word'],
            'machine_learning': ['ml', 'machine learning', 'neural', 'deep learning', 'model', 'classification'],
            'interpretability': ['interpretability', 'explainable', 'xai', 'explain', 'transparency'],
            'data_science': ['data', 'statistics', 'analytics', 'dataset'],
            'reinforcement_learning': ['reinforcement', 'rl', 'agent', 'policy', 'reward']
        }

        text_lower = text.lower()
        found_domains = []
        for domain, keywords in domains_map.items():
            if any(kw in text_lower for kw in keywords):
                found_domains.append(domain)
        return found_domains if found_domains else ['general_ai']

    def search_arxiv_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for relevant papers"""
        try:
            logger.info(f"[SEARCH_ARXIV] Searching for: {query}")
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in search.results():
                papers.append({
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'updated': result.updated.strftime('%Y-%m-%d'),
                    'categories': result.categories,
                    'primary_category': result.primary_category
                })
            
            logger.info(f"[SEARCH_ARXIV] Found {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"[SEARCH_ARXIV] Error: {str(e)}", exc_info=True)
            return []

    def summarize_paper(self, paper_text: str, summary_type: str = 'general') -> Dict[str, Any]:
        """Summarize research paper using Perplexity AI"""
        try:
            prompts = {
                'general': 'Provide a concise general summary of this research paper.',
                'methodology': 'Explain the methodology used in this research paper in detail.',
                'results': 'Summarize the key results and findings from this research.',
                'conclusions': 'What are the main conclusions and implications of this research?'
            }

            prompt = f"""{prompts.get(summary_type, prompts['general'])}

Paper Text:
{paper_text[:4000]}

Provide a structured response with:
1. Summary (2-3 sentences)
2. Key Contributions (3-5 bullet points)
3. Limitations (2-3 points)
4. Future Work Suggestions (2-3 points)"""

            logger.info(f"[SUMMARIZE] Generating {summary_type} summary")
            
            response = perplexity_client.chat.completions.create(
                model=self.perplexity_model,
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing scientific papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.2
            )

            content = response.choices[0].message.content
            logger.info("[SUMMARIZE] Summary generated successfully")

            return {
                'summary_type': summary_type,
                'summary': content,
                'generated_at': datetime.now().isoformat(),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"[SUMMARIZE] Error: {str(e)}", exc_info=True)
            return {
                'summary_type': summary_type,
                'error': str(e),
                'error_type': type(e).__name__,
                'summary': 'Unable to generate summary. Please try again.',
                'generated_at': datetime.now().isoformat(),
                'status': 'error'
            }

    def suggest_experiment_design(self, problem: str, domain: str) -> Dict[str, Any]:
        """Suggest experiment design using Perplexity AI"""
        try:
            prompt = f"""As an expert research methodologist, suggest a comprehensive experiment design for this research problem.

Problem: {problem}
Domain: {domain}

Provide a detailed experimental design including:
1. Experiment Type (e.g., controlled experiment, A/B test, observational study)
2. Independent and Dependent Variables
3. Evaluation Metrics
4. Baseline Methods
5. Hypothesis Statement
6. Sample Size Recommendations
7. Statistical Tests to Use
8. Potential Confounding Factors
9. Data Collection Strategy

Format as structured sections with clear explanations."""

            logger.info(f"[EXPERIMENT_DESIGN] Generating design for: {problem[:50]}...")
            
            response = perplexity_client.chat.completions.create(
                model=self.perplexity_model,
                messages=[
                    {"role": "system", "content": "You are an expert in experimental design for AI/ML research."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )

            content = response.choices[0].message.content
            logger.info("[EXPERIMENT_DESIGN] Design generated successfully")

            return {
                'problem': problem,
                'domain': domain,
                'experiment_design': content,
                'generated_at': datetime.now().isoformat(),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"[EXPERIMENT_DESIGN] Error: {str(e)}", exc_info=True)
            return {
                'problem': problem,
                'domain': domain,
                'error': str(e),
                'error_type': type(e).__name__,
                'experiment_design': 'Unable to generate experiment design. Please try again.',
                'generated_at': datetime.now().isoformat(),
                'status': 'error'
            }

    def generate_research_insights(self, papers: List[Dict], problem: str) -> Dict[str, Any]:
        """Generate insights from multiple papers using Perplexity"""
        try:
            # Compile paper summaries
            papers_summary = "\n\n".join([
                f"Paper {i+1}: {p['title']}\nAbstract: {p['abstract'][:300]}..."
                for i, p in enumerate(papers[:5])
            ])

            prompt = f"""Based on these recent research papers related to "{problem}", provide:

{papers_summary}

Please analyze and provide:
1. Key Trends and Patterns - What common themes emerge across papers?
2. Common Methodologies - What approaches do these papers use?
3. Research Gaps - What areas need more research?
4. Recommended Next Steps - What should be done next?
5. Potential Novel Approaches - What innovative directions could be explored?"""

            logger.info(f"[INSIGHTS] Analyzing {len(papers)} papers for: {problem[:50]}...")
            
            response = perplexity_client.chat.completions.create(
                model=self.perplexity_model,
                messages=[
                    {"role": "system", "content": "You are an AI research analyst with expertise in synthesizing findings from multiple papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.4
            )

            content = response.choices[0].message.content
            logger.info("[INSIGHTS] Insights generated successfully")

            return {
                'problem': problem,
                'papers_analyzed': len(papers),
                'insights': content,
                'generated_at': datetime.now().isoformat(),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"[INSIGHTS] Error: {str(e)}", exc_info=True)
            return {
                'problem': problem,
                'papers_analyzed': len(papers),
                'error': str(e),
                'error_type': type(e).__name__,
                'insights': 'Unable to generate insights. Please try again.',
                'generated_at': datetime.now().isoformat(),
                'status': 'error'
            }


# Initialize assistant
assistant = ResearchAssistant()


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home endpoint - API documentation"""
    return jsonify({
        'service': 'AI Research Assistant API',
        'version': '2.0.0 (FIXED)',
        'status': 'active',
        'endpoints': {
            'home': '/ (GET)',
            'health': '/health (GET)',
            'test_perplexity': '/api/test-perplexity (GET)',
            'parse_problem': '/api/parse-problem (POST)',
            'search_papers': '/api/search-papers (POST)',
            'summarize_paper': '/api/summarize-paper (POST)',
            'suggest_experiments': '/api/suggest-experiments (POST)',
            'generate_insights': '/api/generate-insights (POST)'
        },
        'powered_by': 'Perplexity AI + arXiv API',
        'fixes': [
            'Timeout configured to 120 seconds for API calls',
            'Improved error handling with fallback responses',
            'Better logging for debugging',
            'Proper JSON parsing with fallback'
        ]
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    api_key_status = 'configured' if os.getenv('PERPLEXITY_API_KEY') else 'missing'
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'perplexity_api_key': api_key_status,
        'version': '2.0.0'
    })


@app.route('/api/test-perplexity', methods=['GET'])
def test_perplexity():
    """Test Perplexity API connection"""
    try:
        logger.info("[TEST_PERPLEXITY] Testing connection...")
        response = perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            messages=[
                {"role": "user", "content": "Say 'API is working' briefly."}
            ],
            max_tokens=10
        )
        logger.info("[TEST_PERPLEXITY] Connection successful")
        return jsonify({
            'status': 'success',
            'message': 'Perplexity API is working',
            'response': response.choices[0].message.content,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[TEST_PERPLEXITY] Connection failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'failed',
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Failed to connect to Perplexity API'
        }), 500


@app.route('/api/parse-problem', methods=['POST'])
def parse_problem():
    """Parse research problem statement"""
    try:
        data = request.get_json()
        problem = data.get('problem_statement', '').strip()

        if not problem:
            logger.warning("[PARSE_PROBLEM] Empty problem statement received")
            return jsonify({'error': 'problem_statement is required'}), 400

        logger.info(f"[PARSE_PROBLEM] Received request for: {problem[:50]}...")
        result = assistant.parse_problem_statement(problem)
        logger.info(f"[PARSE_PROBLEM] Status: {result.get('status', 'unknown')}")
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"[PARSE_PROBLEM] Endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Failed to parse problem statement'
        }), 500


@app.route('/api/search-papers', methods=['POST'])
def search_papers():
    """Search arXiv papers"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        max_results = min(int(data.get('max_results', 10)), 20)

        if not query:
            logger.warning("[SEARCH_PAPERS] Empty query received")
            return jsonify({'error': 'query is required'}), 400

        logger.info(f"[SEARCH_PAPERS] Query: {query}, Max results: {max_results}")
        papers = assistant.search_arxiv_papers(query, max_results)

        return jsonify({
            'query': query,
            'total_results': len(papers),
            'papers': papers,
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"[SEARCH_PAPERS] Error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Failed to search papers'
        }), 500


@app.route('/api/summarize-paper', methods=['POST'])
def summarize_paper():
    """Summarize a research paper"""
    try:
        data = request.get_json()
        paper_text = data.get('paper_text', '').strip()
        summary_type = data.get('summary_type', 'general')

        if not paper_text:
            logger.warning("[SUMMARIZE_PAPER] Empty paper text received")
            return jsonify({'error': 'paper_text is required'}), 400

        logger.info(f"[SUMMARIZE_PAPER] Type: {summary_type}, Text length: {len(paper_text)}")
        result = assistant.summarize_paper(paper_text, summary_type)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"[SUMMARIZE_PAPER] Error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Failed to summarize paper'
        }), 500


@app.route('/api/suggest-experiments', methods=['POST'])
def suggest_experiments():
    """Suggest experiment design"""
    try:
        data = request.get_json()
        problem = data.get('problem_statement', '').strip()
        domain = data.get('domain', 'machine_learning')

        if not problem:
            logger.warning("[SUGGEST_EXPERIMENTS] Empty problem statement received")
            return jsonify({'error': 'problem_statement is required'}), 400

        logger.info(f"[SUGGEST_EXPERIMENTS] Problem: {problem[:50]}..., Domain: {domain}")
        result = assistant.suggest_experiment_design(problem, domain)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"[SUGGEST_EXPERIMENTS] Error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Failed to suggest experiment design'
        }), 500


@app.route('/api/generate-insights', methods=['POST'])
def generate_insights():
    """Generate research insights from papers"""
    try:
        data = request.get_json()
        papers = data.get('papers', [])
        problem = data.get('problem_statement', '').strip()

        if not papers or not problem:
            logger.warning("[GENERATE_INSIGHTS] Missing papers or problem statement")
            return jsonify({'error': 'papers and problem_statement are required'}), 400

        logger.info(f"[GENERATE_INSIGHTS] Papers: {len(papers)}, Problem: {problem[:50]}...")
        result = assistant.generate_research_insights(papers, problem)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"[GENERATE_INSIGHTS] Error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Failed to generate insights'
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({'error': 'Method not allowed'}), 405


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Check for API key
    if not os.getenv('PERPLEXITY_API_KEY'):
        logger.warning("⚠️  PERPLEXITY_API_KEY not found in environment variables!")
        logger.warning("Please set it in .env file or as environment variable")
        logger.warning("Continuing without API key - API calls will fail")
    else:
        logger.info("✅ PERPLEXITY_API_KEY configured")
        logger.info(f"✅ API Key length: {len(os.getenv('PERPLEXITY_API_KEY'))} characters")

    logger.info("=" * 80)
    logger.info("🚀 Starting AI Research Assistant Backend (v2.0.0 - FIXED)")
    logger.info("=" * 80)
    logger.info("Timeout Configuration:")
    logger.info("  - Connect timeout: 10s")
    logger.info("  - Read timeout: 120s (2 minutes)")
    logger.info("  - Write timeout: 10s")
    logger.info("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
