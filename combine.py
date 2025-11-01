"""
Combined Streamlit + Flask app
File: streamlit_flask_combined_app.py

How it works:
- This single Python file runs a Flask API in a background thread and a Streamlit UI in the main thread.
- Streamlit talks to the local Flask API at http://localhost:5000/api
- When launching locally use:
      streamlit run streamlit_flask_combined_app.py

Important notes before deploying:
- Add a .env file or set environment variable PERPLEXITY_API_KEY with your Perplexity/OpenAI API key.
- For cloud deployment (Deta Space / Streamlit Cloud): make sure environment variables are configured in the hosting platform.

Requirements (add to requirements.txt in your repo):
- streamlit
- flask
- flask-cors
- python-dotenv
- openai
- httpx
- arxiv
- requests

"""
#------ BEGIN COMBINED CODE ------

# Standard library
import os
import json
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any

# Flask backend imports
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import arxiv

# Streamlit frontend imports
import streamlit as st
import requests

# -----------------------------------------------------------------------------
# Load environment and configure logging
# -----------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Flask API (adapted from your provided backend)
# -----------------------------------------------------------------------------
flask_app = Flask(__name__)
CORS(flask_app)

perplexity_client = OpenAI(
    api_key=os.getenv('PERPLEXITY_API_KEY'),
    base_url="https://api.perplexity.ai",
    perplexity_client = OpenAI(
    api_key=st.secrets["PERPLEXITY_API_KEY"],
    base_url="https://api.perplexity.ai",
    timeout=httpx.Timeout(60.0),
    max_retries=2
)


class ResearchAssistant:
    """Main class for AI Research Assistant functionality"""

    def __init__(self):
        self.perplexity_model = "sonar-pro"

    def parse_problem_statement(self, problem: str) -> Dict[str, Any]:
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

            try:
                parsed_data = json.loads(content)
                logger.info("[PARSE_PROBLEM] JSON parsing successful")
            except json.JSONDecodeError as je:
                logger.warning(f"[PARSE_PROBLEM] JSON parsing failed: {je}. Using fallback.")
                parsed_data = {
                    "keywords": self._extract_keywords(problem),
                    "domains": self._identify_domains(problem),
                    "clarifying_questions": [
                        "What is the specific application domain?",
                        "What are the evaluation metrics?",
                        "What datasets will be used?"
                    ],
                    "suggestions": [content]
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
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'can', 'we', 'how', 'what', 'is', 'this', 'that'}
        words = text.lower().split()
        keywords = [w.strip('.,?!;:') for w in words if w not in stopwords and len(w) > 3]
        return list(set(keywords))[:8]

    def _identify_domains(self, text: str) -> List[str]:
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
        try:
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


@flask_app.route('/')
def home():
    return jsonify({
        'service': 'AI Research Assistant API',
        'version': '2.0.0 (COMBINED)',
        'status': 'active'
    })


@flask_app.route('/health')
def health():
    api_key_status = 'configured' if os.getenv('PERPLEXITY_API_KEY') else 'missing'
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'perplexity_api_key': api_key_status,
        'version': '2.0.0'
    })


@flask_app.route('/api/parse-problem', methods=['POST'])
def parse_problem():
    try:
        data = request.get_json()
        problem = data.get('problem_statement', '').strip()
        if not problem:
            return jsonify({'error': 'problem_statement is required'}), 400
        result = assistant.parse_problem_statement(problem)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[PARSE_PROBLEM] Endpoint error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'message': 'Failed to parse problem statement'}), 500


@flask_app.route('/api/search-papers', methods=['POST'])
def search_papers():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        max_results = min(int(data.get('max_results', 10)), 20)
        if not query:
            return jsonify({'error': 'query is required'}), 400
        papers = assistant.search_arxiv_papers(query, max_results)
        return jsonify({'query': query, 'total_results': len(papers), 'papers': papers, 'status': 'success'})
    except Exception as e:
        logger.error(f"[SEARCH_PAPERS] Error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'message': 'Failed to search papers'}), 500


@flask_app.route('/api/summarize-paper', methods=['POST'])
def summarize_paper():
    try:
        data = request.get_json()
        paper_text = data.get('paper_text', '').strip()
        summary_type = data.get('summary_type', 'general')
        if not paper_text:
            return jsonify({'error': 'paper_text is required'}), 400
        result = assistant.summarize_paper(paper_text, summary_type)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[SUMMARIZE_PAPER] Error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'message': 'Failed to summarize paper'}), 500


@flask_app.route('/api/suggest-experiments', methods=['POST'])
def suggest_experiments():
    try:
        data = request.get_json()
        problem = data.get('problem_statement', '').strip()
        domain = data.get('domain', 'machine_learning')
        if not problem:
            return jsonify({'error': 'problem_statement is required'}), 400
        result = assistant.suggest_experiment_design(problem, domain)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[SUGGEST_EXPERIMENTS] Error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'message': 'Failed to suggest experiment design'}), 500


@flask_app.route('/api/generate-insights', methods=['POST'])
def generate_insights():
    try:
        data = request.get_json()
        papers = data.get('papers', [])
        problem = data.get('problem_statement', '').strip()
        if not papers or not problem:
            return jsonify({'error': 'papers and problem_statement are required'}), 400
        result = assistant.generate_research_insights(papers, problem)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[GENERATE_INSIGHTS] Error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'message': 'Failed to generate insights'}), 500


# Error handlers
@flask_app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@flask_app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


@flask_app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405


# Function to run Flask in a background thread
def run_flask():
    if not os.getenv('PERPLEXITY_API_KEY'):
        logger.warning("⚠️  PERPLEXITY_API_KEY not found in environment variables! Flask endpoints that call Perplexity will fail without it.")
    logger.info("Starting Flask API on http://0.0.0.0:5000")
    flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


# -----------------------------------------------------------------------------
# Streamlit UI (adapted from your provided Streamlit file)
# -----------------------------------------------------------------------------

# Start Flask server in background when Streamlit app starts
if 'flask_thread_started' not in st.session_state:
    thread = threading.Thread(target=run_flask, daemon=True)
    thread.start()
    st.session_state['flask_thread_started'] = True

st.set_page_config(
    page_title="Agentic AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:5000/api"

TIMEOUTS = {
    "search_papers": 60,
    "parse_problem": 120,
    "suggest_experiments": 120,
    "summarize_paper": 90,
    "generate_insights": 180
}

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

if 'papers' not in st.session_state:
    st.session_state['papers'] = []
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ""
if 'paper_to_summarize' not in st.session_state:
    st.session_state['paper_to_summarize'] = ""

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.title("🤖 Navigation")

    page = st.radio(
        "Select Feature",
        ["🏠 Home", "🔍 Problem Analysis", "📚 Paper Search", "📝 Paper Summary", 
         "🧪 Experiment Design", "💡 Research Insights"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ℹ️ About This App")
    st.info(
        "This AI Research Assistant helps you understand research problems, "
        "search academic papers, generate summaries, and design experiments using "
        "Perplexity AI and arXiv."
    )

    st.markdown("### 🔌 Backend Status")
    if st.button("Test Connection", key="health_check"):
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.ok:
                data = response.json()
                st.success("✅ Backend is running")
                st.caption(f"Status: {data.get('status', 'unknown')}")
            else:
                st.error("❌ Backend returned error")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend")
            st.code("python streamlit_flask_combined_app.py", language="bash")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# The rest of the Streamlit UI follows the same structure as your original file
# For readability we include it verbatim but adapted to use API_BASE_URL variable

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 Agentic AI Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    ## Welcome to Your AI-Powered Research Companion

    This intelligent assistant helps AI researchers and data scientists by:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔍 Understand Problems
        - Parse research statements
        - Extract key concepts
        - Identify domains
        - Generate clarifying questions
        """)

    with col2:
        st.markdown("""
        ### 📚 Search & Analyze
        - Search arXiv papers
        - Get relevant literature
        - Summarize findings
        - Extract insights
        """)

    with col3:
        st.markdown("""
        ### 🧪 Design Experiments
        - Suggest methodologies
        - Define variables
        - Recommend metrics
        - Statistical analysis
        """)

    st.markdown("---")
    st.markdown("## 🚀 Quick Start Example")
    st.code('''
# Example Research Problem:
"How can we improve model interpretability in computer vision?"

# The assistant will:
1. Parse and analyze the problem
2. Search relevant papers from arXiv
3. Summarize key findings
4. Suggest experiment designs
5. Generate actionable insights
    ''', language="text")

    st.markdown("---")
    st.markdown("## 📊 Features & Timeouts")
    feature_info = {
        "Problem Analysis": "Analyzes your research question (60-90s)",
        "Paper Search": "Searches arXiv (15-30s)",
        "Paper Summary": "AI-powered paper summarization (45-75s)",
        "Experiment Design": "Suggests research methodology (60-120s)",
        "Research Insights": "Analyzes multiple papers (90-180s)"
    }
    for feature, description in feature_info.items():
        st.markdown(f"**{feature}**: {description}")

    st.markdown("---")
    st.markdown("## ✨ Key Improvements in v2.0")
    improvements = [
        "✅ Increased API timeouts to 120 seconds",
        "✅ Better error handling with fallbacks",
        "✅ Improved logging for debugging",
        "✅ Consistent response format",
        "✅ Clear loading indicators"
    ]
    for improvement in improvements:
        st.markdown(improvement)

elif page == "🔍 Problem Analysis":
    st.markdown('<h2 class="section-header">🔍 Research Problem Analysis</h2>', unsafe_allow_html=True)
    problem = st.text_area("Enter your research problem statement:", height=150)

    if st.button("Analyze Problem"):
        if problem.strip():
            with st.spinner("Analyzing your research problem..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/parse-problem",
                        json={"problem_statement": problem},
                        timeout=TIMEOUTS["parse_problem"]
                    )
                    if response.ok:
                        data = response.json()
                        st.success("✅ Problem analyzed successfully!")
                        st.json(data["parsed_data"])
                    else:
                        st.error(f"API error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a research problem.")




elif page == "📚 Paper Search":
    st.markdown('<h2 class="section-header">📚 Search Academic Papers (arXiv)</h2>', unsafe_allow_html=True)
    query = st.text_input("Enter your research topic or keywords:", st.session_state.get('search_query', ''))
    max_results = st.slider("Number of papers", 5, 20, 10)

    if st.button("Search Papers"):
        if query.strip():
            st.session_state['search_query'] = query
            with st.spinner("Searching papers on arXiv..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/search-papers",
                        json={"query": query, "max_results": max_results},
                        timeout=TIMEOUTS["search_papers"]
                    )
                    if response.ok:
                        data = response.json()
                        st.session_state['papers'] = data["papers"]
                        st.success(f"✅ Found {data['total_results']} papers.")
                    else:
                        st.error(f"API error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Enter a query to search papers.")

    if st.session_state['papers']:
        st.markdown("### Search Results:")
        for i, paper in enumerate(st.session_state['papers'], start=1):
            st.markdown(f"**{i}. [{paper['title']}]({paper['pdf_url']})**")
            st.caption(f"🧑‍💻 Authors: {', '.join(paper['authors'])}")
            st.write(paper['abstract'])
            st.divider()


elif page == "📝 Paper Summary":
    st.markdown('<h2 class="section-header">📝 Paper Summarization</h2>', unsafe_allow_html=True)
    st.info("Paste the paper abstract or key section below to summarize it.")

    text_input = st.text_area("Paste paper content here:", height=250)
    summary_type = st.selectbox("Choose summary type:", ["general", "methodology", "results", "conclusions"])

    if st.button("Generate Summary"):
        if text_input.strip():
            with st.spinner("Generating paper summary..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/summarize-paper",
                        json={"paper_text": text_input, "summary_type": summary_type},
                        timeout=TIMEOUTS["summarize_paper"]
                    )
                    if response.ok:
                        data = response.json()
                        st.success("✅ Summary generated successfully!")
                        st.markdown("### Summary:")
                        st.write(data["summary"])
                    else:
                        st.error(f"API error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please paste the paper content.")


elif page == "🧪 Experiment Design":
    st.markdown('<h2 class="section-header">🧪 Experiment Design Assistant</h2>', unsafe_allow_html=True)
    problem = st.text_area("Enter your research problem statement:", height=150)
    domain = st.text_input("Enter the research domain (e.g., computer vision, NLP):", "machine_learning")

    if st.button("Suggest Experiment Design"):
        if problem.strip():
            with st.spinner("Generating experiment design..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/suggest-experiments",
                        json={"problem_statement": problem, "domain": domain},
                        timeout=TIMEOUTS["suggest_experiments"]
                    )
                    if response.ok:
                        data = response.json()
                        st.success("✅ Experiment design generated successfully!")
                        st.write(data["experiment_design"])
                    else:
                        st.error(f"API error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter your problem statement.")


elif page == "💡 Research Insights":
    st.markdown('<h2 class="section-header">💡 Research Insights Generator</h2>', unsafe_allow_html=True)

    problem = st.text_input("Enter your research problem statement:")
    st.info("Make sure you’ve already searched papers in the '📚 Paper Search' section.")

    if st.button("Generate Insights"):
        if st.session_state['papers'] and problem.strip():
            with st.spinner("Generating insights from analyzed papers..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/generate-insights",
                        json={"papers": st.session_state['papers'], "problem_statement": problem},
                        timeout=TIMEOUTS["generate_insights"]
                    )
                    if response.ok:
                        data = response.json()
                        st.success("✅ Insights generated successfully!")
                        st.markdown("### Insights:")
                        st.write(data["insights"])
                    else:
                        st.error(f"API error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please search papers first and enter a valid problem statement.")

