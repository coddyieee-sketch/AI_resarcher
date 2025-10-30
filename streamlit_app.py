
import streamlit as st
import requests
import json
from datetime import datetime



st.set_page_config(
    page_title="Agentic AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



API_BASE_URL = "http://localhost:5000/api"

# Timeout configuration (CRITICAL FIX)
TIMEOUTS = {
    "search_papers": 60,           # Paper search is usually faster
    "parse_problem": 120,          # 2 minutes for problem analysis
    "suggest_experiments": 120,    # 2 minutes for experiment design
    "summarize_paper": 90,         # 1.5 minutes for summarization
    "generate_insights": 180       # 3 minutes for insights (multiple papers)
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

    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        "- **Backend**: Flask\n"
        "- **LLM**: Perplexity AI\n"
        "- **Papers**: arXiv API\n"
        "- **Frontend**: Streamlit"
    )

    st.markdown("---")
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
            st.code("python app.py", language="bash")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")



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
    st.code("""
# Example Research Problem:
"How can we improve model interpretability in computer vision?"

# The assistant will:
1. Parse and analyze the problem
2. Search relevant papers from arXiv
3. Summarize key findings
4. Suggest experiment designs
5. Generate actionable insights
    """, language="text")

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

    st.markdown("""
    Enter your research problem statement below. The AI will analyze it and provide:
    - **Keywords**: Important concepts in your problem
    - **Domains**: Relevant research areas
    - **Clarifying Questions**: To refine your problem
    - **Suggestions**: Ways to make it more specific
    
    ⏱️ **Expected time**: 60-90 seconds
    """)

    problem_statement = st.text_area(
        "Research Problem Statement",
        value="How can we improve model interpretability in computer vision?",
        height=150,
        help="Describe your research problem or question"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze Problem", type="primary", use_container_width=True)

    if analyze_btn and problem_statement:
        with st.spinner("🔍 Analyzing problem statement... (this may take 60-90 seconds)"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/parse-problem",
                    json={"problem_statement": problem_statement},
                    timeout=TIMEOUTS["parse_problem"]  # 120 seconds
                )

                if response.ok:
                    result = response.json()

                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ Analysis Complete!")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Display results
                    if 'parsed_data' in result:
                        parsed = result['parsed_data']

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🔑 Keywords")
                            if isinstance(parsed.get('keywords'), list):
                                for kw in parsed['keywords']:
                                    st.markdown(f"- `{kw}`")
                            else:
                                st.write(parsed.get('keywords', 'N/A'))

                            st.markdown("### 🎯 Research Domains")
                            if isinstance(parsed.get('domains'), list):
                                for domain in parsed['domains']:
                                    st.markdown(f"- **{domain}**")
                            else:
                                st.write(parsed.get('domains', 'N/A'))

                        with col2:
                            st.markdown("### ❓ Clarifying Questions")
                            if isinstance(parsed.get('clarifying_questions'), list):
                                for q in parsed['clarifying_questions']:
                                    st.markdown(f"- {q}")
                            else:
                                st.write(parsed.get('clarifying_questions', 'N/A'))

                            st.markdown("### 💡 Suggestions")
                            if isinstance(parsed.get('suggestions'), list):
                                for s in parsed['suggestions']:
                                    st.markdown(f"- {s}")
                            else:
                                st.write(parsed.get('suggestions', 'N/A'))

                    # Show full AI response
                    with st.expander("📄 Full AI Response"):
                        st.markdown(result.get('ai_response', 'No response'))

                else:
                    st.error(f"❌ Error: {response.status_code}")
                    st.write(response.text)

            except requests.exceptions.Timeout:
                st.error("❌ Request timed out after 2 minutes")
                st.markdown("""
                ### Why this happened:
                - Complex query requiring extensive processing
                - High API load
                - Network connectivity issues
                
                ### What to try:
                1. Simplify your problem statement
                2. Remove unnecessary details
                3. Try again in a moment
                4. Check your internet connection
                """)
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Flask backend")
                st.markdown("""
                ### Solution:
                Make sure Flask server is running:
                """)
                st.code("python app.py", language="bash")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.info("Check the Flask server logs for more details.")



elif page == "📚 Paper Search":
    st.markdown('<h2 class="section-header">📚 Academic Paper Search</h2>', unsafe_allow_html=True)

    st.markdown(
        "Search for relevant research papers from **arXiv**, "
        "the world's largest open-access archive of scientific papers."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Search Query",
            value="model interpretability computer vision",
            help="Enter keywords, topics, or specific terms"
        )
    with col2:
        max_results = st.number_input("Max Results", min_value=1, max_value=20, value=5)

    search_btn = st.button("🔍 Search Papers", type="primary")

    if search_btn and search_query:
        with st.spinner(f"🔍 Searching arXiv for: {search_query}... (15-30 seconds)"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/search-papers",
                    json={"query": search_query, "max_results": max_results},
                    timeout=TIMEOUTS["search_papers"]  # 60 seconds
                )

                if response.ok:
                    result = response.json()
                    papers = result.get('papers', [])

                    st.success(f"✅ Found {len(papers)} papers")

                    # Store in session state for insights generation
                    st.session_state['papers'] = papers
                    st.session_state['search_query'] = search_query

                    for i, paper in enumerate(papers):
                        with st.expander(f"📄 {paper['title']}", expanded=(i==0)):
                            st.markdown(f"**Authors**: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                            st.markdown(f"**Published**: {paper['published']} | **Updated**: {paper['updated']}")
                            st.markdown(f"**Categories**: {', '.join(paper['categories'][:3])}")
                            st.markdown("**Abstract**:")
                            st.write(paper['abstract'])

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"[📖 View on arXiv]({paper['url']})")
                            with col2:
                                st.markdown(f"[📥 Download PDF]({paper['pdf_url']})")

                            if st.button(f"📝 Summarize this paper", key=f"sum_{i}"):
                                st.session_state['paper_to_summarize'] = paper['abstract']
                                st.info("Switch to 'Paper Summary' tab to see the summary")

                else:
                    st.error(f"❌ Error: {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out")
                st.warning("Try with fewer results or simpler search terms")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


elif page == "📝 Paper Summary":
    st.markdown('<h2 class="section-header">📝 Research Paper Summarization</h2>', unsafe_allow_html=True)

    st.markdown("Get AI-powered summaries of research papers with customizable focus areas.")

    if 'paper_to_summarize' in st.session_state and st.session_state['paper_to_summarize']:
        default_text = st.session_state['paper_to_summarize']
    else:
        default_text = "Enter paper abstract or full text here..."

    paper_text = st.text_area(
        "Paper Text (Abstract or Full Text)",
        value=default_text,
        height=200,
        help="Paste the paper abstract or sections you want to summarize"
    )

    col1, col2 = st.columns([2, 3])
    with col1:
        summary_type = st.selectbox(
            "Summary Focus",
            ["general", "methodology", "results", "conclusions"],
            help="Choose what aspect to focus on"
        )

    with col2:
        summaries = {
            "general": "Overall summary of the paper",
            "methodology": "Focus on research methods and approach",
            "results": "Highlight key findings and results",
            "conclusions": "Main conclusions and implications"
        }
        st.info(f"**{summary_type.title()} Summary**: {summaries[summary_type]}")

    summarize_btn = st.button("📝 Generate Summary", type="primary")

    if summarize_btn and paper_text and len(paper_text) > 50:
        with st.spinner(f"📝 Generating {summary_type} summary... (45-75 seconds)"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/summarize-paper",
                    json={"paper_text": paper_text, "summary_type": summary_type},
                    timeout=TIMEOUTS["summarize_paper"]  # 90 seconds
                )

                if response.ok:
                    result = response.json()

                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ Summary Generated!")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("### 📄 Summary")
                    st.markdown(result.get('summary', 'No summary available'))

                    st.caption(f"Generated: {result.get('generated_at', 'N/A')}")

                else:
                    st.error(f"❌ Error: {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out after 90 seconds")
                st.warning("Try with a shorter text or simpler query")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")



elif page == "🧪 Experiment Design":
    st.markdown('<h2 class="section-header">🧪 Experiment Design Assistant</h2>', unsafe_allow_html=True)

    st.markdown("""
    Get AI-powered suggestions for designing experiments for your research problem.
    The assistant will provide a comprehensive experimental design including variables, 
    metrics, and statistical tests.
    
    ⏱️ **Expected time**: 60-120 seconds
    """)

    col1, col2 = st.columns([3, 1])
    with col1:
        exp_problem = st.text_area(
            "Research Question / Hypothesis",
            value="Does adding attention visualization improve model interpretability in CNNs?",
            height=100
        )
    with col2:
        domain = st.selectbox(
            "Research Domain",
            ["computer_vision", "nlp", "machine_learning", "data_science", "deep_learning", "reinforcement_learning"],
            index=0
        )

    design_btn = st.button("🧪 Generate Experiment Design", type="primary")

    if design_btn and exp_problem:
        with st.spinner("🧪 Designing experiment... (60-120 seconds)"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/suggest-experiments",
                    json={"problem_statement": exp_problem, "domain": domain},
                    timeout=TIMEOUTS["suggest_experiments"]  # 120 seconds
                )

                if response.ok:
                    result = response.json()

                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ Experiment Design Complete!")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("### 🔬 Experimental Design")
                    st.markdown(result.get('experiment_design', 'No design available'))

                    st.caption(f"Generated: {result.get('generated_at', 'N/A')}")

                    design_text = f"""# Experiment Design

Problem: {exp_problem}
Domain: {domain}

{result.get('experiment_design', '')}

Generated: {result.get('generated_at', 'N/A')}
"""
                    st.download_button(
                        label="📥 Download Design",
                        data=design_text,
                        file_name=f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )

                else:
                    st.error(f"❌ Error: {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out after 2 minutes")
                st.warning("Try simplifying your research question")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")



elif page == "💡 Research Insights":
    st.markdown('<h2 class="section-header">💡 Research Insights Generator</h2>', unsafe_allow_html=True)

    st.markdown("""
    Generate comprehensive insights by analyzing multiple research papers together.
    First, search for papers using the **Paper Search** feature, then come back here to generate insights.
    
    ⏱️ **Expected time**: 90-180 seconds (depending on number of papers)
    """)

    if 'papers' in st.session_state and st.session_state['papers']:
        st.info(f"✅ {len(st.session_state['papers'])} papers loaded from previous search")

        problem_context = st.text_input(
            "Research Context / Problem",
            value=st.session_state.get('search_query', ''),
            help="Provide context for insight generation"
        )

        insights_btn = st.button("💡 Generate Insights", type="primary")

        if insights_btn and problem_context:
            with st.spinner(f"💡 Analyzing {len(st.session_state['papers'])} papers and generating insights... (90-180 seconds)"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/generate-insights",
                        json={
                            "papers": st.session_state['papers'],
                            "problem_statement": problem_context
                        },
                        timeout=TIMEOUTS["generate_insights"]  # 180 seconds
                    )

                    if response.ok:
                        result = response.json()

                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"✅ Analyzed {result.get('papers_analyzed', 0)} papers")
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown("### 💡 Research Insights")
                        st.markdown(result.get('insights', 'No insights available'))

                        insights_text = f"""# Research Insights

Problem: {problem_context}
Papers Analyzed: {result.get('papers_analyzed', 0)}

{result.get('insights', '')}

Generated: {result.get('generated_at', 'N/A')}
"""
                        st.download_button(
                            label="📥 Download Insights",
                            data=insights_text,
                            file_name=f"research_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )

                    else:
                        st.error(f"❌ Error: {response.text}")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out after 3 minutes")
                    st.warning("Analyzing many papers takes time. Try with fewer papers or wait a moment.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    else:
        st.warning("⚠️ No papers loaded. Please use **Paper Search** first to load papers.")
        st.markdown("### How to use:")
        st.markdown("""
        1. Go to **📚 Paper Search**
        2. Search for papers related to your research topic
        3. Come back to this page
        4. Generate insights from the loaded papers
        """)


