import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix Sidebar Navigation
old_nav_pattern = r'page = st\.sidebar\.radio\(\s*"Go to:",\s*\[(.*?)\]\s*,\s*\)'
new_nav = """page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Collection & EDA",
        "ETL Process",
        "Text Preprocessing & Feature Extraction",
        "Text Summarization",
        "Text Classification",
        "Named Entity Recognition",
        "Question Answering",
        "Market Basket Analysis",
        "Research Chatbot",
        "Evaluation & Results",
    ],
)"""
content = re.sub(old_nav_pattern, new_nav, content, flags=re.DOTALL)

# 2. Fix Emojis in Sidebar and Home
content = content.replace('st.sidebar.title("📄 ResearchMate AI")', 'st.sidebar.title("ResearchMate AI")')
content = content.replace('st.sidebar.subheader("1. Dataset")', 'st.sidebar.markdown("**Dataset Selection**")')
content = content.replace('st.sidebar.markdown("### Navigation")', '')
content = content.replace('st.sidebar.subheader("2. Navigate")', '')
content = content.replace('st.title("📄 ResearchMate AI")', 'st.title("ResearchMate AI")')
content = content.replace('col1.metric("📚 Papers"', 'col1.metric("Papers"')
content = content.replace('col2.metric("📑 Columns"', 'col2.metric("Columns"')
content = content.replace('col3.metric("🏷️ Categories"', 'col3.metric("Categories"')
content = content.replace('col4.metric("📝 Avg Words"', 'col4.metric("Avg Words"')
content = content.replace('st.subheader("📊 Dataset Health")', 'st.subheader("Dataset Health")')
content = content.replace('st.subheader("📋 Dataset Preview")', 'st.subheader("Dataset Preview")')
content = content.replace('st.subheader("🔬 NLP Pipeline")', 'st.subheader("Core Pipeline Steps")')
content = content.replace('st.success("1️⃣ Text Preprocessing")', 'st.info("Text Preprocessing")')
content = content.replace('st.success("2️⃣ TF-IDF")', 'st.info("TF-IDF Vectorization")')
content = content.replace('st.success("3️⃣ Summarization")', 'st.info("Summarization Models")')
content = content.replace('page == "🏠 Home"', 'page == "Home"')

# 3. Data Collection & EDA
content = content.replace('elif page == "📥 Data Collection & EDA (10 Marks)":', 'elif page == "Data Collection & EDA":')
content = content.replace('st.title("📥 Step 1: Data Collection and Data Loading")', 'st.title("Data Collection & Data Loading")\n    st.caption("Evaluation: 10 Marks Component")')
content = content.replace('st.download_button(\n        "⬇️ Download standardized dataset as CSV"', 'st.download_button(\n        "Download Dataset (CSV)"')

# EDA Sub-page check - EDA was mapped earlier to Data loading if it was changed, wait, let's look at how it was.
content = content.replace('elif page == "📊 EDA (Exploratory Data Analysis)":', 'if False: # Deprecated')
content = content.replace('st.title("📊 Step 2: Exploratory Data Analysis")', 'st.header("Exploratory Data Analysis")')
content = content.replace('st.expander(\n        "ℹ️ What is ETL, and how is it different from EDA?",', 'st.expander(\n        "Understanding ETL vs EDA",')

# 4. ETL Process
content = content.replace('elif page == "🔄 ETL Process (10 Marks)":', 'elif page == "ETL Process":')
content = content.replace('st.title("🔄 Step 3: ETL")', 'st.title("ETL Process")\n    st.caption("Evaluation: 10 Marks Component")')
content = content.replace('if st.button("▶️ Run Load Step"):', 'if st.button("Execute Load Step"):')

# 5. Preprocessing
content = content.replace('elif page == "🧹 Preprocessing & TF-IDF":', 'elif page == "Text Preprocessing & Feature Extraction":')
content = content.replace('st.title("🧹 NLP Concept 1: Text Preprocessing")', 'st.title("Text Preprocessing & Feature Extraction")\n    st.caption("Prerequisite Component")')
content = content.replace('st.subheader("1️⃣ Lowercase")', 'st.subheader("1. Lowercase")')
content = content.replace('st.subheader("2️⃣ No Punctuation")', 'st.subheader("2. Punctuation Removal")')
content = content.replace('st.subheader("3️⃣ Tokens")', 'st.subheader("3. Tokenization")')
content = content.replace('st.subheader("4️⃣ Lemmatized")', 'st.subheader("4. Lemmatization")')
content = content.replace('st.title("⭐ NLP Concept 2: TF-IDF Feature Extraction")', 'st.header("TF-IDF Feature Extraction")')

# 6. Concept 1: Summarization
content = content.replace('elif page == "📝 Concept 1: Text Summarization (Phase 1)":', 'elif page == "Text Summarization":')
content = content.replace('st.title(\n        "📝 NLP Concept 3: Extractive Text Summarization"\n    )', 'st.title("Extractive Text Summarization")\n    st.caption("Phase I - Concept 1")')
content = content.replace('st.title(\n        "📝 NLP Concept 1: Extractive Text Summarization"\n    )', 'st.title("Extractive Text Summarization")\n    st.caption("Phase I - Concept 1")') # Catch variations
content = content.replace('st.title("📝 NLP Concept 1: Extractive Text Summarization")', 'st.title("Extractive Text Summarization")\n    st.caption("Phase I - Concept 1")')
content = content.replace('st.subheader(\n        "🔁 Batch Summarization Preview"\n    )', 'st.subheader("Batch Summarization Preview")')
content = content.replace('st.subheader("🔁 Batch Summarization Preview")', 'st.subheader("Batch Summarization Preview")')

# 7. Concept 2: Classification
content = content.replace('elif page == "🏷️ Concept 2: Text Classification (Phase 1)":', 'elif page == "Text Classification":')
content = content.replace('st.title("🏷️ NLP Concept 4: Text Classification")', 'st.title("Text Classification")\n    st.caption("Phase I - Concept 2")')
content = content.replace('st.title("🏷️ NLP Concept 2: Text Classification")', 'st.title("Text Classification")\n    st.caption("Phase I - Concept 2")')

# 8. Concept 3: NER
content = content.replace('elif page == "🔍 Concept 3: Named Entity (Phase 2)":', 'elif page == "Named Entity Recognition":')
content = content.replace('st.title("🔍 NLP Concept 5: Named Entity Recognition (NER)")', 'st.title("Named Entity Recognition (NER)")\n    st.caption("Phase II - Concept 3")')
content = content.replace('st.title("🔍 NLP Concept 3: Named Entity Recognition (NER)")', 'st.title("Named Entity Recognition (NER)")\n    st.caption("Phase II - Concept 3")')

# 9. Concept 4: QA
content = content.replace('elif page == "❓ Concept 4: QA (Phase 2)":', 'elif page == "Question Answering":')
content = content.replace('st.title("❓ NLP Concept 6: Question Answering (QA)")', 'st.title("Question Answering (QA)")\n    st.caption("Phase II - Concept 4")')
content = content.replace('st.title("❓ NLP Concept 4: Question Answering (QA)")', 'st.title("Question Answering (QA)")\n    st.caption("Phase II - Concept 4")')

# 10. Concept 5: Market Basket
content = content.replace('elif page == "🛒 Concept 5: Market Basket Analysis":', 'elif page == "Market Basket Analysis":')
content = content.replace('st.title("🛒 NLP Concept 5: Market Basket Analysis")', 'st.title("Market Basket Analysis")\n    st.caption("Final Phase - Concept 5")')

# 11. Concept 6: Chatbot
content = content.replace('elif page == "🤖 Concept 6: Chatbot":', 'elif page == "Research Chatbot":')
content = content.replace('st.title("🤖 NLP Concept 6: Research Assistant Chatbot")', 'st.title("Research Assistant Chatbot")\n    st.caption("Final Phase - Concept 6")')

# 12. Evaluation
content = content.replace('elif page == "🎓 Evaluation Dashboard":', 'elif page == "Evaluation & Results":')
content = content.replace('st.title("🎓 Evaluation Dashboard")', 'st.title("Evaluation & Results")')


with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
