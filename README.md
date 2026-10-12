# 🔬 ResearchMate AI
### Intelligent Research Paper Analysis and Classification System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Problem Statement

The rapid growth of scientific literature makes it increasingly difficult for researchers, students, and academics to efficiently explore, classify, and extract insights from large volumes of research papers. Manual reading and analysis of hundreds of papers is time-consuming and error-prone.

**ResearchMate AI** addresses this challenge by providing an intelligent, automated platform for research paper analysis using Natural Language Processing (NLP) techniques.

---

## 🎯 Objectives

- Automate the exploration and preprocessing of large scientific paper datasets
- Apply core NLP techniques to extract meaningful insights from research papers
- Provide an interactive, beginner-friendly dashboard for non-technical users
- Demonstrate end-to-end NLP pipeline from raw data to actionable intelligence

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📊 **Data Collection & EDA** | Flexible CSV loader with automated statistical analysis and visualizations |
| ⚙️ **ETL Pipeline** | Data cleaning, missing value handling, and SQLite database loading |
| 📝 **Text Summarization** | Extractive summarization using TF-IDF sentence ranking |
| 🏷️ **Text Classification** | Category prediction using TF-IDF + Naive Bayes / Logistic Regression |
| 🔍 **Named Entity Recognition** | Key entity extraction using NLTK and spaCy |
| ❓ **Question Answering** | Extractive QA using TF-IDF similarity to find relevant sentences |
| 🛒 **Market Basket Analysis** | Keyword co-occurrence analysis via boolean matrix on abstracts |
| 🤖 **Research Chatbot** | Conversational AI interface for research paper retrieval |
| 📈 **Evaluation Dashboard** | Maps implementation to evaluation rubrics with ROUGE scores |

---

## 🛠️ Technologies Used

- **Frontend & App**: [Streamlit](https://streamlit.io/) 1.38
- **Language**: Python 3.10+
- **NLP Libraries**: NLTK 3.8, spaCy 3.7
- **ML Framework**: scikit-learn 1.5
- **Data Processing**: Pandas 2.2, NumPy 1.26
- **Visualization**: Matplotlib 3.9, Seaborn 0.13, WordCloud
- **Database**: SQLite (via Python built-in `sqlite3`)
- **Evaluation**: ROUGE Score

---

## 🔄 Project Workflow

```
Raw arXiv CSV Dataset
        │
        ▼
┌─────────────────────┐
│  1. Data Collection │  ← Load CSV, detect columns
│     & EDA           │  ← Shape, missing values, category counts
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. ETL Pipeline    │  ← Clean, normalize, load to SQLite
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Preprocessing   │  ← Tokenize, remove stopwords, lemmatize
│     + TF-IDF        │  ← Feature extraction
└────────┬────────────┘
         │
    ┌────┴─────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Summarization│  │Classification│  │     NER      │  │     QA       │
│  (Concept 1) │  │  (Concept 2) │  │  (Concept 3) │  │  (Concept 4) │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
    ▼                                              ▼
┌──────────────┐                       ┌──────────────────┐
│ Market Basket│                       │     Chatbot      │
│  (Concept 5) │                       │   (Concept 6)    │
└──────────────┘                       └──────────────────┘
         │                                        │
         └────────────────┬───────────────────────┘
                          ▼
               ┌─────────────────────┐
               │  Evaluation Dashboard│
               └─────────────────────┘
```

---

## 📁 Project Structure

```
ResearchMate_AI/
├── app.py                      # 🏠 Main Streamlit application (all pages)
├── requirements.txt            # 📦 Python dependencies
├── README.md                   # 📖 Project documentation
├── .gitignore                  # 🚫 Files excluded from Git
│
├── .streamlit/
│   └── config.toml             # 🎨 Streamlit theme configuration
│
├── assets/
│   ├── icon.jpg                # App icon
│   └── logo.jpg                # App logo
│
├── data/
│   ├── sample_arxiv_data.csv   # 📊 Built-in sample dataset (300 papers)
│   └── processed_arxiv_data.csv # Pre-processed data
│
└── utils/
    ├── __init__.py
    ├── data_loader.py          # Step 1: Data collection & loading
    ├── eda.py                  # Step 2: Exploratory Data Analysis
    ├── etl.py                  # Step 3: ETL pipeline
    ├── preprocessing.py        # Step 4: Text preprocessing + TF-IDF
    ├── tfidf_extraction.py     # TF-IDF keyword extraction
    ├── summarization.py        # Concept 1: Extractive summarization
    ├── classification.py       # Concept 2: Text classification
    ├── ner.py                  # Concept 3: Named entity recognition
    ├── qa.py                   # Concept 4: Question answering
    ├── market_basket.py        # Concept 5: Keyword co-occurrence
    └── chatbot.py              # Concept 6: Research chatbot
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step-by-Step Installation

**1. Clone the repository**
```bash
git clone https://github.com/janashree0512/ResearchMate-AI.git
cd ResearchMate-AI
```

**2. Create a virtual environment** (recommended)
```bash
python -m venv venv
```

Activate it:
- **Windows**: `venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download spaCy language model**
```bash
python -m spacy download en_core_web_sm
```

**5. Run the application**
```bash
streamlit run app.py
```

Your browser will automatically open at **http://localhost:8501** 🎉

> **Note:** On first run, the app automatically downloads NLTK language resources (stopwords, tokenizer, etc.). This requires an internet connection the first time only.

---

## 📂 Using Your Own Dataset

The app ships with a built-in **sample dataset** (`data/sample_arxiv_data.csv`, 300 papers) so you can try it immediately.

To use your own **arXiv Scientific Research Paper Dataset**:
1. In the app sidebar, uncheck **"Use built-in sample dataset"**
2. Click **"Upload your arXiv CSV"** and select your file

The loader automatically detects common column name variants:
- Title: `title`
- Abstract: `abstract`, `summary`, `text`
- Category: `category`, `categories`, `primary_category`

---

## 🖼️ Screenshots

> *Screenshots will be added here*

| Dashboard | EDA | Text Classification |
|-----------|-----|---------------------|
| ![Dashboard](assets/screenshots/dashboard.png) | ![EDA](assets/screenshots/eda.png) | ![Classification](assets/screenshots/classification.png) |

| NER | Chatbot | Evaluation |
|-----|---------|------------|
| ![NER](assets/screenshots/ner.png) | ![Chatbot](assets/screenshots/chatbot.png) | ![Evaluation](assets/screenshots/evaluation.png) |

---

## 🔧 Troubleshooting

**NLTK download errors**
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

**"Sample dataset not found" error**
Make sure you run `streamlit run app.py` from inside the `ResearchMate_AI` project folder.

**spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

---

## 🔮 Future Enhancements

- [ ] Integration with live arXiv API for real-time paper fetching
- [ ] Transformer-based summarization (BERT, T5)
- [ ] Semantic search using sentence embeddings
- [ ] Citation network visualization
- [ ] PDF upload and direct text extraction
- [ ] Multi-language support
- [ ] User authentication and saved search history
- [ ] Export analysis results as PDF report

---

## 👩‍💻 Author

**Janashree** — [@janashree0512](https://github.com/janashree0512)

---

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ using Python & Streamlit*
