# 🛒 Turkish E-Commerce Sentiment & Spam Analyzer

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-FF6B35?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](../../LICENSE)

### 🌍 English Documentation | [🇹🇷 Türkçe Dokümantasyon](../tr/README.md)

**Machine Learning Course Project** - December 2025

</div>

---

## 📑 Table of Contents

1. [About the Project](#-about-the-project)
2. [Project Team](#-project-team)
3. [Key Features](#-key-features)
4. [System Architecture](#-system-architecture)
5. [Technology Stack](#-technology-stack)
6. [Dataset](#-dataset)
7. [Methodology](#-methodology)
8. [Model Performance](#-model-performance)
9. [Installation](#-installation)
10. [Usage](#-usage)
11. [API Reference](#-api-reference)
12. [Project Structure](#-project-structure)
13. [Example Analyses](#-example-analyses)
14. [Limitations](#-limitations)
15. [Future Improvements](#-future-improvements)
16. [Contributing](#-contributing)
17. [License](#-license)
18. [Contact](#-contact)

---

## 📖 About the Project

This project is an AI-powered decision support system developed as a **Machine Learning course project** that analyzes Turkish e-commerce reviews.

### 🎯 Problem Statement

Distinguishing fake (spam/bot) reviews from genuine customer reviews on e-commerce platforms is a critical issue:

- **For consumers**: Wrong decisions due to misleading reviews
- **For sellers**: Unfair competition and reputation damage
- **For platforms**: Trust and user experience issues

### 💡 Solution Approach

Our system performs **two main tasks**:

| Task | Description | Method |
|------|-------------|--------|
| **🚫 Spam Detection** | Detect fake, advertising, or bot reviews | Hybrid (Rule-based + IsolationForest) |
| **💭 Sentiment Analysis** | Determine if reviews are positive/negative/neutral | TF-IDF + Logistic Regression |

---

## 👥 Project Team

<div align="center">

| Photo | Student ID | Name | Role | Responsibilities |
|:-----:|:----------:|:----:|:----:|:-----------------|
| 👤 | **--** | **Mustafa Arda Düşova** | Team Lead & Developer | Project management, coding, integration |
| 👤 | **--** | **Fatih Çoban** | Data Research & Analysis | Dataset research, EDA, visualization |
| 👤 | **--** | **Efe Ata** | Model Selection & Optimization | Model selection, hyperparameter tuning |

</div>

**Advisor**: [Advisor Name]  
**Course**: Machine Learning  
**Semester**: Fall 2024-2025

---

## ⭐ Key Features

### 🔧 Manual Data Processing
Processing pipelines **coded from scratch** for educational purposes:
- Turkish text normalization
- Unicode cleaning
- Turkish stemming with TurkishStemmer

### 🎯 High-Performance Prediction
- **TF-IDF + Logistic Regression** based sentiment analysis
- **85%+** accuracy rate
- Fast and lightweight model

### 🔀 Hybrid Spam Detection
A system combining the strengths of two approaches:

```
┌─────────────────────────────────────────┐
│            HYBRID SPAM DETECTION        │
├─────────────────┬───────────────────────┤
│   Rule-Based    │   IsolationForest     │
│  (60% weight)   │   (40% weight)        │
├─────────────────┼───────────────────────┤
│ • URL detection │ • Anomaly detection   │
│ • Emoji count   │ • TF-IDF features     │
│ • Repeat pattern│ • Statistical         │
│ • Generic phrase│   outlier detection   │
└─────────────────┴───────────────────────┘
```

### 📊 Aspect-Based Analysis
Analyzes different topics in reviews separately:

| Aspect | Keywords |
|--------|----------|
| 📦 Shipping | kargo, teslimat, paket (delivery, package) |
| 💰 Price | fiyat, ucuz, pahalı (price, cheap, expensive) |
| ⭐ Quality | kalite, malzeme, sağlam (quality, material, durable) |
| 📞 Customer Service | destek, iade, iletişim (support, refund, communication) |

### 🖥️ Web Interface
User-friendly demo developed with Gradio:
- Real-time analysis
- Visual results display
- Easy integration

---

## 🏗️ System Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    TURKISH E-COMMERCE REVIEW ANALYSIS SYSTEM                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                        📥 1. DATA COLLECTION                            │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐             │ ║
║  │  │  HuggingFace  │   │    GitHub     │   │  Local CSV    │             │ ║
║  │  │    Dataset    │   │   Download    │   │    Upload     │             │ ║
║  │  │   (Primary)   │   │   (Backup)    │   │ (Alternative) │             │ ║
║  │  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘             │ ║
║  │          └───────────────────┼───────────────────┘                      │ ║
║  │                              ▼                                          │ ║
║  │                    ┌─────────────────┐                                  │ ║
║  │                    │  TRSAv1 Dataset │                                  │ ║
║  │                    │  (~60K+ reviews)│                                  │ ║
║  │                    └────────┬────────┘                                  │ ║
║  └─────────────────────────────┼───────────────────────────────────────────┘ ║
║                                ▼                                             ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                     🔧 2. DATA PREPROCESSING                            │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ ║
║  │  │ Smart    │→ │ Unicode  │→ │ Lower    │→ │ URL/     │→ │ Turkish  │  │ ║
║  │  │ CSV Read │  │ Normalize│  │ Case     │  │ Email    │  │ Stemming │  │ ║
║  │  │          │  │          │  │ Convert  │  │ Clean    │  │          │  │ ║
║  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │ ║
║  │                                                                         │ ║
║  │  Additional Operations:                                                 │ ║
║  │  • Reduce repeating characters ("çooook" → "çook")                     │ ║
║  │  • Auto-detect columns (text, label, score)                            │ ║
║  │  • Encoding detection (UTF-8, Latin1, Windows-1254)                    │ ║
║  │                                                                         │ ║
║  └──────────────────────────────┬──────────────────────────────────────────┘ ║
║                                 │                                            ║
║              ┌──────────────────┴──────────────────┐                        ║
║              ▼                                      ▼                        ║
║  ┌──────────────────────────┐      ┌──────────────────────────────────────┐ ║
║  │  💭 3. SENTIMENT ANALYSIS│      │      🚫 4. SPAM DETECTION            │ ║
║  ├──────────────────────────┤      ├──────────────────────────────────────┤ ║
║  │                          │      │                                      │ ║
║  │  ┌────────────────────┐  │      │  ┌────────────────────────────────┐  │ ║
║  │  │ TF-IDF Vectorizer  │  │      │  │      RULE-BASED (60%)          │  │ ║
║  │  │ • 1-2 grams        │  │      │  ├────────────────────────────────┤  │ ║
║  │  │ • Min DF: 2        │  │      │  │ • URL/email/phone detection    │  │ ║
║  │  │ • Max DF: 0.9      │  │      │  │ • Exclamation count (>=4→spam) │  │ ║
║  │  └─────────┬──────────┘  │      │  │ • Emoji count (>=3 → spam)     │  │ ║
║  │            ▼             │      │  │ • Uppercase ratio (>0.6)       │  │ ║
║  │  ┌────────────────────┐  │      │  │ • Generic phrase check         │  │ ║
║  │  │ Logistic Regress.  │  │      │  │ • Short + generic detection    │  │ ║
║  │  │ • Max iter: 1000   │  │      │  └─────────────┬──────────────────┘  │ ║
║  │  │ • 3 classes        │  │      │                ▼                     │ ║
║  │  └─────────┬──────────┘  │      │  ┌────────────────────────────────┐  │ ║
║  │            ▼             │      │  │   ISOLATION FOREST (40%)       │  │ ║
║  │  ┌────────────────────┐  │      │  ├────────────────────────────────┤  │ ║
║  │  │ Output:            │  │      │  │ • 100 estimators               │  │ ║
║  │  │ 0: Negative 😞     │  │      │  │ • 5% contamination             │  │ ║
║  │  │ 1: Neutral 😐      │  │      │  │ • TF-IDF based features        │  │ ║
║  │  │ 2: Positive 😊     │  │      │  │ • Anomaly score calculation    │  │ ║
║  │  └────────────────────┘  │      │  └─────────────┬──────────────────┘  │ ║
║  │                          │      │                ▼                     │ ║
║  │                          │      │  ┌────────────────────────────────┐  │ ║
║  │                          │      │  │  HYBRID DECISION               │  │ ║
║  │                          │      │  │  • 0: Genuine Review           │  │ ║
║  │                          │      │  │  • 1: Spam/Bot                 │  │ ║
║  │                          │      │  │  • -1: Uncertain               │  │ ║
║  │                          │      │  └────────────────────────────────┘  │ ║
║  └──────────┬───────────────┘      └───────────────────┬──────────────────┘ ║
║             │                                          │                     ║
║             └─────────────────┬─────────────────────────┘                    ║
║                               ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                      🖥️ 5. APPLICATION LAYER                           │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │ ║
║  │  │  Gradio Web  │  │   Aspect     │  │   BERT       │  │  REST API   │  │ ║
║  │  │  Interface   │  │   Analysis   │  │ Fine-tuning  │  │  (Future)   │  │ ║
║  │  │   (Demo)     │  │   (Detail)   │  │ (Optional)   │  │             │  │ ║
║  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🛠️ Technology Stack

### Programming Language
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Main development language |

### Machine Learning
| Library | Version | Usage |
|---------|---------|-------|
| scikit-learn | 1.3+ | Model training and evaluation |
| numpy | 1.24+ | Numerical computations |
| pandas | 2.0+ | Data manipulation |

### Natural Language Processing
| Library | Version | Usage |
|---------|---------|-------|
| TurkishStemmer | 1.3+ | Turkish stemming |
| transformers | 4.30+ | BERT model (optional) |

### Visualization
| Library | Version | Usage |
|---------|---------|-------|
| matplotlib | 3.7+ | Chart creation |
| seaborn | 0.12+ | Statistical visualization |

### Web Interface
| Library | Version | Usage |
|---------|---------|-------|
| gradio | 4.0+ | Demo interface |

---

## 📊 Dataset

### TRSAv1 (Turkish Sentiment Analysis v1)

| Property | Value |
|----------|-------|
| **Source** | [HuggingFace](https://huggingface.co/datasets/maydogan23/TRSAv1) / [GitHub](https://github.com/maydogan23/TRSAv1-Dataset) |
| **Size** | ~60,000+ reviews |
| **Language** | Turkish |
| **Scope** | Reviews collected from e-commerce platforms |
| **Classes** | Negative (0), Neutral (1), Positive (2) |
| **Format** | CSV |

### Class Distribution

```
Positive  ████████████████████████████████████████ 45%
Neutral   ████████████████████ 25%
Negative  ████████████████████████ 30%
```

### Sample Data

| Review | Sentiment |
|--------|-----------|
| "Ürün harika geldi, çok memnun kaldım" (Product arrived great, very satisfied) | Positive |
| "Kargo biraz geç geldi ama ürün güzel" (Shipping was a bit late but product is nice) | Neutral |
| "Berbat kalite, kesinlikle tavsiye etmem" (Terrible quality, definitely don't recommend) | Negative |

---

## 📐 Methodology

### 1. Data Preprocessing Pipeline

```python
def turkce_metin_normalize_et(text):
    """
    Steps:
    1. Unicode normalization (NFKC)
    2. Convert to lowercase
    3. URL → <url> tag
    4. Email → <email> tag
    5. Phone → <phone> tag
    6. Reduce repeating characters
    7. Turkish stemming with TurkishStemmer
    """
```

### 2. Feature Extraction

**TF-IDF (Term Frequency - Inverse Document Frequency)**

| Parameter | Value | Description |
|-----------|-------|-------------|
| ngram_range | (1, 2) | Unigram and bigram |
| min_df | 2 | Must appear in at least 2 documents |
| max_df | 0.9 | Must appear in at most 90% of documents |
| max_features | None | Unlimited features |

### 3. Model Training

#### Sentiment Analysis
- **Algorithm**: Logistic Regression
- **Split Ratio**: 80% Train, 10% Validation, 10% Test
- **Optimization**: L2 regularization

#### Spam Detection
- **Hybrid Approach**:
  - Rule-based score (60% weight)
  - IsolationForest anomaly score (40% weight)

---

## 📈 Model Performance

### Sentiment Analysis Metrics

| Model | Accuracy | F1 (Macro) | Precision | Recall |
|-------|----------|------------|-----------|--------|
| **TF-IDF + LR** | ~0.85 | ~0.78 | ~0.79 | ~0.77 |
| BERT (Optional) | ~0.88 | ~0.82 | ~0.83 | ~0.81 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.81 | 0.79 | 0.80 |
| Neutral | 0.72 | 0.70 | 0.71 |
| Positive | 0.85 | 0.88 | 0.86 |

### Spam Detection Metrics

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Hybrid** | ~0.92 | ~0.75 | ~0.80 | ~0.70 |

> **Note**: Values are approximate and may vary with different datasets.

---

## 🚀 Installation

### Requirements

- Python 3.8 or higher
- pip package manager
- (Optional) GPU - for BERT

### Step 1: Clone the Repository

```bash
git clone https://github.com/dusova/Turkish-ECommerce-Sentiment-Spam-Analyzer.git
cd Turkish-ECommerce-Sentiment-Spam-Analyzer
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train Models

```bash
# Data preprocessing (downloads dataset)
python src/preprocessing.py

# Train sentiment analysis model
python src/model.py

# Train spam detection model
python src/spam_detector.py
```

### Step 5: Launch Web Interface

```bash
python src/app.py
```

---

## 💻 Usage

### Programmatic Usage with Python

```python
from src.preprocessing import turkce_metin_normalize_et
from src.model import SentimentModel
from src.spam_detector import SpamDetector

# Load models
sentiment_model = SentimentModel.load("models/sentiment_model.pkl")
spam_model = SpamDetector.load("models/spam_model.pkl")

# Analyze a review
review = "Bu ürün gerçekten harika, çok beğendim!"
normalized = turkce_metin_normalize_et(review)

# Sentiment prediction
sentiment = sentiment_model.predict([normalized])[0]
print(f"Sentiment: {['Negative', 'Neutral', 'Positive'][sentiment]}")

# Spam check
spam_result = spam_model.analyze(review, normalized)
print(f"Spam Probability: {spam_result['spam_olasiligi']:.1%}")
```

### Using with Google Colab

1. Open the notebook in Colab
2. Run all cells with `Runtime > Run all`
3. Click on the Gradio demo link

---

## 📚 API Reference

### `turkce_metin_normalize_et(text, stemming_uygula=True)`

Normalizes Turkish text.

**Parameters:**
- `text` (str): Raw text
- `stemming_uygula` (bool): Apply stemming or not

**Returns:** str - Normalized text

---

### `SentimentModel`

Sentiment analysis model class.

**Methods:**
- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions
- `predict_proba(X)`: Return probabilities
- `save(path)`: Save the model
- `load(path)`: Load the model

---

### `SpamDetector`

Spam detection class.

**Methods:**
- `fit(raw_texts, normalized_texts)`: Train the model
- `analyze(raw_text, normalized_text)`: Analyze a single review
- `predict(texts)`: Batch prediction

---

## 📁 Project Structure

```
Turkish-ECommerce-Sentiment-Spam-Analyzer/
│
├── 📂 data/                          # Datasets
│   └── TRSAv1.csv                    # Main data (auto-downloaded)
│
├── 📂 docs/                          # Documentation
│   ├── tr/
│   │   └── README.md                 # Turkish documentation
│   ├── en/
│   │   └── README.md                 # English documentation (this file)
│   └── images/                       # Images
│
├── 📂 models/                        # Trained models
│   ├── sentiment_model.pkl           # Sentiment model
│   └── spam_model.pkl                # Spam model
│
├── 📂 notebooks/                     # Jupyter Notebooks
│   └── Turkish-ECommerce-...ipynb    # Main notebook
│
├── 📂 src/                           # Source code
│   ├── __init__.py                   # Package init
│   ├── app.py                        # Gradio interface
│   ├── model.py                      # Sentiment model
│   ├── preprocessing.py              # Data processing
│   ├── spam_detector.py              # Spam detection
│   └── utils.py                      # Utility functions
│
├── 📄 config.py                      # Configuration
├── 📄 requirements.txt               # Dependencies
├── 📄 LICENSE                        # MIT License
└── 📄 README.md                      # Main README
```

---

## 🔍 Example Analyses

### Example 1: Positive Review

**Input:**
```
"Ürün beklenenden çok daha iyi çıktı. Kargo hızlıydı, paketleme özenli. 
Kesinlikle tavsiye ederim, 5 yıldız hak ediyor!"
(Product turned out much better than expected. Shipping was fast, packaging careful.
Definitely recommend, deserves 5 stars!)
```

**Output:**
```json
{
  "sentiment": "Positive",
  "confidence": "94%",
  "spam_probability": "12%",
  "spam_label": "Genuine",
  "aspects": {
    "shipping": "Positive",
    "quality": "Positive"
  }
}
```

### Example 2: Spam Review

**Input:**
```
"MUKEMMEL URUN!!! EN IYISI BU!!! www.fakesite.com CLICK DON'T MISS!!!"
```

**Output:**
```json
{
  "sentiment": "Positive",
  "confidence": "72%",
  "spam_probability": "89%",
  "spam_label": "Spam",
  "spam_reasons": [
    "URL detected",
    "Excessive exclamation marks",
    "Generic phrases"
  ]
}
```

---

## ⚠️ Limitations

1. **Language**: Only supports Turkish reviews
2. **Domain**: Optimized for e-commerce reviews
3. **Class Balance**: Performance may decrease in neutral class
4. **Irony/Sarcasm**: Limited detection capability
5. **Context**: Insufficient context in short reviews

---

## 🔮 Future Improvements

- [ ] Multi-language support (English, Arabic)
- [ ] REST API integration
- [ ] Real-time streaming analysis
- [ ] Advanced BERT model
- [ ] Irony/sarcasm detection module
- [ ] Docker container support
- [ ] Web dashboard

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📞 Contact

**Mustafa Arda Düşova**
- GitHub: [@dusova](https://github.com/dusova)
- Email: [arda@codewithmad.com]

---

<div align="center">

### 🎓 2025 Machine Learning Course Project

**Date**: December 2025

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red?style=flat)](https://github.com)

</div>
