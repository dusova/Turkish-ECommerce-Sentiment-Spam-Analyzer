"""
============================================================
Türkçe E-Ticaret Yorum Analizi - Konfigürasyon Dosyası
============================================================
Bu dosya tüm proje ayarlarını merkezi olarak yönetir.
"""

import os
from pathlib import Path

# ============================================================
# TEMEL YOLLAR
# ============================================================

# Proje kök dizini
ROOT_DIR = Path(__file__).parent.absolute()

# Alt dizinler
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"

# Dizinleri oluştur (yoksa)
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================
# VERİ SETİ AYARLARI
# ============================================================

# HuggingFace veri seti adı
HUGGINGFACE_DATASET = "maydogan23/TRSAv1"

# GitHub yedek URL'leri
GITHUB_DATASET_URLS = [
    "https://raw.githubusercontent.com/maydogan23/TRSAv1-Dataset/main/TRSAv1.csv",
    "https://github.com/maydogan23/TRSAv1-Dataset/raw/main/TRSAv1.csv"
]

# Yerel veri dosyası
LOCAL_DATASET_PATH = DATA_DIR / "TRSAv1.csv"

# ============================================================
# MODEL AYARLARI
# ============================================================

# Rastgele tohum (tekrarlanabilirlik için)
RANDOM_SEED = 42

# Veri bölme oranları
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# Örnekleme (None = tüm veri)
SAMPLE_SIZE = 60000

# Model dosyaları
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
SPAM_MODEL_PATH = MODELS_DIR / "spam_model.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# ============================================================
# TF-IDF AYARLARI
# ============================================================

TFIDF_CONFIG = {
    "ngram_range": (1, 2),      # Unigram ve bigram
    "min_df": 2,                # En az 2 belgede geçmeli
    "max_df": 0.9,              # En fazla %90 belgede geçmeli
    "max_features": None,       # Özellik sınırı yok
}

# ============================================================
# SPAM TESPİTİ AYARLARI
# ============================================================

# IsolationForest parametreleri
ISOLATION_FOREST_CONFIG = {
    "n_estimators": 100,
    "contamination": 0.05,      # Beklenen anomali oranı
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# Spam tespiti için TF-IDF
SPAM_TFIDF_CONFIG = {
    "max_features": 2000,
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.9
}

# Jenerik olumlu ifadeler (spam göstergesi)
GENERIC_POSITIVE_PHRASES = [
    "harika", "mukemmel", "super", "tavsiye ederim",
    "kesinlikle alin", "bayildim", "muhtesem", "efsane",
    "cok iyi", "gayet iyi"
]

# Jenerik olumsuz ifadeler (spam göstergesi)
GENERIC_NEGATIVE_PHRASES = [
    "berbat", "rezalet", "asla almayin",
    "dolandiricilik", "iade", "pismanlik"
]

# ============================================================
# ASPEKT ANALİZİ AYARLARI
# ============================================================

ASPECT_KEYWORDS = {
    "kargo_paket": [
        "kargo", "kargolama", "teslimat", "paket", "paketleme",
        "koli", "kurye", "kury", "teslim", "geldi", "ulasti"
    ],
    "fiyat": [
        "fiyat", "pahali", "ucuz", "indirim", "kampanya",
        "deger", "para", "ucret", "hesapli", "uygun"
    ],
    "kalite": [
        "kalite", "malzeme", "iscilik", "dayanikli", "saglam",
        "bozuk", "kirik", "yirtik", "kaliteli", "kalitesiz"
    ],
    "satici": [
        "satici", "magaza", "firma", "musteri hizmetleri",
        "destek", "iletisim", "ilgili", "ilgisiz"
    ],
    "iade": [
        "iade", "degisim", "garanti", "servis", "geri gonder",
        "iptal", "problem", "sorun"
    ],
}

# ============================================================
# DUYGU ANALİZİ AYARLARI
# ============================================================

# Sınıf isimleri
SENTIMENT_CLASSES = {
    0: "Negatif",
    1: "Nötr",
    2: "Pozitif"
}

SPAM_CLASSES = {
    0: "Gerçek",
    1: "Spam",
    -1: "Belirsiz"
}

# Etiket eşleme (farklı formatları standartlaştırma)
LABEL_MAPPING = {
    # Negatif
    "negative": 0, "neg": 0, "olumsuz": 0, "kötü": 0, "kotu": 0,
    "0": 0, "-1": 0,
    # Nötr
    "neutral": 1, "neu": 1, "nötr": 1, "notr": 1,
    "1": 1,
    # Pozitif
    "positive": 2, "pos": 2, "olumlu": 2, "iyi": 2,
    "2": 2
}

# ============================================================
# GRADIO AYARLARI
# ============================================================

GRADIO_CONFIG = {
    "title": "Türkçe E-Ticaret Yorum Analizi",
    "share": True,              # Public link oluştur
    "server_port": 7860,
    "theme": "default"
}

# ============================================================
# BERT AYARLARI (OPSİYONEL)
# ============================================================

BERT_CONFIG = {
    "model_name": "dbmdz/bert-base-turkish-cased",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01
}

# ============================================================
# LOGLAMA AYARLARI
# ============================================================

LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log"
}
