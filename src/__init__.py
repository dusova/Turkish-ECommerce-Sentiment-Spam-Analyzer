"""
============================================================
Türkçe E-Ticaret Yorum Analizi - src Paketi
============================================================
"""

from .preprocessing import (
    turkce_metin_normalize_et,
    akilli_csv_oku,
    metin_sutunu_bul,
    etiket_sutunu_bul
)

from .model import (
    SentimentModel,
    load_sentiment_model,
    train_sentiment_model
)

from .spam_detector import (
    SpamDetector,
    load_spam_model,
    train_spam_model
)

from .utils import (
    veri_indir,
    model_kaydet,
    model_yukle
)

__version__ = "1.0.0"
__author__ = "Mustafa Arda Düşova, Fatih Çoban, Efe Ata"
