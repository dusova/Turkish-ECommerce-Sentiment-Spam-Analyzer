"""
============================================================
Türkçe E-Ticaret Yorum Analizi - Model Eğitim Modülü
============================================================
Bu modül duygu analizi modelinin eğitimi ve kullanımını içerir.

Kullanım:
    from src.model import SentimentModel, train_sentiment_model
    
    model = train_sentiment_model(X_train, y_train)
    tahmin = model.predict(["Harika bir ürün!"])
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, cast

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# Proje konfigürasyonunu yükle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        RANDOM_SEED,
        TRAIN_SIZE,
        TFIDF_CONFIG,
        SENTIMENT_MODEL_PATH,
        SENTIMENT_CLASSES
    )
except ImportError:
    RANDOM_SEED = 42
    TRAIN_SIZE = 0.8
    TFIDF_CONFIG = {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.9}
    SENTIMENT_MODEL_PATH = "models/sentiment_model.pkl"
    SENTIMENT_CLASSES = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}


# ============================================================
# DUYGU ANALİZİ MODELİ
# ============================================================

class SentimentModel:
    """
    Türkçe duygu analizi modeli.
    
    TF-IDF + Logistic Regression tabanlı sınıflandırıcı.
    
    Attributes:
        pipeline: Eğitilmiş sklearn Pipeline
        classes: Sınıf isimleri
        is_trained: Model eğitildi mi?
    
    Örnek:
        >>> model = SentimentModel()
        >>> model.fit(X_train, y_train)
        >>> tahmin = model.predict(["Harika ürün!"])
        >>> print(tahmin)  # [2] (Pozitif)
    """
    
    def __init__(self, tfidf_config: Optional[dict] = None):
        """
        Model oluşturur.
        
        Args:
            tfidf_config: TF-IDF vektörizör ayarları
        """
        self.tfidf_config = tfidf_config or TFIDF_CONFIG
        self.pipeline: Optional[Pipeline] = None
        self.classes = SENTIMENT_CLASSES
        self.is_trained = False
        
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Sklearn pipeline oluşturur."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**self.tfidf_config)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight=None
            ))
        ])
    
    def fit(self, X: List[str], y: List[int]) -> "SentimentModel":
        """
        Modeli eğitir.
        
        Args:
            X: Metin listesi
            y: Etiket listesi (0, 1, 2)
        
        Returns:
            self: Eğitilmiş model
        """
        print(f"[EĞİTİM] Model eğitiliyor ({len(X):,} örnek)...")
        
        baslangic = time.time()
        if self.pipeline is None:
            raise ValueError("Pipeline oluşturulamadı!")
        self.pipeline.fit(X, y)
        gecen_sure = time.time() - baslangic
        
        self.is_trained = True
        
        # TF-IDF özellik sayısı
        tfidf = self.pipeline.named_steps["tfidf"]
        kelime_sayisi = len(tfidf.vocabulary_)
        
        print(f"[OK] Eğitim tamamlandı! Süre: {gecen_sure:.2f} saniye")
        print(f"[BİLGİ] TF-IDF özellik sayısı: {kelime_sayisi:,}")
        
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Tahmin yapar.
        
        Args:
            X: Metin listesi
        
        Returns:
            np.ndarray: Tahmin edilen sınıflar (0, 1, 2)
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi! Önce fit() çağırın.")
        
        if self.pipeline is None:
            raise ValueError("Pipeline oluşturulamadı!")
        return cast(np.ndarray, self.pipeline.predict(X))
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Olasılık tahmini yapar.
        
        Args:
            X: Metin listesi
        
        Returns:
            np.ndarray: Her sınıf için olasılıklar
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        if self.pipeline is None:
            raise ValueError("Pipeline oluşturulamadı!")
        return cast(np.ndarray, self.pipeline.predict_proba(X))
    
    def predict_label(self, text: str) -> str:
        """
        Tek bir metin için duygu etiketi döndürür.
        
        Args:
            text: Metin
        
        Returns:
            str: "Negatif", "Nötr" veya "Pozitif"
        """
        pred = self.predict([text])[0]
        return self.classes[pred]
    
    def evaluate(self, X: List[str], y: List[int]) -> dict:
        """
        Modeli değerlendirir.
        
        Args:
            X: Test metinleri
            y: Gerçek etiketler
        
        Returns:
            dict: Metrikler
        """
        tahminler = self.predict(X)
        
        sonuc = {
            "accuracy": accuracy_score(y, tahminler),
            "f1_macro": f1_score(y, tahminler, average="macro", zero_division=0),
            "precision_macro": precision_score(y, tahminler, average="macro", zero_division=0),
            "recall_macro": recall_score(y, tahminler, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y, tahminler).tolist(),
            "classification_report": classification_report(
                y, tahminler, 
                target_names=list(self.classes.values()),
                output_dict=True
            )
        }
        
        return sonuc
    
    def save(self, path: Optional[str] = None):
        """Modeli dosyaya kaydeder."""
        path = path or str(SENTIMENT_MODEL_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[OK] Model kaydedildi: {path}")
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> "SentimentModel":
        """Modeli dosyadan yükler."""
        path = path or str(SENTIMENT_MODEL_PATH)
        model = joblib.load(path)
        print(f"[OK] Model yüklendi: {path}")
        return model


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def train_sentiment_model(
    X: List[str], 
    y: List[int],
    save_path: Optional[str] = None
) -> SentimentModel:
    """
    Duygu analizi modelini eğitir.
    
    Args:
        X: Metin listesi
        y: Etiket listesi
        save_path: Model kayıt yolu (None ise kaydetmez)
    
    Returns:
        SentimentModel: Eğitilmiş model
    """
    model = SentimentModel()
    model.fit(X, y)
    
    if save_path:
        model.save(save_path)
    
    return model


def load_sentiment_model(path: Optional[str] = None) -> SentimentModel:
    """Kayıtlı modeli yükler."""
    return SentimentModel.load(path)


def train_test_val_split(
    X: List[str], 
    y: List[int],
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple:
    """
    Veriyi train/validation/test olarak böler.
    
    Args:
        X: Metin listesi
        y: Etiket listesi
        train_size: Eğitim oranı (varsayılan: 0.8)
        val_size: Validation oranı (varsayılan: 0.1)
        random_state: Rastgele tohum
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    random_state = random_state or RANDOM_SEED
    
    # İlk bölme: train ve geçici (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_size),
        random_state=random_state,
        stratify=y
    )
    
    # İkinci bölme: val ve test
    val_ratio = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"[BİLGİ] Veri bölümü:")
    print(f"  Eğitim: {len(X_train):,} örnek")
    print(f"  Validation: {len(X_val):,} örnek")
    print(f"  Test: {len(X_test):,} örnek")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# KOMUT SATIRI
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DUYGU ANALİZİ MODELİ EĞİTİMİ")
    print("=" * 60)
    
    from preprocessing import veri_hazirla
    
    # Veriyi hazırla
    print("\n[1/4] Veri hazırlanıyor...")
    veri = veri_hazirla()
    
    # Veriyi böl
    print("\n[2/4] Veri bölünüyor...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(
        veri["metin"].tolist(),
        veri["duygu"].tolist()
    )
    
    # Model eğit
    print("\n[3/4] Model eğitiliyor...")
    model = train_sentiment_model(X_train, y_train)
    
    # Değerlendir
    print("\n[4/4] Model değerlendiriliyor...")
    print("\n--- Validation Seti ---")
    val_sonuc = model.evaluate(X_val, y_val)
    print(f"Accuracy: {val_sonuc['accuracy']:.4f}")
    print(f"F1-Score (Macro): {val_sonuc['f1_macro']:.4f}")
    
    print("\n--- Test Seti ---")
    test_sonuc = model.evaluate(X_test, y_test)
    print(f"Accuracy: {test_sonuc['accuracy']:.4f}")
    print(f"F1-Score (Macro): {test_sonuc['f1_macro']:.4f}")
    
    # Kaydet
    print("\n[KAYIT]")
    model.save()
    
    print("\n" + "=" * 60)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 60)
