"""
============================================================
Türkçe E-Ticaret Yorum Analizi - Spam Tespit Modülü
============================================================
Bu modül spam/bot yorum tespiti işlemlerini içerir.

Kullanım:
    from src.spam_detector import SpamDetector
    
    detector = SpamDetector()
    sonuc = detector.analyze("MUKEMMEL URUN!!! www.site.com")
"""

import os
import sys
import re
import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, cast

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score

# Proje konfigürasyonunu yükle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        RANDOM_SEED,
        SPAM_MODEL_PATH,
        SPAM_TFIDF_CONFIG,
        ISOLATION_FOREST_CONFIG,
        GENERIC_POSITIVE_PHRASES,
        GENERIC_NEGATIVE_PHRASES,
        SPAM_CLASSES
    )
except ImportError:
    RANDOM_SEED = 42
    SPAM_MODEL_PATH = "models/spam_model.pkl"
    SPAM_TFIDF_CONFIG = {"max_features": 2000, "ngram_range": (1, 2)}
    ISOLATION_FOREST_CONFIG = {"n_estimators": 100, "contamination": 0.05}
    GENERIC_POSITIVE_PHRASES = ["harika", "mukemmel", "super"]
    GENERIC_NEGATIVE_PHRASES = ["berbat", "rezalet"]
    SPAM_CLASSES = {0: "Gerçek", 1: "Spam", -1: "Belirsiz"}


# Emoji tespiti için düzenli ifade
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF]+", flags=re.UNICODE)


# ============================================================
# SPAM TESPİT SINIFI
# ============================================================

class SpamDetector:
    """
    Hibrit spam tespit sistemi.
    
    İki yöntemi birleştirir:
    1. Kural Tabanlı: URL, tekrar, jenerik ifade tespiti
    2. IsolationForest: Anomali tespiti
    
    Attributes:
        rule_weight: Kural tabanlı skorun ağırlığı
        anomaly_weight: Anomali skorunun ağırlığı
        pipeline: TF-IDF + LogisticRegression pipeline
        isolation_forest: IsolationForest modeli
    """
    
    def __init__(self, rule_weight: float = 0.6, anomaly_weight: float = 0.4):
        """
        SpamDetector oluşturur.
        
        Args:
            rule_weight: Kural tabanlı skorun ağırlığı
            anomaly_weight: Anomali skorunun ağırlığı
        """
        self.rule_weight = rule_weight
        self.anomaly_weight = anomaly_weight
        
        self.pipeline: Optional[Pipeline] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        
        self.is_trained = False
        self.classes = SPAM_CLASSES
    
    # ============================================================
    # KURAL TABANLI TESPİT
    # ============================================================
    
    @staticmethod
    def _buyuk_harf_orani(metin: str) -> float:
        """Metindeki büyük harf oranını hesaplar."""
        harfler = [k for k in metin if k.isalpha()]
        if not harfler:
            return 0.0
        return sum(1 for k in harfler if k.isupper()) / len(harfler)
    
    def kural_tabanli_skor(self, ham_metin: str, normalize_metin: str) -> int:
        """
        Kural tabanlı spam skoru hesaplar.
        
        Args:
            ham_metin: Orijinal metin
            normalize_metin: Normalize edilmiş metin
        
        Returns:
            int: 1 (spam), 0 (gerçek), -1 (belirsiz)
        """
        metin = normalize_metin
        ham = ham_metin if isinstance(ham_metin, str) else str(ham_metin)
        
        # Göstergeleri kontrol et
        link_var = any(tag in metin for tag in ["<url>", "<email>", "<phone>"])
        unlem_sayisi = ham.count("!")
        emoji_sayisi = len(EMOJI_PATTERN.findall(ham))
        buyuk_harf_orani = self._buyuk_harf_orani(ham)
        kelime_sayisi = len(metin.split())
        
        # Jenerik ifade kontrolü
        jenerik_var = (
            any(ifade in metin for ifade in GENERIC_POSITIVE_PHRASES) or
            any(ifade in metin for ifade in GENERIC_NEGATIVE_PHRASES)
        )
        
        # Puanlama
        toplam_puan = 0
        
        if link_var:
            toplam_puan += 3
        if unlem_sayisi >= 4:
            toplam_puan += 1
        if emoji_sayisi >= 3:
            toplam_puan += 1
        if buyuk_harf_orani > 0.6 and len(ham) > 10:
            toplam_puan += 1
        if kelime_sayisi <= 3 and jenerik_var:
            toplam_puan += 2
        if len(metin) <= 20 and jenerik_var:
            toplam_puan += 1
        
        # Gerçek şikayetleri spam olarak etiketleme
        sikayet_kelimeleri = ["iade", "degisim", "kirik", "bozuk", "gecikti"]
        gercek_sikayet = (
            "<url>" not in metin and
            kelime_sayisi >= 6 and
            any(k in metin for k in sikayet_kelimeleri)
        )
        
        if gercek_sikayet:
            return 0
        
        if toplam_puan >= 3:
            return 1
        elif toplam_puan == 0:
            return 0
        else:
            return -1
    
    # ============================================================
    # MODEL EĞİTİMİ
    # ============================================================
    
    def fit(
        self, 
        ham_metinler: List[str], 
        normalize_metinler: List[str]
    ) -> "SpamDetector":
        """
        Spam tespiti modelini eğitir.
        
        Args:
            ham_metinler: Orijinal metinler
            normalize_metinler: Normalize edilmiş metinler
        
        Returns:
            self: Eğitilmiş model
        """
        print("[EĞİTİM] Spam tespit modeli eğitiliyor...")
        
        # 1. Kural tabanlı etiketler üret
        print("  [1/3] Kural tabanlı etiketler oluşturuluyor...")
        kural_etiketler = [
            self.kural_tabanli_skor(ham, norm)
            for ham, norm in zip(ham_metinler, normalize_metinler)
        ]
        
        # 2. Tekrar eden yorumları tespit et
        metin_sayilari: pd.Series = pd.Series(normalize_metinler).value_counts()  # type: ignore[assignment]
        cok_tekrar = set(metin_sayilari[metin_sayilari >= 10].index.tolist())
        
        for i, metin in enumerate(normalize_metinler):
            if metin in cok_tekrar:
                kural_etiketler[i] = 1
        
        # 3. IsolationForest eğit
        print("  [2/3] IsolationForest eğitiliyor...")
        self.tfidf_vectorizer = TfidfVectorizer(**SPAM_TFIDF_CONFIG)
        tfidf_matris = self.tfidf_vectorizer.fit_transform(normalize_metinler)
        
        self.isolation_forest = IsolationForest(**ISOLATION_FOREST_CONFIG)
        anomali_tahminleri = self.isolation_forest.fit_predict(tfidf_matris)
        anomali_etiketler = (anomali_tahminleri == -1).astype(int)
        
        # 4. Hibrit etiketler oluştur
        print("  [3/3] Hibrit etiketler birleştiriliyor...")
        hibrit_etiketler = []
        for kural, anomali in zip(kural_etiketler, anomali_etiketler):
            if kural == 1:
                hibrit_etiketler.append(1)
            elif kural == 0 and anomali == 0:
                hibrit_etiketler.append(0)
            elif kural == 0 and anomali == 1:
                hibrit_etiketler.append(-1)
            elif kural == -1 and anomali == 1:
                hibrit_etiketler.append(1)
            else:
                hibrit_etiketler.append(-1)
        
        # 5. Kesin etiketli verilerle LogisticRegression eğit
        kesin_indeksler = [i for i, e in enumerate(hibrit_etiketler) if e != -1]
        X_kesin = [normalize_metinler[i] for i in kesin_indeksler]
        y_kesin = [hibrit_etiketler[i] for i in kesin_indeksler]
        
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**SPAM_TFIDF_CONFIG)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ))
        ])
        
        self.pipeline.fit(X_kesin, y_kesin)
        self.is_trained = True
        
        print(f"[OK] Eğitim tamamlandı!")
        print(f"  Toplam: {len(normalize_metinler):,} yorum")
        print(f"  Spam: {sum(1 for e in hibrit_etiketler if e == 1):,}")
        print(f"  Gerçek: {sum(1 for e in hibrit_etiketler if e == 0):,}")
        print(f"  Belirsiz: {sum(1 for e in hibrit_etiketler if e == -1):,}")
        
        return self
    
    # ============================================================
    # TAHMİN
    # ============================================================
    
    def predict(self, metinler: List[str]) -> np.ndarray:
        """Spam tahmini yapar."""
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model henüz eğitilmedi!")
        return cast(np.ndarray, self.pipeline.predict(metinler))
    
    def predict_proba(self, metinler: List[str]) -> np.ndarray:
        """Spam olasılığı döndürür."""
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model henüz eğitilmedi!")
        return cast(np.ndarray, self.pipeline.predict_proba(metinler))
    
    def analyze(self, ham_metin: str, normalize_metin: Optional[str] = None) -> Dict:
        """
        Tek bir yorumu analiz eder.
        
        Args:
            ham_metin: Orijinal metin
            normalize_metin: Normalize metin (None ise hesaplanır)
        
        Returns:
            dict: Analiz sonuçları
        """
        try:
            from .preprocessing import turkce_metin_normalize_et
        except ImportError:
            from preprocessing import turkce_metin_normalize_et  # type: ignore[no-redef]
        
        if normalize_metin is None:
            normalize_metin = turkce_metin_normalize_et(ham_metin)
        
        # Kural tabanlı skor
        kural_skor = self.kural_tabanli_skor(ham_metin, normalize_metin)
        
        # Model tahmini
        if self.is_trained:
            model_tahmin = int(self.predict([normalize_metin])[0])
            model_proba = float(self.predict_proba([normalize_metin])[0, 1])
        else:
            model_tahmin = kural_skor
            model_proba = 1.0 if kural_skor == 1 else 0.0
        
        return {
            "spam_olasiligi": model_proba,
            "tahmin": model_tahmin,
            "etiket": self.classes[model_tahmin],
            "kural_skoru": kural_skor,
            "aciklama": self._aciklama_olustur(ham_metin, normalize_metin)
        }
    
    def _aciklama_olustur(self, ham: str, norm: str) -> str:
        """Spam göstergelerini açıklar."""
        gostergeler = []
        
        if "<url>" in norm or "<email>" in norm:
            gostergeler.append("URL/e-posta içeriyor")
        if ham.count("!") >= 4:
            gostergeler.append("Çok fazla ünlem işareti")
        if len(EMOJI_PATTERN.findall(ham)) >= 3:
            gostergeler.append("Çok fazla emoji")
        if self._buyuk_harf_orani(ham) > 0.6:
            gostergeler.append("Çoğunlukla büyük harf")
        
        if not gostergeler:
            return "Normal yorum"
        
        return "Spam göstergeleri: " + ", ".join(gostergeler)
    
    # ============================================================
    # KAYIT/YÜKLEME
    # ============================================================
    
    def save(self, path: Optional[str] = None):
        """Modeli kaydeder."""
        path = path or str(SPAM_MODEL_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[OK] Spam modeli kaydedildi: {path}")
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> "SpamDetector":
        """Modeli yükler."""
        path = path or str(SPAM_MODEL_PATH)
        model = joblib.load(path)
        print(f"[OK] Spam modeli yüklendi: {path}")
        return model


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def train_spam_model(
    ham_metinler: List[str],
    normalize_metinler: List[str],
    save_path: Optional[str] = None
) -> SpamDetector:
    """Spam modelini eğitir."""
    detector = SpamDetector()
    detector.fit(ham_metinler, normalize_metinler)
    
    if save_path:
        detector.save(save_path)
    
    return detector


def load_spam_model(path: Optional[str] = None) -> SpamDetector:
    """Kayıtlı spam modelini yükler."""
    return SpamDetector.load(path)


# ============================================================
# KOMUT SATIRI
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPAM TESPİT MODELİ EĞİTİMİ")
    print("=" * 60)
    
    from preprocessing import veri_hazirla
    
    # Veriyi hazırla
    print("\n[1/3] Veri hazırlanıyor...")
    veri = veri_hazirla()
    
    # Model eğit
    print("\n[2/3] Model eğitiliyor...")
    detector = train_spam_model(
        veri["ham_metin"].tolist(),
        veri["metin"].tolist()
    )
    
    # Kaydet
    print("\n[3/3] Model kaydediliyor...")
    detector.save()
    
    # Örnek analiz
    print("\n" + "=" * 60)
    print("ÖRNEK ANALİZ")
    print("=" * 60)
    
    ornekler = [
        "MUKEMMEL URUN!!! www.site.com'dan alin",
        "Ürün geldi ama kutusu kırıktı, iade ediyorum",
        "Harika!",
    ]
    
    for ornek in ornekler:
        sonuc = detector.analyze(ornek)
        print(f"\nMetin: {ornek}")
        print(f"  Spam Olasılığı: {sonuc['spam_olasiligi']:.2%}")
        print(f"  Etiket: {sonuc['etiket']}")
        print(f"  Açıklama: {sonuc['aciklama']}")
