"""
============================================================
Türkçe E-Ticaret Yorum Analizi - Gradio Web Arayüzü
============================================================
Bu modül Gradio ile web arayüzü oluşturur.

Kullanım:
    python src/app.py
    
    veya
    
    from src.app import demo
    demo.launch()
"""

import os
import sys
import gradio as gr
from typing import Dict, Any, Optional

# Proje yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import turkce_metin_normalize_et
from src.model import SentimentModel
from src.spam_detector import SpamDetector

# Konfigürasyon
try:
    from config import (
        SENTIMENT_MODEL_PATH,
        SPAM_MODEL_PATH,
        GRADIO_CONFIG,
        ASPECT_KEYWORDS,
        SENTIMENT_CLASSES
    )
except ImportError:
    SENTIMENT_MODEL_PATH = "models/sentiment_model.pkl"
    SPAM_MODEL_PATH = "models/spam_model.pkl"
    GRADIO_CONFIG = {"title": "Türkçe E-Ticaret Yorum Analizi", "share": True}
    ASPECT_KEYWORDS = {}
    SENTIMENT_CLASSES = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}


# ============================================================
# MODEL YÜKLEME
# ============================================================

duygu_modeli = None
spam_modeli = None

def modelleri_yukle():
    """Eğitilmiş modelleri yükler."""
    global duygu_modeli, spam_modeli
    
    try:
        duygu_modeli = SentimentModel.load(str(SENTIMENT_MODEL_PATH))
        print("[OK] Duygu modeli yüklendi")
    except Exception as e:
        print(f"[UYARI] Duygu modeli yüklenemedi: {e}")
        print("        Önce 'python src/model.py' çalıştırın")
        duygu_modeli = None
    
    try:
        spam_modeli = SpamDetector.load(str(SPAM_MODEL_PATH))
        print("[OK] Spam modeli yüklendi")
    except Exception as e:
        print(f"[UYARI] Spam modeli yüklenemedi: {e}")
        print("        Önce 'python src/spam_detector.py' çalıştırın")
        spam_modeli = None


# ============================================================
# ASPEKT ANALİZİ
# ============================================================

def aspekt_analizi(yorum: str) -> dict:
    """
    Aspekt bazlı duygu analizi yapar.
    
    Args:
        yorum: Yorum metni
    
    Returns:
        dict: Aspekt → Duygu eşlemesi
    """
    if duygu_modeli is None:
        return {}
    
    normalize_metin = turkce_metin_normalize_et(yorum)
    cumleler = [c.strip() for c in normalize_metin.replace("!", ".").replace("?", ".").split(".") if c.strip()]
    
    aspekt_sonuclari = {}
    
    for aspekt_adi, anahtar_kelimeler in ASPECT_KEYWORDS.items():
        for cumle in cumleler:
            if any(kelime in cumle for kelime in anahtar_kelimeler):
                tahmin = int(duygu_modeli.predict([cumle])[0])
                aspekt_sonuclari[aspekt_adi] = {
                    "cumle": cumle,
                    "duygu": SENTIMENT_CLASSES[tahmin]
                }
                break
    
    return aspekt_sonuclari


# ============================================================
# ANA ANALİZ FONKSİYONU
# ============================================================

def analiz_yap(yorum: str) -> Dict[str, Any]:
    """
    Bir yorumu tam analiz eder.
    
    Args:
        yorum: Yorum metni
    
    Returns:
        dict: Analiz sonuçları
    """
    if not yorum or not yorum.strip():
        return {"hata": "Lütfen bir yorum girin."}
    
    normalize_metin = turkce_metin_normalize_et(yorum)
    
    sonuc: Dict[str, Any] = {
        "girdi": yorum,
        "normalize": normalize_metin
    }
    
    # Spam analizi
    if spam_modeli is not None:
        spam_sonuc = spam_modeli.analyze(yorum, normalize_metin)
        sonuc["spam_analizi"] = {
            "olasilik": f"{spam_sonuc['spam_olasiligi']:.1%}",
            "etiket": spam_sonuc["etiket"],
            "aciklama": spam_sonuc["aciklama"]
        }
    else:
        sonuc["spam_analizi"] = {"uyari": "Spam modeli yüklenmedi"}
    
    # Duygu analizi
    if duygu_modeli is not None:
        tahmin = int(duygu_modeli.predict([normalize_metin])[0])
        olasiliklar = duygu_modeli.predict_proba([normalize_metin])[0]
        
        sonuc["duygu_analizi"] = {
            "genel_duygu": SENTIMENT_CLASSES[tahmin],
            "olasiliklar": {
                "Negatif": f"{olasiliklar[0]:.1%}",
                "Nötr": f"{olasiliklar[1]:.1%}",
                "Pozitif": f"{olasiliklar[2]:.1%}"
            }
        }
    else:
        sonuc["duygu_analizi"] = {"uyari": "Duygu modeli yüklenmedi"}
    
    # Aspekt analizi
    aspektler = aspekt_analizi(yorum)
    if aspektler:
        sonuc["aspekt_analizi"] = {}
        for aspekt, detay in aspektler.items():
            aspekt_isim = aspekt.replace("_", "/").title()
            sonuc["aspekt_analizi"][aspekt_isim] = detay["duygu"]
    else:
        sonuc["aspekt_analizi"] = {"bilgi": "Spesifik aspekt bulunamadı"}
    
    return sonuc


# ============================================================
# GRADIO ARAYÜZÜ
# ============================================================

def gradio_arayuzu_olustur():
    """Gradio arayüzünü oluşturur."""
    
    arayuz = gr.Interface(
        fn=analiz_yap,
        
        inputs=gr.Textbox(
            lines=4,
            label="📝 Yorum Girin",
            placeholder="E-ticaret yorumunuzu buraya yazın...\n\nÖrnek: Ürün çok güzel geldi, paketleme özenliydi."
        ),
        
        outputs=gr.JSON(label="📊 Analiz Sonuçları"),
        
        title="🛒 Türkçe E-Ticaret Yorum Analizi",
        
        description="""
        ## Bu Demo Ne Yapar?
        
        Girdiğiniz Türkçe e-ticaret yorumunu analiz eder ve şunları gösterir:
        
        1. **🚫 Spam Analizi**: Yorumun sahte/bot olma olasılığı
        2. **💭 Duygu Analizi**: Yorumun genel tonu (olumlu/olumsuz/nötr)
        3. **📊 Aspekt Analizi**: Kargo, fiyat, kalite gibi konulardaki duygu
        
        ---
        
        **Teknoloji**: TF-IDF + Logistic Regression + IsolationForest  
        **Veri Seti**: TRSAv1 (Türkçe e-ticaret yorumları)
        """,
        
        examples=[
            ["Kargo çok hızlı geldi, paketleme de gayet güzeldi. Ürün beklediğim gibiydi, teşekkürler!"],
            ["MUKEMMEL URUN!!! Herkes alsın, çok güzel www.indirim.com"],
            ["Ürün kötüydü ama satıcı çok ilgiliydi, iade işlemi sorunsuz oldu."],
            ["Fiyatına göre idare eder. Ne iyi ne kötü."],
            ["Paket yırtık geldi, ürün kırılmıştı. Tam bir hayal kırıklığı. İade ettim."]
        ],
        
        article="""
        ---
        
        ### Model Hakkında
        
        Bu model, TF-IDF (Term Frequency - Inverse Document Frequency) ve Lojistik Regresyon 
        kullanılarak eğitilmiştir.
        
        **Sınırlamalar:**
        - İroni ve alaycı yorumları anlayamayabilir
        - Çok kısa yorumlarda performans düşebilir
        - Sadece Türkçe yorumlar desteklenir
        
        ---
        
        **Proje Ekibi**: Mustafa Arda Düşova, Fatih Çoban, Efe Ata  
        **Tarih**: Aralık 2025
        """
    )
    
    return arayuz


# ============================================================
# ANA FONKSİYON
# ============================================================

def main():
    """Uygulamayı başlatır."""
    print("=" * 60)
    print("TÜRKÇE E-TİCARET YORUM ANALİZİ - WEB ARAYÜZÜ")
    print("=" * 60)
    
    # Modelleri yükle
    print("\n[1/2] Modeller yükleniyor...")
    modelleri_yukle()
    
    # Arayüzü oluştur ve başlat
    print("\n[2/2] Arayüz başlatılıyor...")
    demo = gradio_arayuzu_olustur()
    
    print("\n" + "=" * 60)
    print("Uygulama başlatılıyor...")
    print("=" * 60)
    
    demo.launch(
        share=GRADIO_CONFIG.get("share", True),
        server_port=GRADIO_CONFIG.get("server_port", 7860),
        server_name="0.0.0.0"  # Tüm IP'lerden erişime izin ver
    )


# Demo nesnesi (import için)
demo = None

if __name__ == "__main__":
    main()
else:
    # Import edildiğinde modelleri yükle
    modelleri_yukle()
    demo = gradio_arayuzu_olustur()
