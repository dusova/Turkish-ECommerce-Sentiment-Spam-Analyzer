"""
============================================================
Türkçe E-Ticaret Yorum Analizi - Yardımcı Fonksiyonlar
============================================================
Bu modül veri indirme, model kaydetme gibi yardımcı işlemleri içerir.
"""

import os
import sys
import requests
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Any, cast

# Proje konfigürasyonunu yükle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        DATA_DIR,
        MODELS_DIR,
        HUGGINGFACE_DATASET,
        GITHUB_DATASET_URLS,
        LOCAL_DATASET_PATH
    )
except ImportError:
    DATA_DIR = Path("data")
    MODELS_DIR = Path("models")
    HUGGINGFACE_DATASET = "maydogan23/TRSAv1"
    GITHUB_DATASET_URLS = [
        "https://raw.githubusercontent.com/maydogan23/TRSAv1-Dataset/main/TRSAv1.csv"
    ]
    LOCAL_DATASET_PATH = DATA_DIR / "TRSAv1.csv"


# ============================================================
# VERİ İNDİRME
# ============================================================

def veri_indir(hedef_yol: Optional[str] = None, kaynak: str = "auto") -> pd.DataFrame:
    """
    Veri setini indirir veya yerel dosyadan okur.
    
    Args:
        hedef_yol: İndirme hedef yolu
        kaynak: "huggingface", "github", "local" veya "auto"
    
    Returns:
        pd.DataFrame: Veri seti
    """
    hedef_yol = hedef_yol or str(LOCAL_DATASET_PATH)
    
    # Dizini oluştur
    os.makedirs(os.path.dirname(hedef_yol), exist_ok=True)
    
    veri = None
    kullanilan_kaynak = None
    
    # Önce yerel dosyayı kontrol et
    if kaynak in ["local", "auto"] and os.path.exists(hedef_yol):
        print(f"[BİLGİ] Yerel dosya bulundu: {hedef_yol}")
        try:
            from .preprocessing import akilli_csv_oku
        except ImportError:
            from preprocessing import akilli_csv_oku  # type: ignore[no-redef]
        veri = akilli_csv_oku(hedef_yol)
        kullanilan_kaynak = "Yerel dosya"
    
    # HuggingFace'den indir
    if veri is None and kaynak in ["huggingface", "auto"]:
        try:
            from datasets import load_dataset, DatasetDict
            
            print(f"[İNDİRİLİYOR] HuggingFace: {HUGGINGFACE_DATASET}")
            
            dataset = load_dataset(HUGGINGFACE_DATASET)
            if isinstance(dataset, DatasetDict):
                split_name = "train" if "train" in dataset else list(dataset.keys())[0]
                veri = dataset[split_name].to_pandas()
            else:
                veri = dataset.to_pandas()  # type: ignore[union-attr]
            
            kullanilan_kaynak = f"HuggingFace: {HUGGINGFACE_DATASET}"
            print(f"[OK] Veri seti indirildi: {len(cast(pd.DataFrame, veri)):,} satır")
            
            # Yerel olarak kaydet
            cast(pd.DataFrame, veri).to_csv(hedef_yol, index=False)
            print(f"[OK] Yerel olarak kaydedildi: {hedef_yol}")
            
        except Exception as e:
            print(f"[UYARI] HuggingFace'den indirilemedi: {e}")
    
    # GitHub'dan indir
    if veri is None and kaynak in ["github", "auto"]:
        for url in GITHUB_DATASET_URLS:
            try:
                print(f"[İNDİRİLİYOR] GitHub: {url[:50]}...")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Dosyaya kaydet
                with open(hedef_yol, 'wb') as f:
                    f.write(response.content)
                
                try:
                    from .preprocessing import akilli_csv_oku
                except ImportError:
                    from preprocessing import akilli_csv_oku  # type: ignore[no-redef]
                veri = akilli_csv_oku(hedef_yol)
                
                kullanilan_kaynak = "GitHub"
                print(f"[OK] Veri seti indirildi: {len(veri):,} satır")
                break
                
            except Exception as e:
                print(f"[UYARI] GitHub'dan indirilemedi: {e}")
                continue
    
    if veri is None:
        raise RuntimeError(
            "Veri seti indirilemedi! Lütfen manuel olarak indirin:\n"
            f"1. {GITHUB_DATASET_URLS[0]} adresinden indirin\n"
            f"2. {hedef_yol} olarak kaydedin"
        )
    
    print(f"[BİLGİ] Kullanılan kaynak: {kullanilan_kaynak}")
    
    return cast(pd.DataFrame, veri)


# ============================================================
# MODEL KAYDETME/YÜKLEME
# ============================================================

def model_kaydet(model, dosya_adi: str, klasor: Optional[str] = None):
    """
    Modeli pickle formatında kaydeder.
    
    Args:
        model: Kaydedilecek model
        dosya_adi: Dosya adı (örn: "sentiment_model.pkl")
        klasor: Hedef klasör (varsayılan: models/)
    """
    klasor = klasor or str(MODELS_DIR)
    os.makedirs(klasor, exist_ok=True)
    
    tam_yol = os.path.join(klasor, dosya_adi)
    joblib.dump(model, tam_yol)
    
    print(f"[OK] Model kaydedildi: {tam_yol}")
    return tam_yol


def model_yukle(dosya_adi: str, klasor: Optional[str] = None):
    """
    Kaydedilmiş modeli yükler.
    
    Args:
        dosya_adi: Dosya adı
        klasor: Kaynak klasör
    
    Returns:
        Model nesnesi
    """
    klasor = klasor or str(MODELS_DIR)
    tam_yol = os.path.join(klasor, dosya_adi)
    
    if not os.path.exists(tam_yol):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {tam_yol}\n"
            f"Önce 'python src/model.py' ile modeli eğitin."
        )
    
    model = joblib.load(tam_yol)
    print(f"[OK] Model yüklendi: {tam_yol}")
    
    return model


# ============================================================
# DİĞER YARDIMCI FONKSİYONLAR
# ============================================================

def proje_bilgisi():
    """Proje hakkında bilgi yazdırır."""
    print("=" * 60)
    print("TÜRKÇE E-TİCARET YORUM ANALİZİ")
    print("=" * 60)
    print(f"Veri Klasörü: {DATA_DIR}")
    print(f"Model Klasörü: {MODELS_DIR}")
    print(f"Veri Seti: {HUGGINGFACE_DATASET}")
    print("=" * 60)


def dosya_boyutu_formatla(boyut_byte: Union[int, float]) -> str:
    """Dosya boyutunu okunabilir formata çevirir."""
    boyut: float = float(boyut_byte)
    for birim in ['B', 'KB', 'MB', 'GB']:
        if boyut < 1024:
            return f"{boyut:.2f} {birim}"
        boyut /= 1024
    return f"{boyut:.2f} TB"


def gpu_kontrol():
    """GPU kullanılabilirliğini kontrol eder."""
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            gpu_adi = torch.cuda.get_device_name(0)
            print(f"[OK] GPU bulundu: {gpu_adi}")
            return True
        else:
            print("[BİLGİ] GPU bulunamadı, CPU kullanılacak")
            return False
    except ImportError:
        print("[BİLGİ] PyTorch kurulu değil")
        return False


# ============================================================
# KOMUT SATIRI
# ============================================================

if __name__ == "__main__":
    proje_bilgisi()
    
    print("\n[TEST] Veri indirme testi...")
    try:
        veri = veri_indir()
        print(f"[OK] {len(veri):,} satır indirildi")
    except Exception as e:
        print(f"[HATA] {e}")
