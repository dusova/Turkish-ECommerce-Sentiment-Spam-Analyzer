"""
============================================================
Türkçe E-Ticaret Yorum Analizi - Veri Ön İşleme Modülü
============================================================
Bu modül metin normalizasyonu ve veri temizleme işlemlerini içerir.

Kullanım:
    from src.preprocessing import turkce_metin_normalize_et
    
    temiz_metin = turkce_metin_normalize_et("Harika bir ürün!!!")
"""

import os
import sys
import csv
import re
import unicodedata
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

# TurkishStemmer kurulu değilse uyarı ver
try:
    from TurkishStemmer import TurkishStemmer
    stemmer = TurkishStemmer()
    STEMMER_AVAILABLE = True
except ImportError:
    print("[UYARI] TurkishStemmer kurulu değil. 'pip install TurkishStemmer' ile kurun.")
    STEMMER_AVAILABLE = False
    stemmer = None

# Proje konfigürasyonunu yükle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        GENERIC_POSITIVE_PHRASES,
        GENERIC_NEGATIVE_PHRASES,
        LABEL_MAPPING,
        SENTIMENT_CLASSES
    )
except ImportError:
    # Varsayılan değerler
    GENERIC_POSITIVE_PHRASES = ["harika", "mukemmel", "super"]
    GENERIC_NEGATIVE_PHRASES = ["berbat", "rezalet"]
    LABEL_MAPPING = {}
    SENTIMENT_CLASSES = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}


# ============================================================
# METİN NORMALİZASYONU
# ============================================================

def turkce_metin_normalize_et(metin: str, stemming_uygula: bool = True) -> str:
    """
    Türkçe metni makine öğrenmesi için uygun hale getirir.
    
    İşlemler:
    1. Unicode normalizasyonu (özel karakterleri standartlaştırır)
    2. Küçük harfe çevirme
    3. URL'leri <url> etiketi ile değiştirme
    4. E-postaları <email> etiketi ile değiştirme
    5. Telefon numaralarını <phone> etiketi ile değiştirme
    6. Tekrar eden karakterleri azaltma ("çooook" -> "çook")
    7. Fazla boşlukları temizleme
    8. Stemming (kelime köküne indirgeme) - opsiyonel
    
    Args:
        metin: Ham metin
        stemming_uygula: True ise kelimelere stemming uygulanır
    
    Returns:
        str: Normalize edilmiş metin
    
    Örnek:
        >>> turkce_metin_normalize_et("HARIKA BİR ÜRÜN!!! www.site.com")
        'harik bir ürün <url>'
    """
    # Boş veya None kontrolü
    if not isinstance(metin, str):
        metin = "" if pd.isna(metin) else str(metin)
    
    # Unicode normalizasyonu + küçük harf
    sonuc = unicodedata.normalize("NFKC", metin).strip().lower()
    
    # URL'leri değiştir (http://... veya www. ile başlayanlar)
    sonuc = re.sub(r"(https?://\S+|www\.\S+)", " <url> ", sonuc)
    
    # E-posta adreslerini değiştir
    sonuc = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " <email> ", sonuc)
    
    # Türk telefon numaralarını değiştir (+90 5XX XXX XX XX formatı)
    sonuc = re.sub(r"\b(\+?90)?\s?(\(?\d{3}\)?)\s?\d{3}\s?\d{2}\s?\d{2}\b", " <phone> ", sonuc)
    
    # 3+ tekrar eden karakterleri 2'ye indir ("çooook" -> "çook")
    sonuc = re.sub(r"(.)\1{2,}", r"\1\1", sonuc)
    
    # Fazla boşlukları tek boşluğa indir
    sonuc = re.sub(r"\s+", " ", sonuc).strip()
    
    # STEMMING: Kelimeleri köklerine indir
    if stemming_uygula and sonuc and STEMMER_AVAILABLE and stemmer is not None:
        kelimeler = sonuc.split()
        kokler = []
        for kelime in kelimeler:
            # Özel etiketleri (url, email, phone) koruyalım
            if kelime.startswith("<") and kelime.endswith(">"):
                kokler.append(kelime)
            else:
                try:
                    kok = stemmer.stem(kelime)
                    kokler.append(kok if kok else kelime)
                except:
                    kokler.append(kelime)
        sonuc = " ".join(kokler)
    
    return sonuc


# ============================================================
# CSV OKUMA
# ============================================================

def akilli_csv_oku(dosya_yolu: str) -> pd.DataFrame:
    """
    CSV dosyasını akıllı şekilde okur.
    
    Bu fonksiyon:
    1. Dosyanın ilk 4KB'ini okur (hız için)
    2. Farklı encoding'leri dener (UTF-8, Latin1 vs.)
    3. Ayırıcı karakteri otomatik tespit eder
    4. Pandas DataFrame olarak döndürür
    
    Args:
        dosya_yolu: CSV dosyasının yolu
    
    Returns:
        pd.DataFrame: Okunan veri
    """
    # Dosyanın ilk 4096 byte'ini oku (örnek olarak yeterli)
    with open(dosya_yolu, "rb") as dosya:
        ham_veri = dosya.read(4096)
    
    # Farklı encoding'leri dene
    denenen_encodingler = ["utf-8-sig", "utf-8", "latin1", "windows-1254"]
    bulunan_encoding = None
    ornek_metin = ""
    
    for enc in denenen_encodingler:
        try:
            ornek_metin = ham_veri.decode(enc, errors="ignore")
            bulunan_encoding = enc
            break
        except Exception:
            continue
    
    print(f"[BİLGİ] Tespit edilen karakter kodlaması: {bulunan_encoding}")
    
    # Ayırıcı karakteri tespit et
    try:
        dialect = csv.Sniffer().sniff(ornek_metin, delimiters=",;\t")
        ayirici = dialect.delimiter
    except Exception:
        ayirici = ","
    
    ayirici_adi = {",": "virgül", ";": "noktalı virgül", "\t": "tab"}.get(ayirici, ayirici)
    print(f"[BİLGİ] Tespit edilen ayırıcı: {ayirici_adi} ({repr(ayirici)})")
    
    # Pandas ile oku
    veri_cercevesi = pd.read_csv(
        dosya_yolu,
        sep=ayirici,
        engine="python",
        encoding=bulunan_encoding
    )
    
    return veri_cercevesi


# ============================================================
# SÜTUN TESPİTİ
# ============================================================

# Metin sütununu bulmak için olası isimler
METIN_ANAHTAR_KELIMELERI = [
    "text", "review", "comment", "sentence",
    "yorum", "content", "body", "metin",
    "cumle", "cümle", "tweet"
]

# Duygu etiketi sütununu bulmak için olası isimler
ETIKET_ANAHTAR_KELIMELERI = [
    "label", "sentiment", "class", "polarity",
    "duygu", "target", "etiket", "kategori",
    "category", "emotion", "tag"
]

# Puan/yıldız sütununu bulmak için olası isimler
PUAN_ANAHTAR_KELIMELERI = [
    "rating", "star", "stars", "score", "rate",
    "puan", "yildiz", "rating_score"
]


def isimden_sutun_bul(sutun_listesi, anahtar_kelimeler: list) -> Optional[str]:
    """
    Sütun isimlerinde anahtar kelimeleri arar.
    
    Args:
        sutun_listesi: DataFrame'in sütun isimleri
        anahtar_kelimeler: Aranacak kelimeler
    
    Returns:
        str veya None: Bulunan sütun adı
    """
    kucuk_harf_sutunlar = {sutun.lower(): sutun for sutun in sutun_listesi}
    
    # Tam eşleşme ara
    for anahtar in anahtar_kelimeler:
        if anahtar in kucuk_harf_sutunlar:
            return kucuk_harf_sutunlar[anahtar]
    
    # Kısmi eşleşme ara
    for sutun in sutun_listesi:
        sutun_kucuk = sutun.lower()
        if any(anahtar in sutun_kucuk for anahtar in anahtar_kelimeler):
            return sutun
    
    return None


def metin_sutunu_bul(veri_cercevesi: pd.DataFrame) -> Optional[str]:
    """
    İçeriğe bakarak metin sütununu tespit eder.
    
    Mantık: Ortalama karakter uzunluğu en yüksek olan string sütunu
    muhtemelen yorum metnidir.
    
    Args:
        veri_cercevesi: pandas DataFrame
    
    Returns:
        str veya None: Bulunan sütun adı
    """
    # Önce isimden bulmayı dene
    isimden = isimden_sutun_bul(veri_cercevesi.columns, METIN_ANAHTAR_KELIMELERI)
    if isimden is not None:
        return isimden
    
    # İsimden bulamadıysak içeriğe bak
    en_iyi_sutun = None
    en_iyi_skor = -1
    
    for sutun in veri_cercevesi.columns:
        seri = veri_cercevesi[sutun]
        
        if seri.dtype != "object" and not str(seri.dtype).startswith("string"):
            continue
        
        ornek = seri.dropna().astype(str).head(5000)
        
        if len(ornek) < 100:
            continue
        
        ortalama_uzunluk = ornek.map(len).mean()
        benzersiz_sayi = ornek.nunique()
        skor = ortalama_uzunluk * np.log1p(benzersiz_sayi)
        
        if skor > en_iyi_skor:
            en_iyi_skor = skor
            en_iyi_sutun = sutun
    
    return en_iyi_sutun


def etiket_sutunu_bul(veri_cercevesi: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    İçeriğe bakarak etiket sütununu tespit eder.
    
    Returns:
        tuple: (sütun_adı, tespit_yöntemi) veya (None, None)
    """
    # Önce isimden bulmayı dene
    isimden = isimden_sutun_bul(veri_cercevesi.columns, ETIKET_ANAHTAR_KELIMELERI)
    if isimden is not None:
        return isimden, "isim_eslesmesi"
    
    # İsimden bulamadıysak içeriğe bak
    en_iyi_sutun = None
    en_iyi_skor = -1
    
    for sutun in veri_cercevesi.columns:
        skor = _etiket_sutunu_skorla(veri_cercevesi[sutun])
        if skor > en_iyi_skor:
            en_iyi_skor = skor
            en_iyi_sutun = sutun
    
    if en_iyi_skor >= 3:
        return en_iyi_sutun, "icerik_analizi"
    
    return None, None


def _etiket_sutunu_skorla(seri: pd.Series) -> float:
    """Bir sütunun duygu etiketi olma olasılığını skorlar."""
    seri_temiz = seri.dropna()
    
    if len(seri_temiz) < 500:
        return -1
    
    sayisal: pd.Series = pd.to_numeric(seri_temiz, errors="coerce")  # type: ignore[assignment]
    sayisal_temiz = sayisal.dropna()
    benzersiz_sayisal = set(sayisal_temiz.unique().tolist()) if len(sayisal_temiz) > 0 else set()
    
    metinsel = seri_temiz.astype(str).str.lower().str.strip()
    benzersiz_metinsel = set(metinsel.unique().tolist())
    
    skor = 0.0
    
    # Sayısal kalıplar
    if benzersiz_sayisal.issubset({0, 1, 2}) and len(benzersiz_sayisal) >= 2:
        skor += 5
    if benzersiz_sayisal.issubset({-1, 0, 1}) and len(benzersiz_sayisal) >= 2:
        skor += 4
    if benzersiz_sayisal.issubset({1, 2, 3, 4, 5}) and len(benzersiz_sayisal) >= 3:
        skor += 2
    
    # Metinsel kalıplar
    duygu_kelimesi_sayisi = 0
    for deger in list(benzersiz_metinsel)[:200]:
        if any(kelime in deger for kelime in ["neg", "olumsuz", "kötü", "kotu"]):
            duygu_kelimesi_sayisi += 1
        if any(kelime in deger for kelime in ["neu", "nötr", "notr"]):
            duygu_kelimesi_sayisi += 1
        if any(kelime in deger for kelime in ["pos", "olumlu", "iyi"]):
            duygu_kelimesi_sayisi += 1
    
    if duygu_kelimesi_sayisi >= 2 and len(benzersiz_metinsel) <= 10:
        skor += 4
    
    if len(benzersiz_metinsel) > 50 and len(benzersiz_sayisal) > 50:
        skor -= 5
    
    if 2 <= len(benzersiz_metinsel) <= 6:
        skor += 1
    
    return skor


def puan_sutunu_bul(veri_cercevesi: pd.DataFrame) -> Optional[str]:
    """Puan/yıldız (1-5) sütununu bulur."""
    isimden = isimden_sutun_bul(veri_cercevesi.columns, PUAN_ANAHTAR_KELIMELERI)
    if isimden is not None:
        return isimden
    
    for sutun in veri_cercevesi.columns:
        sayisal: pd.Series = pd.to_numeric(veri_cercevesi[sutun], errors="coerce")  # type: ignore[assignment]
        sayisal_temiz = sayisal.dropna()
        benzersiz = set(sayisal_temiz.unique().tolist()) if len(sayisal_temiz) > 0 else set()
        
        if benzersiz.issubset({1, 2, 3, 4, 5}) and len(benzersiz) >= 3:
            return sutun
    
    return None


# ============================================================
# ETİKET DÖNÜŞTÜRME
# ============================================================

def etiketi_duyguya_donustur(etiket_serisi: pd.Series) -> pd.Series:
    """
    Farklı formatlardaki etiketleri standart duygu değerlerine (0, 1, 2) dönüştürür.
    
    Desteklenen formatlar:
    - Sayısal: 0,1,2 veya -1,0,1
    - Metinsel: "negative"/"positive"/"neutral" veya Türkçe karşılıkları
    
    Args:
        etiket_serisi: pandas Series - ham etiket değerleri
    
    Returns:
        pandas Series: 0 (negatif), 1 (nötr), 2 (pozitif) değerleri
    """
    sayisal_degerler: pd.Series = pd.to_numeric(etiket_serisi, errors="coerce")  # type: ignore[assignment]
    
    def sayisal_esle(deger):  # type: ignore[no-untyped-def]
        if pd.isna(deger):
            return None
        try:
            deger_int = int(deger)
            if deger_int in [0, 1, 2]:
                return deger_int
            if deger_int in [-1, 0, 1]:
                return deger_int + 1
        except (ValueError, TypeError):
            pass
        return None
    
    duygu_degerleri: pd.Series = sayisal_degerler.apply(sayisal_esle)  # type: ignore[assignment]
    
    if duygu_degerleri.isna().any():
        metin_degerler = etiket_serisi.astype(str).str.lower().str.strip()
        metinsel_eslemeler = metin_degerler.map(LABEL_MAPPING)
        duygu_degerleri = duygu_degerleri.fillna(metinsel_eslemeler)
    
    return duygu_degerleri


# ============================================================
# ANA FONKSİYON
# ============================================================

def veri_hazirla(dosya_yolu: Optional[str] = None) -> pd.DataFrame:
    """
    Veri setini indirir/okur ve ön işleme uygular.
    
    Args:
        dosya_yolu: CSV dosya yolu (None ise otomatik indirir)
    
    Returns:
        pd.DataFrame: Hazırlanmış veri
    """
    try:
        from .utils import veri_indir
    except ImportError:
        from utils import veri_indir  # type: ignore[no-redef]
    
    # Veriyi indir veya oku
    if dosya_yolu is None:
        veri = veri_indir()
    else:
        veri = akilli_csv_oku(dosya_yolu)
    
    # Sütunları tespit et
    metin_sutunu = metin_sutunu_bul(veri)
    etiket_sutunu, _ = etiket_sutunu_bul(veri)
    
    if metin_sutunu is None:
        raise ValueError("Metin sütunu bulunamadı!")
    
    print(f"[BİLGİ] Metin sütunu: {metin_sutunu}")
    print(f"[BİLGİ] Etiket sütunu: {etiket_sutunu}")
    
    # Veriyi hazırla
    veri = veri.copy()
    veri["ham_metin"] = veri[metin_sutunu].astype(str)
    veri["metin"] = veri["ham_metin"].apply(turkce_metin_normalize_et)
    
    # Boş metinleri temizle
    veri = veri[veri["metin"].str.len() > 0].reset_index(drop=True)
    
    # Duygu etiketlerini oluştur
    if etiket_sutunu is not None:
        veri["duygu"] = etiketi_duyguya_donustur(veri[etiket_sutunu])
        veri = veri[veri["duygu"].isin([0, 1, 2])].copy()
        veri["duygu"] = veri["duygu"].astype(int)
    
    print(f"[BİLGİ] İşlenen yorum sayısı: {len(veri):,}")
    
    # DataFrame olduğundan emin ol
    if isinstance(veri, pd.Series):
        veri = veri.to_frame()
    return veri


# ============================================================
# KOMUT SATIRI
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VERİ ÖN İŞLEME")
    print("=" * 60)
    
    # Örnek kullanım
    ornek_metin = "HARIKA BİR ÜRÜN!!! www.site.com'dan aldım 😍😍😍"
    temiz = turkce_metin_normalize_et(ornek_metin)
    
    print(f"\nÖrnek Normalizasyon:")
    print(f"  Ham: {ornek_metin}")
    print(f"  Temiz: {temiz}")
