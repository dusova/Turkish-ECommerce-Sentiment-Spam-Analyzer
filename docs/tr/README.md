# 🛒 Türkçe E-Ticaret Yorumlarında Spam Tespiti ve Duygu Analizi

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-FF6B35?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](../../LICENSE)

### 🌍 [English Documentation](../en/README.md) | 🇹🇷 Türkçe Dokümantasyon

**Makine Öğrenmesi Dönem Projesi** - Aralık 2025

</div>

---

## 📑 İçindekiler

1. [Proje Hakkında](#-proje-hakkında)
2. [Proje Ekibi](#-proje-ekibi)
3. [Temel Özellikler](#-temel-özellikler)
4. [Sistem Mimarisi](#-sistem-mimarisi)
5. [Teknoloji Yığını](#-teknoloji-yığını)
6. [Veri Seti](#-veri-seti)
7. [Metodoloji](#-metodoloji)
8. [Model Performansı](#-model-performansı)
9. [Kurulum](#-kurulum)
10. [Kullanım](#-kullanım)
11. [API Referansı](#-api-referansı)
12. [Dosya Yapısı](#-dosya-yapısı)
13. [Örnek Analizler](#-örnek-analizler)
14. [Kısıtlamalar](#-kısıtlamalar)
15. [Gelecek Geliştirmeler](#-gelecek-geliştirmeler)
16. [Katkıda Bulunma](#-katkıda-bulunma)
17. [Lisans](#-lisans)
18. [İletişim](#-iletişim)

---

## 📖 Proje Hakkında

Bu proje, **Makine Öğrenmesi dersi dönem projesi** kapsamında geliştirilmiş, Türkçe e-ticaret yorumlarını analiz eden yapay zeka destekli bir karar destek sistemidir.

### 🎯 Problem Tanımı

E-ticaret platformlarında sahte (spam/bot) yorumlar ve gerçek müşteri yorumlarının ayırt edilmesi kritik bir sorundur:

- **Tüketiciler için**: Yanıltıcı yorumlar nedeniyle yanlış kararlar
- **Satıcılar için**: Haksız rekabet ve itibar kaybı
- **Platformlar için**: Güvenilirlik ve kullanıcı deneyimi sorunları

### 💡 Çözüm Yaklaşımı

Geliştirdiğimiz sistem **iki ana görevi** yerine getirir:

| Görev | Açıklama | Yöntem |
|-------|----------|--------|
| **🚫 Spam Tespiti** | Sahte, reklam amaçlı veya bot yorumlarını tespit | Hibrit (Kural + IsolationForest) |
| **💭 Duygu Analizi** | Yorumların olumlu/olumsuz/nötr olduğunu belirleme | TF-IDF + Logistic Regression |

---

## 👥 Proje Ekibi

<div align="center">

| Fotoğraf | Öğrenci No | İsim | Rol | Sorumluluklar |
|:--------:|:----------:|:----:|:---:|:--------------|
| 👤 | **--** | **Mustafa Arda Düşova** | Ekip Lideri & Geliştirici | Proje yönetimi, kod geliştirme, entegrasyon |
| 👤 | **--** | **Fatih Çoban** | Veri Araştırması & Analiz | Veri seti araştırması, EDA, görselleştirme |
| 👤 | **--** | **Efe Ata** | Model Belirleme & Optimizasyon | Model seçimi, hiperparametre ayarı |

</div>

**Danışman**: [Danışman Adı]  
**Ders**: Makine Öğrenmesi  
**Dönem**: 2024-2025 Güz

---

## ⭐ Temel Özellikler

### 🔧 Manuel Veri İşleme
Eğitim amaçlı olarak hazır kütüphaneler yerine **sıfırdan kodlanmış** işleme boru hatları:
- Türkçe metin normalizasyonu
- Unicode temizliği
- TurkishStemmer ile kök bulma

### 🎯 Yüksek Performanslı Tahminleme
- **TF-IDF + Logistic Regression** tabanlı duygu analizi
- **%85+** doğruluk oranı
- Hızlı ve hafif model

### 🔀 Hibrit Spam Tespiti
İki yaklaşımın güçlü yönlerini birleştiren sistem:

```
┌─────────────────────────────────────────┐
│            HİBRİT SPAM TESPİT           │
├─────────────────┬───────────────────────┤
│  Kural Tabanlı  │   IsolationForest     │
│  (%60 ağırlık)  │   (%40 ağırlık)       │
├─────────────────┼───────────────────────┤
│ • URL tespiti   │ • Anomali tespiti     │
│ • Emoji sayısı  │ • TF-IDF özellikler   │
│ • Tekrar kalıp  │ • İstatistiksel       │
│ • Jenerik ifade │   aykırı değer        │
└─────────────────┴───────────────────────┘
```

### 📊 Aspekt Bazlı Analiz
Yorumlardaki farklı konuları ayrı ayrı analiz eder:

| Aspekt | Anahtar Kelimeler |
|--------|-------------------|
| 📦 Kargo | kargo, teslimat, paket, kutu |
| 💰 Fiyat | fiyat, ucuz, pahalı, eder |
| ⭐ Kalite | kalite, malzeme, sağlam |
| 📞 Müşteri Hizmetleri | destek, iade, iletişim |

### 🖥️ Web Arayüzü
Gradio ile geliştirilen kullanıcı dostu demo:
- Gerçek zamanlı analiz
- Görsel sonuç gösterimi
- Kolay entegrasyon

---

## 🏗️ Sistem Mimarisi

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    TÜRKÇE E-TİCARET YORUM ANALİZ SİSTEMİ                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                        📥 1. VERİ TOPLAMA                               │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐             │ ║
║  │  │  HuggingFace  │   │    GitHub     │   │  Yerel CSV    │             │ ║
║  │  │    Dataset    │   │   Download    │   │    Upload     │             │ ║
║  │  │   (Öncelik)   │   │   (Yedek)     │   │  (Alternatif) │             │ ║
║  │  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘             │ ║
║  │          └───────────────────┼───────────────────┘                      │ ║
║  │                              ▼                                          │ ║
║  │                    ┌─────────────────┐                                  │ ║
║  │                    │  TRSAv1 Dataset │                                  │ ║
║  │                    │  (~60K+ yorum)  │                                  │ ║
║  │                    └────────┬────────┘                                  │ ║
║  └─────────────────────────────┼───────────────────────────────────────────┘ ║
║                                ▼                                             ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                     🔧 2. VERİ ÖN İŞLEME                                │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                         │ ║
║  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ ║
║  │  │ Akıllı   │→ │ Unicode  │→ │ Küçük    │→ │ URL/     │→ │ Türkçe   │  │ ║
║  │  │ CSV Oku  │  │ Normalize│  │ Harf     │  │ Email    │  │ Stemming │  │ ║
║  │  │          │  │          │  │ Çevir    │  │ Temizle  │  │          │  │ ║
║  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │ ║
║  │                                                                         │ ║
║  │  Ek İşlemler:                                                           │ ║
║  │  • Tekrar eden karakterleri azaltma ("çooook" → "çook")                │ ║
║  │  • Sütun otomatik tespiti (metin, etiket, puan)                        │ ║
║  │  • Encoding tespiti (UTF-8, Latin1, Windows-1254)                      │ ║
║  │                                                                         │ ║
║  └──────────────────────────────┬──────────────────────────────────────────┘ ║
║                                 │                                            ║
║              ┌──────────────────┴──────────────────┐                        ║
║              ▼                                      ▼                        ║
║  ┌──────────────────────────┐      ┌──────────────────────────────────────┐ ║
║  │  💭 3. DUYGU ANALİZİ     │      │      🚫 4. SPAM TESPİTİ              │ ║
║  ├──────────────────────────┤      ├──────────────────────────────────────┤ ║
║  │                          │      │                                      │ ║
║  │  ┌────────────────────┐  │      │  ┌────────────────────────────────┐  │ ║
║  │  │ TF-IDF Vektörizer  │  │      │  │      KURAL TABANLI (%60)       │  │ ║
║  │  │ • 1-2 gram         │  │      │  ├────────────────────────────────┤  │ ║
║  │  │ • Min DF: 2        │  │      │  │ • URL/email/telefon tespiti    │  │ ║
║  │  │ • Max DF: 0.9      │  │      │  │ • Ünlem sayısı (>=4 → spam)    │  │ ║
║  │  └─────────┬──────────┘  │      │  │ • Emoji sayısı (>=3 → spam)    │  │ ║
║  │            ▼             │      │  │ • Büyük harf oranı (>0.6)      │  │ ║
║  │  ┌────────────────────┐  │      │  │ • Jenerik ifade kontrolü       │  │ ║
║  │  │ Logistic Regress.  │  │      │  │ • Kısa + genel yorum tespiti   │  │ ║
║  │  │ • Max iter: 1000   │  │      │  └─────────────┬──────────────────┘  │ ║
║  │  │ • 3 sınıf          │  │      │                ▼                     │ ║
║  │  └─────────┬──────────┘  │      │  ┌────────────────────────────────┐  │ ║
║  │            ▼             │      │  │   ISOLATION FOREST (%40)       │  │ ║
║  │  ┌────────────────────┐  │      │  ├────────────────────────────────┤  │ ║
║  │  │ Çıktı:             │  │      │  │ • 100 estimator                │  │ ║
║  │  │ 0: Negatif 😞      │  │      │  │ • %5 contamination             │  │ ║
║  │  │ 1: Nötr 😐         │  │      │  │ • TF-IDF tabanlı özellikler    │  │ ║
║  │  │ 2: Pozitif 😊      │  │      │  │ • Anomali skoru hesaplama      │  │ ║
║  │  └────────────────────┘  │      │  └─────────────┬──────────────────┘  │ ║
║  │                          │      │                ▼                     │ ║
║  │                          │      │  ┌────────────────────────────────┐  │ ║
║  │                          │      │  │  HİBRİT KARAR                  │  │ ║
║  │                          │      │  │  • 0: Gerçek Yorum             │  │ ║
║  │                          │      │  │  • 1: Spam/Bot                 │  │ ║
║  │                          │      │  │  • -1: Belirsiz                │  │ ║
║  │                          │      │  └────────────────────────────────┘  │ ║
║  └──────────┬───────────────┘      └───────────────────┬──────────────────┘ ║
║             │                                          │                     ║
║             └─────────────────┬─────────────────────────┘                    ║
║                               ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                      🖥️ 5. UYGULAMA KATMANI                            │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │ ║
║  │  │  Gradio Web  │  │   Aspekt     │  │   BERT       │  │  REST API   │  │ ║
║  │  │   Arayüzü    │  │   Analizi    │  │ Fine-tuning  │  │  (Gelecek)  │  │ ║
║  │  │   (Demo)     │  │   (Detay)    │  │ (Opsiyonel)  │  │             │  │ ║
║  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🛠️ Teknoloji Yığını

### Programlama Dili
| Teknoloji | Versiyon | Amaç |
|-----------|----------|------|
| Python | 3.8+ | Ana geliştirme dili |

### Makine Öğrenmesi
| Kütüphane | Versiyon | Kullanım |
|-----------|----------|----------|
| scikit-learn | 1.3+ | Model eğitim ve değerlendirme |
| numpy | 1.24+ | Sayısal hesaplamalar |
| pandas | 2.0+ | Veri manipülasyonu |

### Doğal Dil İşleme
| Kütüphane | Versiyon | Kullanım |
|-----------|----------|----------|
| TurkishStemmer | 1.3+ | Türkçe kök bulma |
| transformers | 4.30+ | BERT modeli (opsiyonel) |

### Görselleştirme
| Kütüphane | Versiyon | Kullanım |
|-----------|----------|----------|
| matplotlib | 3.7+ | Grafik oluşturma |
| seaborn | 0.12+ | İstatistiksel görselleştirme |

### Web Arayüzü
| Kütüphane | Versiyon | Kullanım |
|-----------|----------|----------|
| gradio | 4.0+ | Demo arayüzü |

---

## 📊 Veri Seti

### TRSAv1 (Turkish Sentiment Analysis v1)

| Özellik | Değer |
|---------|-------|
| **Kaynak** | [HuggingFace](https://huggingface.co/datasets/maydogan23/TRSAv1) / [GitHub](https://github.com/maydogan23/TRSAv1-Dataset) |
| **Boyut** | ~60,000+ yorum |
| **Dil** | Türkçe |
| **Kapsam** | E-ticaret platformlarından toplanan yorumlar |
| **Sınıflar** | Negatif (0), Nötr (1), Pozitif (2) |
| **Format** | CSV |

### Sınıf Dağılımı

```
Pozitif  ████████████████████████████████████████ 45%
Nötr     ████████████████████ 25%
Negatif  ████████████████████████ 30%
```

### Örnek Veriler

| Yorum | Duygu |
|-------|-------|
| "Ürün harika geldi, çok memnun kaldım" | Pozitif |
| "Kargo biraz geç geldi ama ürün güzel" | Nötr |
| "Berbat kalite, kesinlikle tavsiye etmem" | Negatif |

---

## 📐 Metodoloji

### 1. Veri Ön İşleme Pipeline

```python
def turkce_metin_normalize_et(metin):
    """
    Adımlar:
    1. Unicode normalizasyonu (NFKC)
    2. Küçük harfe çevirme
    3. URL → <url> etiketi
    4. Email → <email> etiketi
    5. Telefon → <phone> etiketi
    6. Tekrar eden karakterleri azaltma
    7. TurkishStemmer ile kök bulma
    """
```

### 2. Özellik Çıkarımı

**TF-IDF (Term Frequency - Inverse Document Frequency)**

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| ngram_range | (1, 2) | Unigram ve bigram |
| min_df | 2 | En az 2 belgede geçmeli |
| max_df | 0.9 | En fazla %90 belgede geçmeli |
| max_features | None | Sınırsız özellik |

### 3. Model Eğitimi

#### Duygu Analizi
- **Algoritma**: Logistic Regression
- **Bölme Oranı**: %80 Eğitim, %10 Doğrulama, %10 Test
- **Optimizasyon**: L2 regularization

#### Spam Tespiti
- **Hibrit Yaklaşım**:
  - Kural tabanlı skor (%60 ağırlık)
  - IsolationForest anomali skoru (%40 ağırlık)

---

## 📈 Model Performansı

### Duygu Analizi Metrikleri

| Model | Accuracy | F1 (Macro) | Precision | Recall |
|-------|----------|------------|-----------|--------|
| **TF-IDF + LR** | ~0.85 | ~0.78 | ~0.79 | ~0.77 |
| BERT (Opsiyonel) | ~0.88 | ~0.82 | ~0.83 | ~0.81 |

### Sınıf Bazlı Performans

| Sınıf | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negatif | 0.81 | 0.79 | 0.80 |
| Nötr | 0.72 | 0.70 | 0.71 |
| Pozitif | 0.85 | 0.88 | 0.86 |

### Spam Tespiti Metrikleri

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Hibrit** | ~0.92 | ~0.75 | ~0.80 | ~0.70 |

> **Not**: Değerler yaklaşık olup, farklı veri setlerinde değişebilir.

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- pip paket yöneticisi
- (Opsiyonel) GPU - BERT için

### Adım 1: Repoyu Klonlayın

```bash
git clone https://github.com/dusova/Turkish-ECommerce-Sentiment-Spam-Analyzer.git
cd Turkish-ECommerce-Sentiment-Spam-Analyzer
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### Adım 4: Modelleri Eğitin

```bash
# Veri ön işleme (veri setini indirir)
python src/preprocessing.py

# Duygu analizi modelini eğit
python src/model.py

# Spam tespit modelini eğit
python src/spam_detector.py
```

### Adım 5: Web Arayüzünü Başlatın

```bash
python src/app.py
```

---

## 💻 Kullanım

### Python ile Programatik Kullanım

```python
from src.preprocessing import turkce_metin_normalize_et
from src.model import SentimentModel
from src.spam_detector import SpamDetector

# Modelleri yükle
duygu_modeli = SentimentModel.load("models/sentiment_model.pkl")
spam_modeli = SpamDetector.load("models/spam_model.pkl")

# Yorum analiz et
yorum = "Bu ürün gerçekten harika, çok beğendim!"
normalize = turkce_metin_normalize_et(yorum)

# Duygu tahmini
duygu = duygu_modeli.predict([normalize])[0]
print(f"Duygu: {['Negatif', 'Nötr', 'Pozitif'][duygu]}")

# Spam kontrolü
spam_sonuc = spam_modeli.analyze(yorum, normalize)
print(f"Spam Olasılığı: {spam_sonuc['spam_olasiligi']:.1%}")
```

### Google Colab ile Kullanım

1. Notebook'u Colab'da açın
2. `Runtime > Run all` ile tüm hücreleri çalıştırın
3. Gradio demo linkine tıklayın

---

## 📚 API Referansı

### `turkce_metin_normalize_et(metin, stemming_uygula=True)`

Türkçe metni normalize eder.

**Parametreler:**
- `metin` (str): Ham metin
- `stemming_uygula` (bool): Stemming uygulanacak mı

**Dönüş:** str - Normalize edilmiş metin

---

### `SentimentModel`

Duygu analizi model sınıfı.

**Metodlar:**
- `fit(X, y)`: Model eğitir
- `predict(X)`: Tahmin yapar
- `predict_proba(X)`: Olasılık döndürür
- `save(path)`: Modeli kaydeder
- `load(path)`: Model yükler

---

### `SpamDetector`

Spam tespit sınıfı.

**Metodlar:**
- `fit(ham_metinler, normalize_metinler)`: Model eğitir
- `analyze(ham_metin, normalize_metin)`: Tek yorum analiz eder
- `predict(metinler)`: Toplu tahmin yapar

---

## 📁 Dosya Yapısı

```
Turkish-ECommerce-Sentiment-Spam-Analyzer/
│
├── 📂 data/                          # Veri setleri
│   └── TRSAv1.csv                    # Ana veri (otomatik indirilir)
│
├── 📂 docs/                          # Dokümantasyon
│   ├── tr/
│   │   └── README.md                 # Türkçe dokümantasyon (bu dosya)
│   ├── en/
│   │   └── README.md                 # İngilizce dokümantasyon
│   └── images/                       # Görseller
│
├── 📂 models/                        # Eğitilmiş modeller
│   ├── sentiment_model.pkl           # Duygu modeli
│   └── spam_model.pkl                # Spam modeli
│
├── 📂 notebooks/                     # Jupyter Notebooks
│   └── Turkish-ECommerce-...ipynb    # Ana notebook
│
├── 📂 src/                           # Kaynak kodlar
│   ├── __init__.py                   # Paket init
│   ├── app.py                        # Gradio arayüzü
│   ├── model.py                      # Duygu modeli
│   ├── preprocessing.py              # Veri işleme
│   ├── spam_detector.py              # Spam tespiti
│   └── utils.py                      # Yardımcı fonksiyonlar
│
├── 📄 config.py                      # Konfigürasyon
├── 📄 requirements.txt               # Bağımlılıklar
├── 📄 LICENSE                        # MIT Lisansı
└── 📄 README.md                      # Ana README
```

---

## 🔍 Örnek Analizler

### Örnek 1: Pozitif Yorum

**Girdi:**
```
"Ürün beklenenden çok daha iyi çıktı. Kargo hızlıydı, paketleme özenli. 
Kesinlikle tavsiye ederim, 5 yıldız hak ediyor!"
```

**Çıktı:**
```json
{
  "duygu": "Pozitif",
  "güven": "94%",
  "spam_olasiligi": "12%",
  "spam_etiket": "Gerçek",
  "aspektler": {
    "kargo": "Pozitif",
    "kalite": "Pozitif"
  }
}
```

### Örnek 2: Spam Yorum

**Girdi:**
```
"MUKEMMEL URUN!!! EN IYISI BU!!! www.sahtesite.com TIKLAYIN KAÇIRMAYIN!!!"
```

**Çıktı:**
```json
{
  "duygu": "Pozitif",
  "güven": "72%",
  "spam_olasiligi": "89%",
  "spam_etiket": "Spam",
  "spam_nedenleri": [
    "URL tespit edildi",
    "Aşırı ünlem kullanımı",
    "Jenerik ifadeler"
  ]
}
```

---

## ⚠️ Kısıtlamalar

1. **Dil**: Sadece Türkçe yorumları destekler
2. **Alan**: E-ticaret yorumları için optimize edilmiştir
3. **Veri Dengesi**: Nötr sınıfta performans düşüşü yaşanabilir
4. **İroni/Alay**: Algılama kapasitesi sınırlıdır
5. **Bağlam**: Kısa yorumlarda yetersiz bağlam sorunu

---

## 🔮 Gelecek Geliştirmeler

- [ ] Çok dilli destek (İngilizce, Arapça)
- [ ] REST API entegrasyonu
- [ ] Gerçek zamanlı streaming analiz
- [ ] Daha gelişmiş BERT modeli
- [ ] İroni/alay tespiti modülü
- [ ] Docker konteyner desteği
- [ ] Web dashboard

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen [CONTRIBUTING.md](../../CONTRIBUTING.md) dosyasını inceleyin.

---

## 📄 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır.

---

## 📞 İletişim

**Mustafa Arda Düşova**
- GitHub: [@dusova](https://github.com/dusova)
- Email: [arda@codewithmad.com]

---

<div align="center">

### 🎓 2025 Makine Öğrenmesi Dönem Projesi

**Tarih**: Aralık 2025

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red?style=flat)](https://github.com)

</div>
