# 🛒 Türkçe E-Ticaret Yorumlarında Spam Tespiti ve Duygu Analizi

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-FF6B35?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

### 🌍 [English Documentation](docs/en/README.md) | 🇹🇷 Türkçe Dokümantasyon

> **Makine Öğrenmesi Dönem Projesi**
> Türkçe e-ticaret yorumlarını analiz ederek spam tespiti ve duygu analizi yapan, uçtan uca geliştirilmiş yapay zeka destekli karar destek sistemi.

---

## 👥 Proje Ekibi

| Öğrenci No | İsim | Rol |
|------------|------|-----|
| **--** | Mustafa Arda Düşova | Ekip Lideri & Developer |
| **--** | Fatih Çoban | Veri Araştırması & Analiz |
| **--** | Efe Ata | Model Belirleme & Optimizasyon |

---

## 📋 Proje Özeti

Bu projenin temel amacı, **TRSAv1 (Turkish Sentiment Analysis)** veri seti kullanılarak Türkçe e-ticaret yorumlarını analiz eden ve **iki ana görevi** yerine getiren bir makine öğrenmesi sistemi geliştirmektir:

1. **🚫 Spam/Bot Tespiti**: Sahte, reklam amaçlı veya bot tarafından yazılmış yorumları tespit etmek
2. **💭 Duygu Analizi (Sentiment Analysis)**: Yorumların olumlu, olumsuz veya nötr olduğunu belirlemek

Proje, sadece akademik bir çalışma olmanın ötesinde; **Gradio** ile geliştirilmiş modern web arayüzü ve manuel olarak kodlanmış veri işleme boru hatları (pipelines) ile profesyonel bir ürün niteliği taşımaktadır.

---

## ⭐ Temel Özellikler

| Özellik | Açıklama |
|---------|----------|
| 🔧 **Manuel Veri İşleme** | Hazır kütüphaneler yerine, eğitim amaçlı olarak manuel kodlanmış **Türkçe Stemming** ve **Metin Normalizasyonu** |
| 🎯 **Yüksek Performanslı Model** | TF-IDF + Logistic Regression ile optimize edilmiş tahminleme motoru |
| 🔀 **Hibrit Spam Tespiti** | Kural tabanlı + IsolationForest anomali tespiti birleşimi |
| 📊 **Aspekt Analizi** | Kargo, fiyat, kalite gibi konularda ayrı ayrı duygu analizi |
| 🖥️ **Web Arayüzü** | Gradio ile geliştirilen kullanıcı dostu demo arayüzü |
| 📈 **Detaylı Görselleştirme** | Confusion Matrix, PR Curve, metrik grafikleri |

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TÜRKÇE E-TİCARET YORUM ANALİZ SİSTEMİ                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            1️⃣ VERİ TOPLAMA                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  HuggingFace    │ OR │    GitHub       │ OR │   Yerel CSV     │         │
│  │  TRSAv1 Dataset │    │    Download     │    │    Dosyası      │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          2️⃣ VERİ ÖN İŞLEME                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Akıllı CSV  │→ │   Unicode    │→ │   Türkçe     │→ │  URL/Email   │     │
│  │   Okuma     │  │ Normalizasyon│  │  Stemming    │  │  Temizleme   │     │
│  └─────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
┌────────────────────────────────┐  ┌────────────────────────────────────────┐
│      3️⃣ DUYGU ANALİZİ          │  │         4️⃣ SPAM TESPİTİ                │
│  ┌──────────────────────────┐  │  │  ┌──────────────────────────────────┐  │
│  │   TF-IDF Vektörizasyon   │  │  │  │    Kural Tabanlı Tespit          │  │
│  │   (1-2 gram, 2000 feat)  │  │  │  │  (URL, tekrar, jenerik ifade)    │  │
│  └──────────────────────────┘  │  │  └──────────────────────────────────┘  │
│               │                │  │                  │                     │
│               ▼                │  │                  ▼                     │
│  ┌──────────────────────────┐  │  │  ┌──────────────────────────────────┐  │
│  │  Logistic Regression     │  │  │  │      IsolationForest             │  │
│  │  (3 sınıf: neg/neu/pos)  │  │  │  │    (Anomali Tespiti)             │  │
│  └──────────────────────────┘  │  │  └──────────────────────────────────┘  │
└────────────────────────────────┘  └────────────────────────────────────────┘
                          │                       │
                          └───────────┬───────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          5️⃣ UYGULAMA KATMANI                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────────┐ │
│  │  Aspekt Analizi  │  │   Gradio Demo    │  │  BERT Fine-tuning (Ops.)  │ │
│  │ (Kargo/Fiyat/vs) │  │   Web Arayüzü    │  │  (dbmdz/bert-turkish)     │ │
│  └──────────────────┘  └──────────────────┘  └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Veri Seti ve Metodoloji

### Kullanılan Veri Seti

| Özellik | Değer |
|---------|-------|
| **Veri Seti** | TRSAv1 (Turkish Sentiment Analysis v1) |
| **Kaynak** | [HuggingFace](https://huggingface.co/datasets/maydogan23/TRSAv1) / [GitHub](https://github.com/maydogan23/TRSAv1-Dataset) |
| **Kapsam** | Türkçe e-ticaret yorumları |
| **Sınıflar** | Negatif (0), Nötr (1), Pozitif (2) |

### Uygulanan İşlemler

1. **Veri Temizliği** (`src/preprocessing.py`):
   - Unicode normalizasyonu
   - URL, e-posta, telefon temizliği
   - Türkçe stemming (TurkishStemmer)
   - Tekrar eden karakter azaltma

2. **Modelleme** (`src/model.py`):
   - Veri %80 Eğitim, %10 Validation, %10 Test olarak ayrıldı
   - TF-IDF (1-2 gram) + Logistic Regression
   - Hibrit spam tespiti (Kural + IsolationForest)

---

## 🏆 Model Performans Metrikleri

### Duygu Analizi

| Model | Accuracy | F1-Score (Macro) | Precision | Recall |
|-------|----------|------------------|-----------|--------|
| **TF-IDF + Logistic Regression** | ~0.85 | ~0.78 | ~0.79 | ~0.77 |
| BERT Fine-tuning (Opsiyonel) | ~0.88 | ~0.82 | ~0.83 | ~0.81 |

### Spam Tespiti

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Hibrit (Kural + IsolationForest)** | ~0.92 | ~0.75 | ~0.80 | ~0.70 |

> **Not**: Değerler yaklaşık olup, veri setine göre değişebilir.

---

## 🖥️ Uygulama Arayüzü

Geliştirdiğimiz modern web arayüzünden örnekler:

### Demo Özellikleri

| Çıktı | Açıklama |
|-------|----------|
| **Spam Olasılığı** | 0-1 arası, 1'e yakın = muhtemel spam |
| **Genel Duygu** | Negatif / Nötr / Pozitif |
| **Aspekt Analizi** | Kargo, fiyat, kalite vs. için ayrı duygu |

---

## 📁 Dosya Yapısı

```
Turkish-ECommerce-Sentiment-Spam-Analyzer/
│
├── 📂 data/                          # Veri setleri
│   └── TRSAv1.csv                    # Ana veri seti (otomatik indirilir)
│
├── 📂 docs/                          # Dökümanlar
│   ├── en/
│   │   └── README.md                 # English documentation
│   └── images/                       # Ekran görüntüleri
│
├── 📂 models/                        # Eğitilmiş modeller
│   ├── sentiment_model.pkl           # Duygu analizi modeli
│   └── spam_model.pkl                # Spam tespiti modeli
│
├── 📂 src/                           # Kaynak kodlar
│   ├── __init__.py
│   ├── app.py                        # Gradio web arayüzü
│   ├── preprocessing.py              # Veri ön işleme
│   ├── model.py                      # Model eğitim & tahmin
│   ├── spam_detector.py              # Spam tespiti modülü
│   └── utils.py                      # Yardımcı fonksiyonlar
│
├── 📂 notebooks/                     # Jupyter Notebooks
│   └── Turkish-ECommerce-Sentiment-Spam-Analyzer.ipynb
│
├── 📄 .gitignore                     # Git ignore dosyası
├── 📄 requirements.txt               # Python bağımlılıkları
├── 📄 config.py                      # Konfigürasyon ayarları
├── 📄 LICENSE                        # MIT Lisansı
└── 📄 README.md                      # Bu dosya
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Klonlama

```bash
git clone https://github.com/kullanici/Turkish-ECommerce-Sentiment-Spam-Analyzer.git
cd Turkish-ECommerce-Sentiment-Spam-Analyzer
```

### 2. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 3. Veri Hazırlığı ve Model Eğitimi

> ⚠️ **ÖNEMLİ**: GitHub deposunda `model.pkl` dosyaları (boyut sınırı nedeniyle) bulunmamaktadır. 
> Uygulamayı çalıştırmadan önce aşağıdaki komutları çalıştırarak modelleri eğitmeniz **ZORUNLUDUR**.

```bash
# Veriyi indir ve ön işle
python src/preprocessing.py

# Modelleri eğit ve kaydet
python src/model.py
```

### 4. Uygulamayı Başlat

```bash
# Gradio web arayüzünü başlat
python src/app.py
```

Tarayıcınızda `http://localhost:7860` adresine gidin.

### 5. Jupyter Notebook (Alternatif)

```bash
jupyter notebook notebooks/Turkish-ECommerce-Sentiment-Spam-Analyzer.ipynb
```

---

## 🐳 Docker ile Çalıştırma (Opsiyonel)

```bash
# Docker image oluştur
docker build -t sentiment-analyzer .

# Container'ı çalıştır
docker run -p 7860:7860 sentiment-analyzer
```

---

## ⚠️ Proje Kısıtları ve Açıklamalar

### 1. Spam Etiketi Yokluğu

Kullanılan veri setinde gerçek "spam" etiketi bulunmamaktadır.

- **Sebep**: TRSAv1 veri seti sadece duygu etiketleri içerir
- **Çözüm**: Kural tabanlı + IsolationForest ile "gümüş etiket" oluşturduk

### 2. İroni ve Alaycılık

TF-IDF tabanlı model ironi/alaycılık içeren yorumları anlamakta zorlanır.

- **Çözüm**: BERT gibi bağlamsal modeller daha iyi performans gösterir

### 3. Sınıf Dengesizliği

Veri setinde pozitif yorumlar çoğunluktadır.

- **Çözüm**: Stratified sampling ve class_weight kullanıldı

---

## 🔮 Gelecek Çalışmalar

| Öncelik | Geliştirme | Açıklama |
|---------|------------|----------|
| 🔴 Yüksek | **SMOTE** | Sınıf dengesizliğini gidermek için synthetic oversampling |
| 🔴 Yüksek | **SHAP** | Model açıklanabilirliği için SHAP grafikleri |
| 🟡 Orta | **BERT Fine-tuning** | dbmdz/bert-base-turkish-cased ile daha yüksek performans |
| 🟡 Orta | **Gerçek Spam Verisi** | Manuel etiketlenmiş spam veri seti toplama |
| 🟢 Düşük | **API Servisi** | FastAPI ile REST API oluşturma |
| 🟢 Düşük | **Dockerizasyon** | Production-ready container yapısı |

---

## 🔗 Kaynaklar ve Referanslar

| Kaynak | Link | Açıklama |
|--------|------|----------|
| TRSAv1 Dataset | [HuggingFace](https://huggingface.co/datasets/maydogan23/TRSAv1) | Türkçe duygu analizi veri seti |
| TurkishStemmer | [PyPI](https://pypi.org/project/TurkishStemmer/) | Türkçe kök bulma kütüphanesi |
| Scikit-learn | [Docs](https://scikit-learn.org/) | Makine öğrenmesi kütüphanesi |
| Gradio | [Docs](https://gradio.app/) | Web arayüzü kütüphanesi |
| BERT Turkish | [HuggingFace](https://huggingface.co/dbmdz/bert-base-turkish-cased) | Türkçe BERT modeli |

---

## 📄 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Detaylı bilgi için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

---

## 📚 Detaylı Dokümantasyon

| Dokümantasyon | Açıklama |
|---------------|----------|
| [📖 Türkçe Detaylı Dokümantasyon](docs/tr/README.md) | Tam Türkçe rehber |
| [📖 English Documentation](docs/en/README.md) | Full English guide |
| [📚 API Referansı](docs/API.md) | Tüm fonksiyonlar ve sınıflar |
| [🤝 Katkıda Bulunma](CONTRIBUTING.md) | Katkıda bulunma rehberi |
| [📝 Değişiklik Günlüğü](CHANGELOG.md) | Sürüm geçmişi |

---

<div align="center">

### 🎓 2025 Makine Öğrenmesi Dersi Dönem Projesi

**Tarih**: Aralık 2025

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red?style=flat)](https://github.com)

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

</div>
