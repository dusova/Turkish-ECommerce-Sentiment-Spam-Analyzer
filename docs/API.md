# 📚 API Dokümantasyonu / API Documentation

<div align="center">

**[🇹🇷 Türkçe](#türkçe-dokümantasyon) | [🇬🇧 English](#english-documentation)**

</div>

---

# Türkçe Dokümantasyon

Bu dokümantasyon, Türkçe E-Ticaret Yorum Analizi projesinin tüm modüllerini ve fonksiyonlarını detaylı şekilde açıklar.

## İçindekiler

1. [src.preprocessing](#srcpreprocessing)
2. [src.model](#srcmodel)
3. [src.spam_detector](#srcspam_detector)
4. [src.utils](#srcutils)
5. [src.app](#srcapp)
6. [config](#config)

---

## src.preprocessing

Metin normalizasyonu ve veri temizleme işlemlerini içeren modül.

### Fonksiyonlar

#### `turkce_metin_normalize_et(metin, stemming_uygula=True)`

Türkçe metni makine öğrenmesi için uygun hale getirir.

**Parametreler:**
| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `metin` | str | - | Ham metin |
| `stemming_uygula` | bool | True | Stemming uygulanacak mı? |

**Dönüş:** `str` - Normalize edilmiş metin

**Uygulanan İşlemler:**
1. Unicode normalizasyonu (NFKC)
2. Küçük harfe çevirme
3. URL → `<url>` etiketi
4. E-posta → `<email>` etiketi
5. Telefon → `<phone>` etiketi
6. Tekrar eden karakterleri azaltma
7. TurkishStemmer ile kök bulma

**Örnek:**
```python
from src.preprocessing import turkce_metin_normalize_et

ham = "HARIKA BİR ÜRÜN!!! www.site.com 😍😍😍"
temiz = turkce_metin_normalize_et(ham)
print(temiz)  # "harik bir ürün <url>"
```

---

#### `akilli_csv_oku(dosya_yolu)`

CSV dosyasını akıllı şekilde okur (encoding ve ayırıcı otomatik tespit).

**Parametreler:**
| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `dosya_yolu` | str | CSV dosyasının yolu |

**Dönüş:** `pd.DataFrame`

**Desteklenen Encodingler:**
- UTF-8-sig
- UTF-8
- Latin1
- Windows-1254

**Örnek:**
```python
from src.preprocessing import akilli_csv_oku

veri = akilli_csv_oku("data/yorumlar.csv")
print(veri.shape)  # (60000, 3)
```

---

#### `metin_sutunu_bul(veri_cercevesi)`

DataFrame'de metin sütununu otomatik tespit eder.

**Parametreler:**
| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `veri_cercevesi` | pd.DataFrame | Pandas DataFrame |

**Dönüş:** `str | None` - Bulunan sütun adı

---

#### `etiket_sutunu_bul(veri_cercevesi)`

DataFrame'de duygu etiketi sütununu otomatik tespit eder.

**Parametreler:**
| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `veri_cercevesi` | pd.DataFrame | Pandas DataFrame |

**Dönüş:** `Tuple[str | None, str | None]` - (sütun_adı, tespit_yöntemi)

---

#### `veri_hazirla(dosya_yolu=None)`

Veri setini indirir/okur ve ön işleme uygular.

**Parametreler:**
| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `dosya_yolu` | str | None | CSV dosya yolu (None ise otomatik indirir) |

**Dönüş:** `pd.DataFrame` - Hazırlanmış veri

**Çıktı DataFrame Sütunları:**
- `ham_metin`: Orijinal yorum
- `metin`: Normalize edilmiş metin
- `duygu`: Duygu etiketi (0, 1, 2)

---

## src.model

Duygu analizi modelinin eğitimi ve kullanımını içeren modül.

### Sınıflar

#### `SentimentModel`

TF-IDF + Logistic Regression tabanlı duygu analizi modeli.

**Özellikler:**
| Özellik | Tip | Açıklama |
|---------|-----|----------|
| `pipeline` | Pipeline | Sklearn Pipeline |
| `classes` | dict | Sınıf isimleri |
| `is_trained` | bool | Model eğitildi mi? |

**Metodlar:**

##### `__init__(tfidf_config=None)`
Model oluşturur.

```python
from src.model import SentimentModel

model = SentimentModel()
# veya özel config ile
model = SentimentModel(tfidf_config={
    "ngram_range": (1, 3),
    "max_features": 5000
})
```

##### `fit(X, y)`
Modeli eğitir.

**Parametreler:**
| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `X` | List[str] | Metin listesi |
| `y` | List[int] | Etiket listesi (0, 1, 2) |

**Dönüş:** `SentimentModel` - Eğitilmiş model

```python
model.fit(X_train, y_train)
```

##### `predict(X)`
Tahmin yapar.

**Parametreler:**
| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `X` | List[str] | Metin listesi |

**Dönüş:** `np.ndarray` - Tahmin edilen sınıflar

```python
tahminler = model.predict(["Harika ürün!", "Berbat kalite"])
print(tahminler)  # [2, 0]
```

##### `predict_proba(X)`
Olasılık tahmini yapar.

**Dönüş:** `np.ndarray` - Shape: (n_samples, 3)

```python
olasiliklar = model.predict_proba(["Harika ürün!"])
print(olasiliklar)  # [[0.05, 0.10, 0.85]]
```

##### `evaluate(X, y)`
Model performansını değerlendirir.

**Dönüş:** `dict` - Metrikler

```python
metrikler = model.evaluate(X_test, y_test)
print(metrikler["accuracy"])  # 0.85
```

##### `save(path)`
Modeli dosyaya kaydeder.

```python
model.save("models/sentiment_model.pkl")
```

##### `load(path)` (classmethod)
Modeli dosyadan yükler.

```python
model = SentimentModel.load("models/sentiment_model.pkl")
```

---

## src.spam_detector

Spam/bot yorum tespiti işlemlerini içeren modül.

### Sınıflar

#### `SpamDetector`

Hibrit spam tespit sistemi (Kural + IsolationForest).

**Özellikler:**
| Özellik | Tip | Varsayılan | Açıklama |
|---------|-----|------------|----------|
| `rule_weight` | float | 0.6 | Kural ağırlığı |
| `anomaly_weight` | float | 0.4 | Anomali ağırlığı |
| `is_trained` | bool | False | Model eğitildi mi? |

**Metodlar:**

##### `__init__(rule_weight=0.6, anomaly_weight=0.4)`

```python
from src.spam_detector import SpamDetector

detector = SpamDetector()
# veya özel ağırlıklarla
detector = SpamDetector(rule_weight=0.7, anomaly_weight=0.3)
```

##### `kural_tabanli_skor(ham_metin, normalize_metin)`
Kural tabanlı spam skoru hesaplar.

**Dönüş:** `int` - 1 (spam), 0 (gerçek), -1 (belirsiz)

**Kontrol Edilen Kurallar:**
- URL/email/telefon varlığı (+3 puan)
- Ünlem sayısı >= 4 (+1 puan)
- Emoji sayısı >= 3 (+1 puan)
- Büyük harf oranı > 0.6 (+1 puan)
- Kısa + jenerik ifade (+2 puan)

##### `fit(ham_metinler, normalize_metinler)`
Spam tespit modelini eğitir.

```python
detector.fit(ham_listesi, normalize_listesi)
```

##### `analyze(ham_metin, normalize_metin=None)`
Tek bir yorumu analiz eder.

**Dönüş:** `dict`

```python
sonuc = detector.analyze("MUKEMMEL!!! www.site.com")
print(sonuc)
# {
#   "spam_olasiligi": 0.87,
#   "tahmin": 1,
#   "etiket": "Spam",
#   "kural_skoru": 1,
#   "aciklama": "URL tespit edildi, aşırı ünlem..."
# }
```

##### `predict(metinler)`
Toplu tahmin yapar.

```python
tahminler = detector.predict(normalize_listesi)
```

##### `save(path)` / `load(path)`
Model kaydetme ve yükleme.

---

## src.utils

Yardımcı fonksiyonları içeren modül.

### Fonksiyonlar

#### `veri_indir(hedef_yol=None, kaynak="auto")`

Veri setini indirir veya yerel dosyadan okur.

**Parametreler:**
| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `hedef_yol` | str | None | İndirme hedef yolu |
| `kaynak` | str | "auto" | "huggingface", "github", "local" veya "auto" |

**Dönüş:** `pd.DataFrame`

**Öncelik Sırası (auto mod):**
1. Yerel dosya
2. HuggingFace
3. GitHub

```python
from src.utils import veri_indir

veri = veri_indir()  # Otomatik indirir
veri = veri_indir(kaynak="huggingface")  # Sadece HuggingFace
```

#### `model_kaydet(model, yol)`
Modeli pickle formatında kaydeder.

#### `model_yukle(yol)`
Modeli pickle formatından yükler.

---

## src.app

Gradio web arayüzü modülü.

### Fonksiyonlar

#### `analiz_yap(yorum)`

Bir yorumu tam analiz eder.

**Parametreler:**
| Parametre | Tip | Açıklama |
|-----------|-----|----------|
| `yorum` | str | Yorum metni |

**Dönüş:** `dict` - Analiz sonuçları

```python
from src.app import analiz_yap

sonuc = analiz_yap("Ürün harika, kargo hızlıydı!")
print(sonuc)
# {
#   "girdi": "Ürün harika, kargo hızlıydı!",
#   "normalize": "ürün harik kargo hızl",
#   "spam_analizi": {"olasilik": "12%", "etiket": "Gerçek"},
#   "duygu_analizi": {"genel_duygu": "Pozitif", "olasiliklar": {...}},
#   "aspekt_analizi": {"Kargo": "Pozitif", "Kalite": "Pozitif"}
# }
```

#### `aspekt_analizi(yorum)`
Aspekt bazlı duygu analizi yapar.

### Arayüzü Başlatma

```python
from src.app import demo

# Yerel sunucu
demo.launch()

# Paylaşımlı link ile
demo.launch(share=True)

# Belirli port
demo.launch(server_port=7860)
```

---

## config

Merkezi konfigürasyon dosyası.

### Yol Sabitleri

| Sabit | Tip | Açıklama |
|-------|-----|----------|
| `ROOT_DIR` | Path | Proje kök dizini |
| `DATA_DIR` | Path | Veri klasörü |
| `MODELS_DIR` | Path | Model klasörü |
| `LOGS_DIR` | Path | Log klasörü |

### Model Parametreleri

| Sabit | Değer | Açıklama |
|-------|-------|----------|
| `RANDOM_SEED` | 42 | Rastgele tohum |
| `TRAIN_SIZE` | 0.8 | Eğitim oranı |
| `SAMPLE_SIZE` | 60000 | Örnekleme boyutu |

### TF-IDF Ayarları

```python
TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "max_features": None
}
```

### Aspekt Anahtar Kelimeleri

```python
ASPECT_KEYWORDS = {
    "kargo_teslimat": ["kargo", "teslimat", "gönderim", ...],
    "fiyat_performans": ["fiyat", "ucuz", "pahalı", ...],
    "kalite_malzeme": ["kalite", "malzeme", "sağlam", ...],
    "musteri_hizmetleri": ["iade", "destek", "iletişim", ...]
}
```

---

# English Documentation

This documentation explains all modules and functions of the Turkish E-Commerce Review Analysis project in detail.

## Table of Contents

1. [src.preprocessing](#srcpreprocessing-1)
2. [src.model](#srcmodel-1)
3. [src.spam_detector](#srcspam_detector-1)
4. [src.utils](#srcutils-1)
5. [src.app](#srcapp-1)
6. [config](#config-1)

---

## src.preprocessing

Module containing text normalization and data cleaning operations.

### Functions

#### `turkce_metin_normalize_et(text, stemming_uygula=True)`

Prepares Turkish text for machine learning.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | - | Raw text |
| `stemming_uygula` | bool | True | Apply stemming? |

**Returns:** `str` - Normalized text

**Applied Operations:**
1. Unicode normalization (NFKC)
2. Convert to lowercase
3. URL → `<url>` tag
4. Email → `<email>` tag
5. Phone → `<phone>` tag
6. Reduce repeating characters
7. Turkish stemming with TurkishStemmer

**Example:**
```python
from src.preprocessing import turkce_metin_normalize_et

raw = "HARIKA BİR ÜRÜN!!! www.site.com 😍😍😍"
clean = turkce_metin_normalize_et(raw)
print(clean)  # "harik bir ürün <url>"
```

---

#### `akilli_csv_oku(file_path)`

Smart CSV reading (auto-detect encoding and delimiter).

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | str | Path to CSV file |

**Returns:** `pd.DataFrame`

---

#### `veri_hazirla(file_path=None)`

Downloads/reads dataset and applies preprocessing.

**Returns:** `pd.DataFrame` - Prepared data

**Output DataFrame Columns:**
- `ham_metin`: Original review
- `metin`: Normalized text
- `duygu`: Sentiment label (0, 1, 2)

---

## src.model

Module containing sentiment analysis model training and usage.

### Classes

#### `SentimentModel`

TF-IDF + Logistic Regression based sentiment analysis model.

**Methods:**

##### `fit(X, y)`
Train the model.

##### `predict(X)`
Make predictions.

##### `predict_proba(X)`
Return probabilities.

##### `evaluate(X, y)`
Evaluate model performance.

##### `save(path)` / `load(path)`
Save and load model.

**Example:**
```python
from src.model import SentimentModel

# Train
model = SentimentModel()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(["Great product!", "Terrible quality"])
print(predictions)  # [2, 0]

# Save
model.save("models/sentiment_model.pkl")

# Load
model = SentimentModel.load("models/sentiment_model.pkl")
```

---

## src.spam_detector

Module containing spam/bot detection operations.

### Classes

#### `SpamDetector`

Hybrid spam detection system (Rule-based + IsolationForest).

**Methods:**

##### `fit(raw_texts, normalized_texts)`
Train the spam detection model.

##### `analyze(raw_text, normalized_text=None)`
Analyze a single review.

**Returns:** `dict`
```python
{
  "spam_olasiligi": 0.87,  # Spam probability
  "tahmin": 1,              # Prediction (0=genuine, 1=spam)
  "etiket": "Spam",         # Label
  "kural_skoru": 1,         # Rule-based score
  "aciklama": "..."         # Explanation
}
```

##### `predict(texts)`
Batch prediction.

##### `save(path)` / `load(path)`
Save and load model.

---

## src.utils

Module containing utility functions.

### Functions

#### `veri_indir(target_path=None, source="auto")`

Download dataset or read from local file.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_path` | str | None | Download target path |
| `source` | str | "auto" | "huggingface", "github", "local" or "auto" |

---

## src.app

Gradio web interface module.

### Functions

#### `analiz_yap(review)`

Fully analyze a review.

**Returns:** `dict` - Analysis results

### Launch Interface

```python
from src.app import demo

demo.launch(share=True)  # With shareable link
```

---

## config

Central configuration file.

### Path Constants

| Constant | Type | Description |
|----------|------|-------------|
| `ROOT_DIR` | Path | Project root directory |
| `DATA_DIR` | Path | Data folder |
| `MODELS_DIR` | Path | Models folder |

### Model Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `RANDOM_SEED` | 42 | Random seed |
| `TRAIN_SIZE` | 0.8 | Training ratio |
| `SAMPLE_SIZE` | 60000 | Sample size |

---

<div align="center">

**[🔝 Başa Dön / Back to Top](#-api-dokümantasyonu--api-documentation)**

</div>
