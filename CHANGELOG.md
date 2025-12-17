# 📝 Değişiklik Günlüğü / Changelog

Bu dosya projedeki tüm önemli değişiklikleri belgelemiştir.  
This file documents all notable changes to this project.

Format [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) standardına dayanır.

---

## [1.0.0] - 2025-12-17

### 🎉 İlk Sürüm / Initial Release

#### ✨ Eklenenler / Added

**Veri İşleme / Data Processing**
- Türkçe metin normalizasyonu (`turkce_metin_normalize_et`)
- TurkishStemmer ile kök bulma desteği
- Akıllı CSV okuyucu (encoding ve ayırıcı otomatik tespit)
- Otomatik sütun tespiti (metin, etiket, puan)
- URL, e-posta, telefon temizleme
- Unicode normalizasyonu (NFKC)

**Duygu Analizi / Sentiment Analysis**
- TF-IDF + Logistic Regression tabanlı model
- 3 sınıflı sınıflandırma (Negatif, Nötr, Pozitif)
- ~%85 doğruluk oranı
- Model kaydetme ve yükleme
- Olasılık tahminleri

**Spam Tespiti / Spam Detection**
- Hibrit spam tespit sistemi
- Kural tabanlı tespit (%60 ağırlık):
  - URL/email/telefon tespiti
  - Emoji ve ünlem sayısı kontrolü
  - Büyük harf oranı kontrolü
  - Jenerik ifade tespiti
- IsolationForest anomali tespiti (%40 ağırlık)
- ~%92 doğruluk oranı

**Aspekt Analizi / Aspect Analysis**
- Kargo/teslimat aspekti
- Fiyat/performans aspekti
- Kalite/malzeme aspekti
- Müşteri hizmetleri aspekti

**Web Arayüzü / Web Interface**
- Gradio ile interaktif demo
- Gerçek zamanlı analiz
- JSON formatında sonuçlar
- Paylaşılabilir link desteği

**Dokümantasyon / Documentation**
- Türkçe README
- İngilizce README
- API dokümantasyonu
- CONTRIBUTING rehberi
- CHANGELOG dosyası

**Veri Seti Desteği / Dataset Support**
- HuggingFace TRSAv1 entegrasyonu
- GitHub yedek indirme
- Yerel dosya okuma

**Konfigürasyon / Configuration**
- Merkezi config.py dosyası
- Modüler proje yapısı
- requirements.txt

---

## [Planlanmış / Planned]

### 🔮 Gelecek Sürümler / Future Releases

#### v1.1.0 (Planlanıyor / Planned)
- [ ] BERT fine-tuning desteği
- [ ] REST API endpoint'leri
- [ ] Docker konteyner desteği
- [ ] Toplu işlem (batch processing)

#### v1.2.0 (Planlanıyor / Planned)
- [ ] Çok dilli destek (İngilizce, Arapça)
- [ ] Gerçek zamanlı streaming analiz
- [ ] Web dashboard
- [ ] Veritabanı entegrasyonu

#### v2.0.0 (Planlanıyor / Planned)
- [ ] İroni/alay tespiti
- [ ] Çok etiketli sınıflandırma
- [ ] Özelleştirilebilir kurallar
- [ ] Cloud deployment (AWS/GCP)

---

## Sürüm Numaralandırma / Versioning

Bu proje [Semantic Versioning](https://semver.org/) kullanır:

- **MAJOR**: Geriye dönük uyumsuz değişiklikler
- **MINOR**: Geriye dönük uyumlu yeni özellikler
- **PATCH**: Geriye dönük uyumlu hata düzeltmeleri

---

## Değişiklik Tipleri / Change Types

| Emoji | Tip | Açıklama |
|-------|-----|----------|
| ✨ | Added | Yeni özellik |
| 🔄 | Changed | Mevcut özellikte değişiklik |
| ⚠️ | Deprecated | Yakında kaldırılacak özellik |
| 🗑️ | Removed | Kaldırılan özellik |
| 🐛 | Fixed | Hata düzeltmesi |
| 🔒 | Security | Güvenlik düzeltmesi |

---

<div align="center">

**Tarih / Date**: Aralık 2025 / December 2025

</div>
