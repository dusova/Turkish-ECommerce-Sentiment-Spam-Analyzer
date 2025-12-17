# 🤝 Katkıda Bulunma Rehberi / Contributing Guide

<div align="center">

**[🇹🇷 Türkçe](#türkçe) | [🇬🇧 English](#english)**

</div>

---

# Türkçe

Projeye katkıda bulunmak istediğiniz için teşekkürler! 🎉

Bu rehber, projeye nasıl katkıda bulunabileceğinizi açıklar.

## 📋 İçindekiler

1. [Davranış Kuralları](#davranış-kuralları)
2. [Nasıl Katkıda Bulunabilirim?](#nasıl-katkıda-bulunabilirim)
3. [Geliştirme Ortamı Kurulumu](#geliştirme-ortamı-kurulumu)
4. [Kod Standartları](#kod-standartları)
5. [Commit Mesajları](#commit-mesajları)
6. [Pull Request Süreci](#pull-request-süreci)

---

## 📜 Davranış Kuralları

Bu proje, tüm katılımcıların saygılı ve kapsayıcı bir ortamda çalışmasını bekler.

### Kabul Edilebilir Davranışlar

- ✅ Yapıcı eleştiri
- ✅ Farklı görüşlere saygı
- ✅ Yardımseverlik
- ✅ Profesyonel iletişim

### Kabul Edilemez Davranışlar

- ❌ Hakaret veya aşağılama
- ❌ Taciz
- ❌ Spam
- ❌ Kişisel saldırılar

---

## 🛠️ Nasıl Katkıda Bulunabilirim?

### 🐛 Hata Bildirimi

1. **Mevcut issue'ları kontrol edin** - Belki aynı hata zaten bildirilmiştir
2. **Yeni issue açın** - Detaylı bilgi verin:
   - Hata açıklaması
   - Adım adım yeniden oluşturma
   - Beklenen davranış
   - Gerçekleşen davranış
   - Ortam bilgileri (Python versiyonu, OS vb.)

### 💡 Özellik Önerisi

1. **Discussion açın** - Önce tartışalım
2. **Issue oluşturun** - Kabul edilirse
3. **PR gönderin** - İmplementasyonla birlikte

### 📝 Dokümantasyon

- Yazım hatalarını düzeltin
- Eksik açıklamaları ekleyin
- Örnekler ekleyin
- Çevirileri iyileştirin

### 💻 Kod Katkısı

1. Projeyi fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi yapın
4. Test edin
5. PR gönderin

---

## 🔧 Geliştirme Ortamı Kurulumu

### 1. Repoyu Fork edin ve Klonlayın

```bash
git clone https://github.com/KULLANICI_ADINIZ/Turkish-ECommerce-Sentiment-Spam-Analyzer.git
cd Turkish-ECommerce-Sentiment-Spam-Analyzer
```

### 2. Sanal Ortam Oluşturun

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Geliştirme Bağımlılıklarını Yükleyin

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Varsa
```

### 4. Pre-commit Hook'larını Kurun (Opsiyonel)

```bash
pip install pre-commit
pre-commit install
```

---

## 📏 Kod Standartları

### Python Stil Rehberi

Bu proje **PEP 8** standartlarını takip eder.

#### Genel Kurallar

```python
# ✅ DOĞRU
def fonksiyon_adi(parametre_bir, parametre_iki):
    """Kısa açıklama."""
    sonuc = parametre_bir + parametre_iki
    return sonuc


# ❌ YANLIŞ
def FonksiyonAdi(parametreBir,parametreIki):
    sonuc=parametreBir+parametreIki
    return sonuc
```

#### Docstring Formatı

Google stili docstring kullanıyoruz:

```python
def fonksiyon(parametre: str, opsiyonel: int = 10) -> dict:
    """
    Fonksiyonun kısa açıklaması.
    
    Daha uzun açıklama burada olabilir. Birden fazla
    paragraf kullanılabilir.
    
    Args:
        parametre: Parametrenin açıklaması.
        opsiyonel: Opsiyonel parametrenin açıklaması.
            Varsayılan değer 10.
    
    Returns:
        Dönüş değerinin açıklaması.
    
    Raises:
        ValueError: Hata durumunun açıklaması.
    
    Example:
        >>> fonksiyon("test")
        {"sonuc": "test"}
    """
    pass
```

#### Type Hints

Type hint kullanımı teşvik edilir:

```python
from typing import List, Dict, Optional, Tuple

def analiz_et(
    metin: str,
    stemming: bool = True
) -> Tuple[str, float]:
    """..."""
    pass
```

### Dosya Organizasyonu

```python
"""
Modül docstring'i
"""

# Standart kütüphaneler
import os
import sys

# Üçüncü parti kütüphaneler
import numpy as np
import pandas as pd

# Yerel modüller
from src.preprocessing import turkce_metin_normalize_et

# Sabitler
MAX_LENGTH = 1000

# Sınıflar
class SinifAdi:
    pass

# Fonksiyonlar
def fonksiyon():
    pass

# Ana kod
if __name__ == "__main__":
    pass
```

---

## 💬 Commit Mesajları

[Conventional Commits](https://www.conventionalcommits.org/) formatını kullanıyoruz:

### Format

```
<tip>(<kapsam>): <açıklama>

[opsiyonel gövde]

[opsiyonel footer]
```

### Tipler

| Tip | Açıklama |
|-----|----------|
| `feat` | Yeni özellik |
| `fix` | Hata düzeltmesi |
| `docs` | Dokümantasyon değişikliği |
| `style` | Kod formatı (işlevsellik değişmez) |
| `refactor` | Kod yeniden düzenleme |
| `test` | Test ekleme/düzeltme |
| `chore` | Bakım işleri |

### Örnekler

```bash
# Yeni özellik
git commit -m "feat(model): add BERT support for sentiment analysis"

# Hata düzeltmesi
git commit -m "fix(preprocessing): handle empty strings in normalization"

# Dokümantasyon
git commit -m "docs(readme): add installation instructions for Windows"

# Refactoring
git commit -m "refactor(spam): simplify rule-based scoring logic"
```

---

## 🔄 Pull Request Süreci

### 1. Branch Oluşturun

```bash
git checkout -b feature/ozellik-adi
# veya
git checkout -b fix/hata-adi
```

### 2. Değişikliklerinizi Yapın

- Küçük, odaklı commit'ler yapın
- Her commit çalışır durumda olmalı
- Test ekleyin

### 3. Push Edin

```bash
git push origin feature/ozellik-adi
```

### 4. PR Açın

GitHub'da "Compare & pull request" butonuna tıklayın.

### PR Şablonu

```markdown
## Değişiklik Açıklaması

Bu PR ne yapıyor? Neden gerekli?

## Değişiklik Tipi

- [ ] Yeni özellik
- [ ] Hata düzeltmesi
- [ ] Dokümantasyon
- [ ] Refactoring
- [ ] Diğer (açıklayın)

## Test

Değişiklikleri nasıl test ettiniz?

## Kontrol Listesi

- [ ] Kod PEP 8 standartlarına uygun
- [ ] Docstring'ler eklenmiş
- [ ] Testler geçiyor
- [ ] Dokümantasyon güncellenmiş

## İlgili Issue

Closes #123
```

### 5. Review Süreci

- En az 1 onay gereklidir
- CI testleri geçmelidir
- Çakışmalar çözülmelidir

---

## 🙏 Teşekkürler!

Katkılarınız bu projeyi daha iyi hale getirir. Her katkı değerlidir!

---

# English

Thank you for wanting to contribute to the project! 🎉

This guide explains how you can contribute to the project.

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Environment Setup](#development-environment-setup)
4. [Code Standards](#code-standards)
5. [Commit Messages](#commit-messages)
6. [Pull Request Process](#pull-request-process)

---

## 📜 Code of Conduct

This project expects all participants to work in a respectful and inclusive environment.

### Acceptable Behavior

- ✅ Constructive criticism
- ✅ Respect for different opinions
- ✅ Helpfulness
- ✅ Professional communication

### Unacceptable Behavior

- ❌ Insults or belittling
- ❌ Harassment
- ❌ Spam
- ❌ Personal attacks

---

## 🛠️ How Can I Contribute?

### 🐛 Bug Reports

1. **Check existing issues** - Maybe the same bug has already been reported
2. **Open new issue** - Provide detailed information:
   - Bug description
   - Step-by-step reproduction
   - Expected behavior
   - Actual behavior
   - Environment info (Python version, OS, etc.)

### 💡 Feature Suggestions

1. **Open a Discussion** - Let's discuss first
2. **Create an Issue** - If accepted
3. **Submit a PR** - With implementation

### 📝 Documentation

- Fix typos
- Add missing explanations
- Add examples
- Improve translations

### 💻 Code Contribution

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Test
5. Submit a PR

---

## 🔧 Development Environment Setup

### 1. Fork and Clone the Repo

```bash
git clone https://github.com/dusova/Turkish-ECommerce-Sentiment-Spam-Analyzer.git
cd Turkish-ECommerce-Sentiment-Spam-Analyzer
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

---

## 📏 Code Standards

### Python Style Guide

This project follows **PEP 8** standards.

### Docstring Format

We use Google style docstrings:

```python
def function(parameter: str, optional: int = 10) -> dict:
    """
    Short description of the function.
    
    Args:
        parameter: Description of parameter.
        optional: Description of optional parameter.
            Default value is 10.
    
    Returns:
        Description of return value.
    
    Example:
        >>> function("test")
        {"result": "test"}
    """
    pass
```

---

## 💬 Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation change |
| `style` | Code format (no functionality change) |
| `refactor` | Code refactoring |
| `test` | Adding/fixing tests |
| `chore` | Maintenance tasks |

### Examples

```bash
git commit -m "feat(model): add BERT support for sentiment analysis"
git commit -m "fix(preprocessing): handle empty strings in normalization"
git commit -m "docs(readme): add installation instructions for Windows"
```

---

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/feature-name
```

### 2. Make Your Changes

- Make small, focused commits
- Each commit should be in working state
- Add tests

### 3. Push

```bash
git push origin feature/feature-name
```

### 4. Open a PR

Click "Compare & pull request" button on GitHub.

### 5. Review Process

- At least 1 approval is required
- CI tests must pass
- Conflicts must be resolved

---

## 🙏 Thank You!

Your contributions make this project better. Every contribution is valuable!

---

<div align="center">

**[🔝 Back to Top](#-katkıda-bulunma-rehberi--contributing-guide)**

</div>
