# Crypto Backtest Optimizer

🚀 GPU hızlandırmalı, yüksek performanslı kripto para ticaret stratejisi geri test ve optimizasyon framework'ü. Gerçek paralel GPU hesaplama ile saniyeler içinde binlerce strateji kombinasyonunu test edin.

## 🚀 Özellikler

- **⚡ Ultra Hızlı GPU Optimizasyonu**: 2700x'e kadar performans artışı (2500+ test/saniye)
- **📊 3 Farklı Strateji**: RSI, MACD, EMA - her biri GPU optimizasyonlu
- **🔧 Çoklu Optimizasyon Yöntemleri**:
  - TensorFlow GPU (gerçek paralel hesaplama)
  - Grid Search (sistematik parametre taraması)
  - Random Search (stokastik örnekleme)
- **📈 Gerçek Piyasa Verileri**: Binance Futures'tan BTC, ETH, SOL verileri (5m'den 1d'ye)
- **🎯 Kapsamlı Metrikler**: Sharpe oranı, maksimum düşüş, kazanma oranı, toplam getiri
- **🔄 Otomatik Kurulum**: Tek komutla GPU/CUDA/TensorFlow kurulumu

## 📊 Performans Karşılaştırması

| Strateji | Parametre Sayısı | CPU Süresi | GPU Süresi | Hızlanma | Test/Saniye (GPU) |
|----------|-----------------|------------|------------|----------|-------------------|
| RSI | 20,286 | ~6 saat | **8 saniye** | **2,700x** | 2,535 |
| MACD | 1,320 | ~30 dakika | **3 saniye** | **600x** | 440 |
| EMA | 88 | ~3 dakika | **1 saniye** | **180x** | 88 |

> 💡 **Not**: GPU testleri NVIDIA RTX serisi kartlarda gerçekleştirilmiştir. Gerçek paralel hesaplama sayesinde tüm parametreler aynı anda test edilir.

## 📋 Gereksinimler

### Minimum Gereksinimler
- Python 3.8+
- 4GB RAM
- WSL2 (Windows kullanıcıları için)

### Önerilen Gereksinimler (GPU Optimizasyonu için)
- NVIDIA GPU (CUDA 11.2+ desteği)
- 8GB+ GPU belleği
- Ubuntu 20.04+ veya WSL2
- CUDA Toolkit 11.2-12.0
- cuDNN 8.1+

## 🔧 Kurulum

Detaylı kurulum talimatları için [KURULUM.md](docs/KURULUM.md) dosyasına bakın.

### Hızlı Kurulum

#### Otomatik Kurulum (Önerilen)
```bash
# GPU/CUDA/TensorFlow dahil tam kurulum
bash install.sh
```

#### Manuel Kurulum
```bash
# Sanal ortam oluştur ve aktifleştir
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# GPU desteğini doğrula (isteğe bağlı)
python -c "import tensorflow as tf; print(f'GPU Sayısı: {len(tf.config.list_physical_devices("GPU"))}')"
```

## ⚡ Hızlı Başlangıç

### 1. Ortamı Hazırlayın
```bash
# Sanal ortamı aktifleştir
source .venv_wsl/bin/activate
```

### 2. GPU ile Ultra Hızlı Optimizasyon (Önerilen)
```bash
# RSI stratejisi - 20,000+ parametre, ~8 saniye
python tests/backtest_runner.py --rsi --tensorflow-gpu

# MACD stratejisi - 1,300+ parametre, ~3 saniye
python tests/backtest_runner.py --macd --tensorflow-gpu

# EMA stratejisi - 88 parametre, ~1 saniye
python tests/backtest_runner.py --ema --tensorflow-gpu

# Tüm stratejileri sırayla optimize et
python tests/backtest_runner.py --all-tensorflow-gpu
```

### 3. CPU ile Optimizasyon (GPU yoksa)
```bash
# Temel grid search ile test
python tests/backtest_runner.py --rsi --basic-strategy
python tests/backtest_runner.py --macd --basic-strategy
python tests/backtest_runner.py --ema --basic-strategy

# Random search (sadece RSI)
python tests/backtest_runner.py --rsi --random-search
```

Detaylı kullanım örnekleri için [HIZLI_BASLANGIC.md](docs/HIZLI_BASLANGIC.md) dosyasına bakın.

## 📁 Proje Yapısı

```
├── strategies/              # Ticaret stratejileri
│   ├── rsi/                # RSI stratejisi
│   │   ├── rsi_strategy.py         # Temel RSI mantığı
│   │   └── rsi_gpu_optimizer.py    # GPU hızlandırmalı optimizasyon
│   ├── macd/               # MACD stratejisi
│   │   ├── macd_strategy.py        # Temel MACD mantığı
│   │   └── macd_gpu_optimizer.py   # GPU hızlandırmalı optimizasyon
│   └── ema/                # EMA stratejisi
│       ├── ema_strategy.py         # Temel EMA mantığı
│       └── ema_gpu_optimizer.py    # GPU hızlandırmalı optimizasyon
├── tests/
│   └── backtest_runner.py  # Universal test çalıştırıcı (tüm stratejiler)
├── data/                   # Piyasa verileri (CSV formatında)
│   ├── fetch_binance_data.py       # Veri çekme aracı
│   └── *.csv               # BTC, ETH, SOL verileri (5m-1d)
├── results/                # Optimizasyon sonuçları (otomatik oluşturulur)
├── docs/                   # Dokümantasyon
│   ├── KURULUM.md         # Detaylı kurulum kılavuzu
│   ├── HIZLI_BASLANGIC.md # Kullanım örnekleri
│   └── SORUN_GIDERME.md   # Yaygın sorunlar ve çözümler
├── config.json            # Strateji parametreleri ve ayarları
├── requirements.txt       # Python bağımlılıkları
├── install.sh            # Otomatik GPU/CUDA kurulum scripti
└── CLAUDE.md             # AI asistan için proje kılavuzu
```

## 🎯 Desteklenen Stratejiler

### RSI (Relative Strength Index)
**Momentum tabanlı strateji** - Aşırı alım/satım bölgelerini tespit eder
- **Sinyal Mantığı**: RSI < 30 (aşırı satım) → AL, RSI > 70 (aşırı alım) → SAT
- **Parametreler**: Periyot (5-50), aşırı alım (60-90), aşırı satım (10-40)
- **En İyi Kullanım**: Yatay piyasalar, kısa vadeli dalgalanmalar
- [Detaylı dokümantasyon →](strategies/rsi/README.md)

### MACD (Moving Average Convergence Divergence)
**Trend takip stratejisi** - MACD ve sinyal çizgisi kesişimleri
- **Sinyal Mantığı**: MACD sinyal çizgisini yukarı keser → AL, aşağı keser → SAT
- **Parametreler**: Hızlı EMA (5-20), Yavaş EMA (20-50), Sinyal (5-15)
- **En İyi Kullanım**: Trend olan piyasalar, orta vadeli işlemler
- [Detaylı dokümantasyon →](strategies/macd/README.md)

### EMA (Exponential Moving Average)
**Basit trend takibi** - Hızlı/yavaş EMA kesişimleri
- **Sinyal Mantığı**: Hızlı EMA > Yavaş EMA → AL, Hızlı EMA < Yavaş EMA → SAT
- **Parametreler**: Hızlı periyot (5-20), Yavaş periyot (20-100)
- **En İyi Kullanım**: Güçlü trendler, uzun vadeli pozisyonlar
- [Detaylı dokümantasyon →](strategies/ema/README.md)

## 🛠️ Optimizasyon Yöntemleri

### 1. TensorFlow GPU (Ultra Hızlı - Önerilen)
- **Teknoloji**: Gerçek paralel GPU hesaplama, XLA JIT derleme
- **Performans**: 2500+ test/saniye, 2700x CPU'dan hızlı
- **Özellikler**: Vektörize operasyonlar, batch processing, mixed precision
- **Kullanım**: `--tensorflow-gpu` parametresi

### 2. Grid Search (CPU)
- **Teknoloji**: Sistematik parametre taraması
- **Performans**: Orta hız, tüm kombinasyonları test eder
- **Özellikler**: Deterministik, tekrarlanabilir sonuçlar
- **Kullanım**: `--basic-strategy` parametresi

### 3. Random Search (RSI için)
- **Teknoloji**: Stokastik parametre örnekleme
- **Performans**: Hızlı yaklaşık sonuçlar
- **Özellikler**: Geniş parametre uzayında etkili
- **Kullanım**: `--random-search` parametresi

## 💻 Komut Satırı Parametreleri

```bash
python tests/backtest_runner.py [STRATEJI] [OPTİMİZASYON]
```

**Strateji Seçenekleri:**
- `--rsi`: RSI stratejisini test et
- `--macd`: MACD stratejisini test et
- `--ema`: EMA stratejisini test et
- `--all`: Tüm stratejileri sırayla test et

**Optimizasyon Seçenekleri:**
- `--tensorflow-gpu`: GPU ile ultra hızlı optimizasyon
- `--basic-strategy`: CPU ile grid search
- `--random-search`: Stokastik örnekleme (sadece RSI)
- `--all-tensorflow-gpu`: Tüm stratejileri GPU ile optimize et

## 📈 Veri Yönetimi

### Mevcut Veriler
- **Çiftler**: BTC/USDT, ETH/USDT, SOL/USDT
- **Zaman Dilimleri**: 5m, 15m, 30m, 1h, 4h, 1d
- **Kaynak**: Binance Futures
- **Format**: CSV (timestamp, open, high, low, close, volume)

### Yeni Veri Çekme
```bash
python data/fetch_binance_data.py
```

## 🐛 Sorun Giderme

Yaygın sorunlar ve çözümleri için [SORUN_GIDERME.md](docs/SORUN_GIDERME.md) dosyasına bakın.

### Hızlı Çözümler
- **GPU bulunamadı**: `nvidia-smi` ile kontrol edin, `bash install.sh` ile CUDA kurun
- **TensorFlow hatası**: `pip install --upgrade tensorflow[and-cuda]`
- **Bellek hatası**: GPU batch size'ı config.json'da azaltın

## 🗺️ Yol Haritası

- ✅ GPU hızlandırmalı backtest optimizasyonu
- ✅ RSI, MACD, EMA stratejileri
- ✅ Binance veri entegrasyonu
- 🔄 Daha fazla teknik indikatör (Bollinger Bands, Stochastic)
- 🔄 Machine Learning tabanlı strateji optimizasyonu
- 🔄 Gerçek zamanlı trading botu
- 🔄 Web arayüzü ve API

Detaylar için [ROADMAP.md](ROADMAP.md) dosyasına bakın.

## 🏆 Öne Çıkan Özellikler

- **Gerçek GPU Paralelizmi**: For döngüsü yok, saf tensor operasyonları
- **XLA JIT Derleme**: Maksimum GPU performansı
- **Otomatik Bağımlılık Kontrolü**: Eksik paketleri otomatik tespit ve kurulum önerileri
- **Performans Raporlaması**: Stratejiler arası karşılaştırmalı analiz
- **Profesyonel Metrikler**: Sharpe oranı, maksimum düşüş, kazanma oranı

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🤝 Katkıda Bulunma

Pull request'ler kabul edilir. Büyük değişiklikler için önce issue açınız.

### Katkı Kuralları
- Kod yorumları ve dokümantasyon Türkçe olmalıdır
- Yeni stratejiler GPU optimizasyonu içermelidir
- Test coverage %80+ olmalıdır
- Black ve isort ile formatlanmalıdır

## 📧 İletişim

- **GitHub Issues**: Hata raporları ve özellik istekleri
- **Discussions**: Genel sorular ve tartışmalar
- **Wiki**: Detaylı kılavuzlar ve örnekler

---

<div align="center">

**⚡ GPU Gücüyle Saniyeler İçinde Binlerce Strateji Testi ⚡**

*Geliştirici: [@wkaandemir](https://github.com/wkaandemir)*

</div>