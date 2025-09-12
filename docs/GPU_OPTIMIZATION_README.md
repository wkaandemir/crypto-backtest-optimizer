# 🚀 **%100 GPU OPTIMIZED** RSI Optimizasyon Sistemi

## ✨ Yeni Özellikler - Version 3.0

Bu sistem, RSI trading stratejisi parametrelerini **%100 GPU optimizasyonu** ile hızlandırır ve sonuçları görsel SVG raporları olarak sunar.

### 🚀 **YENİ** Ana Özellikler:
- **🔥 TensorFlow GPU Hibrit Optimizer**: 169+ tests/sec hız (EN HIZLI!)
- **⚡ CuPy GPU Acceleration**: 50-100 tests/sec hız
- **🧠 Neural Network Optimization**: TensorFlow ile akıllı parametre optimizasyonu
- **📊 SVG Rapor Oluşturma**: Detaylı ve görsel SVG raporları
- **🔄 Otomatik Fallback**: GPU yoksa CPU ile çalışır
- **💾 Memory Optimization**: Mixed precision (FP16) + XLA JIT
- **📈 Batch Processing**: 500 paralel kombinasyon işleme

## 📋 Gereksinimler

### Temel Gereksinimler:
```bash
pip install numpy pandas
```

### GPU Desteği için:
```bash
# CUDA 12.x için (önerilen)
pip install cupy-cuda12x

# CUDA Toolkit kurulumu (WSL2 için)
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run --silent --toolkit
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

## 🎯 Kullanım

### 1. GPU Optimizasyonu Çalıştırma

```bash
# 🚀 100% GPU Optimized (RECOMMENDED)
python tests/backtest_runner.py --rsi --tensorflow-gpu
```

Bu komut:
- TensorFlow GPU hibrit optimizer kullanır
- 5000 farklı RSI parametre kombinasyonu test eder
- GPU'da parametre üretimi + CPU'da güvenilir backtesting
- 169+ tests/saniye hızla çalışır
- Sonuçları SVG raporu olarak kaydeder

### 2. Özel Parametre Aralıkları

`strategies/rsi/rsi_gpu_optimizer.py` dosyasında parametreleri özelleştirebilirsiniz:

```python
config = HybridGPUConfig(
    batch_size=500,              # Batch size (paralel işlem)
    period_min=5,                # RSI periyot minimum
    period_max=50,               # RSI periyot maximum
    oversold_min=15.0,           # Oversold minimum
    oversold_max=45.0,           # Oversold maximum
    overbought_min=55.0,         # Overbought minimum
    overbought_max=85.0          # Overbought maximum
)
```

### 3. SVG Raporu Görüntüleme

Oluşturulan SVG raporları `reports/` klasöründe kaydedilir:
- `reports/rsi_hybrid_gpu_YYYYMMDD_HHMMSS.svg`

Raporu görüntülemek için:
1. Dosyayı bir web tarayıcısında açın
2. Veya VS Code'da SVG preview uzantısı kullanın

## 📊 Rapor İçeriği

SVG raporu şunları içerir:

### Ana Tablo
- Parametre setleri (Period, Oversold, Overbought)
- Test edilen tarih aralığı (rastgele 1 yıl)
- Performans metrikleri:
  - Toplam Getiri (%)
  - Sharpe Oranı
  - Maximum Drawdown
  - İşlem Sayısı
  - Win Rate
  - Profit Factor
  - Calmar Ratio
- Görsel durum etiketleri (MÜKEMMEL, İYİ, ORTA, KÖTÜ)

### Özet İstatistikler
- Ortalama, medyan, en iyi/kötü değerler
- Standart sapma

### Görselleştirmeler
- Getiri dağılım histogramı
- GPU performans bilgileri
- En iyi 3 parametre seti

## 🔧 Özelleştirme

### Veri Kaynağı Değiştirme

`tests/backtest_runner.py` dosyasında varsayılan olarak `btcusdt_5m.csv` kullanılır. Farklı timeframe'ler için:

Mevcut veri dosyaları:
- `btcusdt_1m.csv` - 5 dakikalık
- `btcusdt_5m.csv` - 5 dakikalık
- `btcusdt_15m.csv` - 15 dakikalık
- `btcusdt_30m.csv` - 30 dakikalık
- `btcusdt_1h.csv` - 1 saatlik
- `btcusdt_4h.csv` - 4 saatlik
- `btcusdt_1d.csv` - Günlük

### Test Sayısını Artırma

Varsayılan olarak 5000 kombinasyon test edilir. Daha fazla test için:
```python
opt_results = optimizer.optimize_parameters(
    data=self.data, 
    num_tests=10000  # 5000'den 10000'e çıkarıldı
)
```

## 🎨 SVG Rapor Örnekleri

Ana komut çalıştırıldığında otomatik olarak SVG raporu oluşturulur:
```bash
python tests/backtest_runner.py --rsi --tensorflow-gpu
```

Bu komut otomatik olarak `reports/rsi_hybrid_gpu_YYYYMMDD_HHMMSS.svg` dosyasını oluşturur.

## ⚡ Performans

### 🚀 TensorFlow GPU Hibrit (RECOMMENDED):
- **169+ tests/saniye** (EN HIZLI)
- 5000 test ~30 saniye
- RTX 4060 Ti ile test edildi
- ~500MB GPU memory kullanımı
- Mixed Precision (FP16) + XLA JIT optimization

## 🐛 Sorun Giderme

### GPU Bulunamadı Hatası
```bash
⚠️ GPU not available. Falling back to CPU (will be slower)
```
**Çözüm**: CUDA ve CuPy'nin düzgün kurulduğundan emin olun.

### Import Hatası
```bash
❌ Error: GPU optimizer not available
```
**Çözüm**: Gerekli paketleri kurun:
```bash
pip install cupy-cuda11x numpy pandas
```

## 📝 Notlar

- Her parametre kombinasyonu farklı bir zaman diliminde test edilir
- Bu yaklaşım overfitting'i azaltır ve daha robust parametreler bulur
- SVG raporları vektör tabanlıdır ve her boyutta net görünür
- Raporlar Excel'e aktarılabilir CSV verisi olarak da export edilebilir (gelecek özellik)

## 🚀 Gelecek Özellikler

- [ ] CSV export desteği
- [ ] Multi-GPU desteği
- [ ] Diğer stratejiler için GPU optimizasyonu (MACD, Bollinger Bands, vb.)
- [ ] Gerçek zamanlı optimizasyon görselleştirmesi
- [ ] Web tabanlı rapor görüntüleyici

## 📧 Destek

Sorularınız veya önerileriniz için issue açabilirsiniz.