# Hızlı Başlangıç Kılavuzu

Bu kılavuz, projeyi kurduktan sonra hızlıca çalıştırmaya başlamanız için hazırlanmıştır.

## 🎯 İlk Adımlar

### 1. Sanal Ortamı Aktifleştirin

```bash
# WSL/Linux
source .venv_wsl/bin/activate

# macOS
source .venv_mac/bin/activate
```

### 2. Kurulumu Doğrulayın

```bash
# GPU kontrolü (varsa)
python -c "import tensorflow as tf; print('GPU:', len(tf.config.list_physical_devices('GPU')))"
```

## 🚀 En Hızlı Test (GPU)

GPU'nuz varsa, RSI stratejisini 8 saniyede optimize edin:

```bash
python tests/backtest_runner.py --rsi --tensorflow-gpu
```

20,000+ parametre kombinasyonu test edilir ve en iyi sonuçlar `results/` klasörüne kaydedilir.

## 💻 CPU ile Çalıştırma

GPU'nuz yoksa endişelenmeyin, CPU'da da çalışır:

```bash
# Tek strateji
python tests/backtest_runner.py --rsi --basic-strategy

# Tüm stratejiler
python tests/backtest_runner.py --all --basic-strategy
```

## 📊 Strateji Örnekleri

### RSI Stratejisi
```bash
# GPU optimizasyonu
python tests/backtest_runner.py --rsi --tensorflow-gpu

# Grid search (CPU)
python tests/backtest_runner.py --rsi --basic-strategy

# Random search
python tests/backtest_runner.py --rsi --random-search
```

### MACD Stratejisi
```bash
# GPU optimizasyonu
python tests/backtest_runner.py --macd --tensorflow-gpu

# Grid search (CPU)
python tests/backtest_runner.py --macd --basic-strategy
```

### EMA Stratejisi
```bash
# GPU optimizasyonu
python tests/backtest_runner.py --ema --tensorflow-gpu

# Grid search (CPU)
python tests/backtest_runner.py --ema --basic-strategy
```

### Tüm Stratejiler
```bash
# GPU ile tümü (sırayla)
python tests/backtest_runner.py --all-tensorflow-gpu

# CPU ile tümü
python tests/backtest_runner.py --all --basic-strategy
```

## 📈 Sonuçları İnceleme

Optimizasyon sonuçları `results/` klasörüne kaydedilir:

```bash
# Sonuçları listele
ls -la results/

# En son RSI sonucunu görüntüle
cat results/rsi_optimization_*.csv | head -20
```

Her CSV dosyası şunları içerir:
- Parametre kombinasyonları
- Toplam getiri
- Sharpe oranı
- Maksimum düşüş
- Kazanma oranı
- İşlem sayısı

## 🎛️ Özel Parametreler

### Veri Dosyası Değiştirme

`config.json` dosyasını düzenleyerek farklı veri kullanabilirsiniz:

```json
{
  "data_settings": {
    "default_data_file": "data/ethusdt_1h.csv"
  }
}
```

### Parametre Aralıklarını Değiştirme

```json
{
  "strategies": {
    "rsi": {
      "period_range": [10, 20],
      "oversold_range": [25, 35],
      "overbought_range": [65, 75]
    }
  }
}
```

## 📝 Python'dan Kullanım

### Basit Backtest
```python
from strategies.rsi.rsi_strategy import RSIStrategy, RSIParameters
import pandas as pd

# Veri yükle
data = pd.read_csv('data/btcusdt_1h.csv')

# Parametreler
params = RSIParameters(period=14, oversold=30, overbought=70)

# Backtest çalıştır
strategy = RSIStrategy()
result = strategy.backtest(data, params)

print(f"Getiri: {result.total_return:.2%}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

### Optimizasyon
```python
from strategies.rsi.rsi_strategy import RSIStrategy
import pandas as pd

# Veri yükle
data = pd.read_csv('data/btcusdt_1h.csv')

# Optimize et
strategy = RSIStrategy()
results = strategy.optimize_parameters(data, mode='tensorflow_gpu')

# En iyi parametreler
best = results.nlargest(1, 'sharpe_ratio')
print("En iyi parametreler:", best)
```

## 🔄 Canlı Veri Güncelleme

Binance'tan yeni veri çekmek için:

```bash
python data/fetch_binance_data.py
```

Bu script otomatik olarak:
- BTC, ETH, SOL verilerini günceller
- 5m, 15m, 30m, 1h, 4h, 1d timeframe'leri indirir
- CSV formatında `data/` klasörüne kaydeder

## ⚡ Performans İpuçları

### GPU Kullanımı
1. TensorFlow'un GPU'yu gördüğünden emin olun
2. Batch size'ı GPU belleğine göre ayarlayın
3. Mixed precision'ı aktif edin (config.json)

### CPU Optimizasyonu
1. Daha az parametre kombinasyonu kullanın
2. Random search tercih edin
3. Multiprocessing'i aktif tutun

## 🎯 Örnek Senaryo: İlk Optimizasyon

```bash
# 1. Sanal ortamı aktifleştir
source .venv_wsl/bin/activate

# 2. GPU'yu kontrol et
python -c "import tensorflow as tf; print('GPU var mı?', len(tf.config.list_physical_devices('GPU')) > 0)"

# 3. RSI stratejisini optimize et
python tests/backtest_runner.py --rsi --tensorflow-gpu

# 4. Sonuçları incele
ls -la results/
cat results/rsi_optimization_*.csv | head -5

# 5. En iyi parametreleri not al ve kullan
```

## 🔍 Çıktı Formatı

Tipik bir optimizasyon çıktısı:

```
===========================================
🚀 RSI Strategy - TensorFlow GPU Optimizer
===========================================
Data shape: (10000, 6)
Parameter combinations: 20,286

Optimizing with GPU...
✅ Optimization complete in 8.34 seconds

Top 5 Results:
period  oversold  overbought  total_return  sharpe_ratio
14      28        72          0.4523        1.86
12      30        70          0.4412        1.82
...

Results saved to: results/rsi_optimization_20240114_123456.csv
```

## ❓ Sık Sorulan Sorular

**S: GPU algılanmıyor ne yapmalıyım?**
C: [SORUN_GIDERME.md](SORUN_GIDERME.md) dosyasına bakın.

**S: Hangi strateji en iyisi?**
C: Piyasa koşullarına bağlı. Hepsini test edin.

**S: Kendi stratejimi nasıl eklerim?**
C: `strategies/` klasöründeki örnekleri inceleyin.

## 🎓 Sonraki Adımlar

1. Farklı veri setleri ile test edin
2. Parametre aralıklarını optimize edin
3. Birden fazla stratejiyi kombinleyin
4. Gerçek zamanlı trading'e geçiş yapın

## 💡 İpuçları

- İlk olarak küçük veri setleri ile test edin
- GPU varsa her zaman `--tensorflow-gpu` kullanın
- Sonuçları Excel'de analiz edebilirsiniz
- `config.json` ile ince ayar yapın