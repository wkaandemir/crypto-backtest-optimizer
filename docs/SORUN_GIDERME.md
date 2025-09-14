# Sorun Giderme Kılavuzu

Bu dokümanda karşılaşabileceğiniz yaygın sorunlar ve çözümleri bulunmaktadır.

## 🔴 GPU Sorunları

### GPU Algılanmıyor

**Belirti:**
```
GPU Sayısı: 0
```

**Çözümler:**

1. **NVIDIA sürücüsünü kontrol edin:**
```bash
nvidia-smi
```

2. **CUDA kurulumunu doğrulayın:**
```bash
nvcc --version
# CUDA 12.6 göstermeli
```

3. **TensorFlow'u GPU desteği ile yeniden kurun:**
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

4. **WSL2 kullanıcıları için:**
- Windows'ta NVIDIA GPU sürücüsünü güncelleyin
- WSL2 kernel'ini güncelleyin: `wsl --update`

### CUDA Out of Memory

**Belirti:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Çözümler:**

1. **Batch size'ı azaltın** (`config.json`):
```json
{
  "optimization_settings": {
    "batch_size": 1000
  }
}
```

2. **GPU bellek büyümesini etkinleştirin:**
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### cuDNN Bulunamadı

**Belirti:**
```
Could not load dynamic library 'libcudnn.so.9'
```

**Çözüm:**
```bash
sudo apt-get install libcudnn9-cuda-12
```

## 🔴 Python/Pip Sorunları

### ModuleNotFoundError

**Belirti:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Çözümler:**

1. **Sanal ortamın aktif olduğundan emin olun:**
```bash
source .venv_wsl/bin/activate
```

2. **Bağımlılıkları yeniden kurun:**
```bash
pip install -r requirements.txt
```

### Version Conflict

**Belirti:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Çözüm:**
```bash
# Temiz kurulum
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 🔴 Veri Sorunları

### Veri Dosyası Bulunamadı

**Belirti:**
```
FileNotFoundError: data/btcusdt_1h.csv
```

**Çözümler:**

1. **Veriyi indirin:**
```bash
python data/fetch_binance_data.py
```

2. **Dosya yolunu kontrol edin:**
```bash
ls -la data/
```

### Veri Formatı Hatası

**Belirti:**
```
KeyError: 'close'
```

**Çözüm:**
CSV dosyanızın şu sütunları içerdiğinden emin olun:
- timestamp
- open
- high
- low
- close
- volume

## 🔴 Performans Sorunları

### Çok Yavaş Çalışıyor

**CPU'da yavaş:**
- Random search kullanın: `--random-search`
- Parametre aralıklarını daraltın
- Daha az veri kullanın

**GPU'da yavaş:**
- GPU'nun gerçekten kullanıldığını kontrol edin:
```bash
watch -n 1 nvidia-smi
```

### Bellek Yetersizliği (RAM)

**Belirti:**
```
MemoryError: Unable to allocate array
```

**Çözümler:**

1. **Veri boyutunu azaltın:**
```python
data = pd.read_csv('data/btcusdt_1h.csv').tail(5000)  # Son 5000 satır
```

2. **Swap alanı ekleyin:**
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 🔴 WSL2 Özel Sorunları

### WSL2'de GPU Görünmüyor

1. **Windows'ta PowerShell (Admin):**
```powershell
wsl --update
wsl --shutdown
```

2. **NVIDIA WSL2 sürücüsünü yükleyin:**
- [NVIDIA WSL2 Driver](https://developer.nvidia.com/cuda/wsl) sayfasından indirin

### WSL2 Dosya Erişim Yavaş

**Çözüm:** Proje dosyalarını WSL2 dosya sistemine taşıyın:
```bash
# Windows'tan WSL2'ye kopyala
cp -r /mnt/c/Users/username/project ~/project
cd ~/project
```

## 🔴 Strateji Sorunları

### Hiç Sinyal Üretmiyor

**Belirti:**
```
İşlem sayısı: 0
```

**Olası sebepler:**
- Parametreler çok katı (örn: RSI oversold=10, overbought=90)
- Veri aralığı çok kısa
- Strateji mantığında hata

**Çözüm:**
Daha geniş parametreler deneyin:
```python
params = RSIParameters(period=14, oversold=35, overbought=65)
```

### NaN veya Inf Değerler

**Belirti:**
```
ValueError: Input contains NaN, infinity or a value too large
```

**Çözümler:**

1. **Veriyi temizleyin:**
```python
data = data.dropna()
data = data.replace([np.inf, -np.inf], np.nan).dropna()
```

2. **Sıfıra bölme kontrolü ekleyin:**
```python
returns = np.where(prices[:-1] != 0, (prices[1:] - prices[:-1]) / prices[:-1], 0)
```

## 🔴 Kurulum Sorunları

### install.sh Çalışmıyor

**Çözüm:**
```bash
# İzinleri düzelt
chmod +x install.sh

# Manuel olarak çalıştır
bash install.sh
```

### pip install Donuyor

**Çözümler:**

1. **Timeout'u artırın:**
```bash
pip install -r requirements.txt --timeout 1000
```

2. **Cache'i temizleyin:**
```bash
pip cache purge
```

## 🔴 Sonuç Dosyası Sorunları

### results/ Klasörü Oluşmuyor

**Çözüm:**
```bash
mkdir -p results
```

### CSV Encoding Hatası

**Belirti:**
```
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Çözüm:**
```python
df = pd.read_csv('file.csv', encoding='latin-1')
# veya
df = pd.read_csv('file.csv', encoding='cp1252')
```

## 🔧 Genel Debugging İpuçları

### 1. Verbose Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. GPU Monitoring
```bash
# GPU kullanımını izle
watch -n 1 nvidia-smi

# TensorFlow log seviyesini artır
export TF_CPP_MIN_LOG_LEVEL=0
```

### 3. Profiling
```python
import cProfile
cProfile.run('strategy.optimize_parameters(data, "tensorflow_gpu")')
```

### 4. Memory Profiling
```bash
pip install memory_profiler
python -m memory_profiler tests/backtest_runner.py --rsi --basic-strategy
```

## 📞 Destek

Sorun devam ediyorsa:

1. **Hata mesajını kaydedin:**
```bash
python tests/backtest_runner.py --rsi --tensorflow-gpu 2>&1 | tee error.log
```

2. **Sistem bilgilerini toplayın:**
```bash
python -c "import sys; print('Python:', sys.version)"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
nvidia-smi
nvcc --version
```

3. **GitHub'da issue açın:**
- Hata mesajını
- Sistem bilgilerini
- Çalıştırdığınız komutu
- `config.json` içeriğini paylaşın

## 🔄 Temiz Kurulum

Her şey başarısız olursa:

```bash
# 1. Sanal ortamı sil
deactivate
rm -rf .venv_wsl

# 2. Cache'leri temizle
pip cache purge
rm -rf ~/.cache/pip

# 3. Yeniden kur
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. GPU varsa TensorFlow GPU kur
pip uninstall tensorflow -y
pip install tensorflow[and-cuda]
```

## ✅ Kontrol Listesi

Kurulum sonrası kontroller:

- [ ] Python 3.8+ kurulu mu?
- [ ] Sanal ortam aktif mi?
- [ ] `nvidia-smi` çalışıyor mu? (GPU varsa)
- [ ] TensorFlow import ediliyor mu?
- [ ] Veri dosyaları mevcut mu?
- [ ] `results/` klasörü yazılabilir mi?
- [ ] Basit bir test çalışıyor mu?