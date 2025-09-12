# Crypto Trading Strategy Optimizer

GPU hızlandırmalı yüksek performanslı ticaret stratejisi geri test ve optimizasyon framework'ü.

## Gereksinimler

- Python 3.8+
- WSL2 (Windows kullanıcıları için)
- NVIDIA GPU (isteğe bağlı ancak performans için önerilir)

## Hızlı Kurulum (Tek Komut)

Tüm gereksinimleri (CUDA, cuDNN, Python paketleri) otomatik olarak kurmak için:

```bash
curl -sSL https://raw.githubusercontent.com/yourusername/crypto-strategy-optimizer/main/install.sh | bash
```

Veya repository'yi klonladıysanız:

```bash
chmod +x install.sh && ./install.sh
```

Bu komut:
- GPU'nuzu kontrol eder
- GPU varsa CUDA 12.6 ve cuDNN kurar
- Python sanal ortamını oluşturur
- TensorFlow'u (GPU varsa GPU destekli) kurar
- Tüm gereksinimleri kurar
- Kurulumu doğrular

## Manuel Kurulum (İsteğe Bağlı)

<details>
<summary>Manuel kurulum adımlarını görmek için tıklayın</summary>

### Adım 1: GPU Kurulumu (İsteğe Bağlı ancak Önerilir - NVIDIA GPU'nuz varsa İLK OLARAK bunu yapın)

#### 1.1 GPU Durumunu Kontrol Edin
```bash
nvidia-smi
```

GPU'nuzu görüyorsanız, CUDA kurulumuna devam edin. Görmüyorsanız, Adım 2'ye geçin.

#### 1.2 CUDA Toolkit Kurulumu
```bash
# CUDA keyring'i indir
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Paket listesini güncelle ve CUDA 12.6'yı kur
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

#### 1.3 cuDNN Kurulumu
```bash
# CUDA 12 için cuDNN 9 kur
sudo apt-get install libcudnn9-cuda-12
```

#### 1.4 Ortam Değişkenlerini Yapılandırma
```bash
# CUDA'yı PATH ve LD_LIBRARY_PATH'e ekle
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Değişiklikleri uygula
source ~/.bashrc

# CUDA kurulumunu doğrula
nvcc --version
```

### Adım 2: Python Ortamı Kurulumu

#### 2.1 Sanal Ortam Oluşturma
```bash
python3 -m venv .venv_wsl
```

#### 2.2 Sanal Ortamı Aktifleştirme
```bash
source .venv_wsl/bin/activate
```

### Adım 3: Bağımlılıkları Kurma

#### 3.1 Temel Kurulum (Sadece CPU)
```bash
pip install -r requirements.txt
```

#### 3.2 GPU Destekli Kurulum (Adım 1'i tamamladıysanız)
```bash
# Önce varsa CPU-only TensorFlow'u kaldır
pip uninstall tensorflow -y

# GPU destekli TensorFlow kur
pip install tensorflow[and-cuda]

# Diğer gereksinimleri kur
pip install -r requirements.txt
```

### Adım 4: Kurulumu Doğrulama

#### 4.1 Python Bağımlılıklarını Doğrulama
```bash
python -c "import pandas, numpy, tensorflow; print('Tüm paketler başarıyla içe aktarıldı')"
```

#### 4.2 GPU Kurulumunu Doğrulama (varsa)
```bash
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'Kullanılabilir GPU Sayısı: {len(gpus)}'); print('GPU İsimleri:', gpus)"
```

GPU algılanırsa, şöyle bir çıktı görmelisiniz:
```
Kullanılabilir GPU Sayısı: 1
GPU İsimleri: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

</details>

## Hızlı Başlangıç

### TensorFlow GPU Optimizasyonu ile Çalıştırma (En Hızlı - GPU Gerektirir)
```bash
# Tekil stratejiler
python tests/backtest_runner.py --rsi --tensorflow-gpu
python tests/backtest_runner.py --macd --tensorflow-gpu
python tests/backtest_runner.py --ema --tensorflow-gpu

# Tüm stratejiler sırayla
python tests/backtest_runner.py --all-tensorflow-gpu
```

### Temel Grid Search ile Çalıştırma (CPU'da Çalışır)
```bash
# Tekil stratejiler
python tests/backtest_runner.py --rsi --basic-strategy
python tests/backtest_runner.py --macd --basic-strategy
python tests/backtest_runner.py --ema --basic-strategy

# Tüm stratejiler
python tests/backtest_runner.py --all --basic-strategy
```

### Random Search ile Çalıştırma (CPU'da Çalışır)
```bash
python tests/backtest_runner.py --rsi --random-search
```

## Proje Yapısı

```
├── strategies/          # Ticaret stratejileri (RSI, MACD, EMA)
├── tests/              # Test çalıştırıcı ve yardımcı araçlar
├── data/               # Piyasa verileri (BTC, ETH, SOL)
├── reports/            # Oluşturulan SVG raporları
├── requirements.txt    # Python bağımlılıkları
└── CLAUDE.md           # AI asistan talimatları
```

## Desteklenen Stratejiler

- **RSI**: Göreceli Güç Endeksi (Relative Strength Index)
- **MACD**: Hareketli Ortalama Yakınsama Iraksama (Moving Average Convergence Divergence)
- **EMA**: Üstel Hareketli Ortalama (Exponential Moving Average)

## Optimizasyon Yöntemleri

- **TensorFlow GPU**: Parametre üretimi için GPU kullanan hibrit optimizasyon (en hızlı)
- **Grid Search**: Sistematik parametre alanı keşfi
- **Random Search**: Stokastik parametre keşfi

## Sorun Giderme

### GPU Algılanmıyor
TensorFlow GPU'nuzu algılamıyorsa:

1. CUDA ve cuDNN'nin düzgün kurulduğundan emin olun:
```bash
nvcc --version  # CUDA 12.6 göstermeli
nvidia-smi      # GPU'nuzu göstermeli
```

2. TensorFlow GPU uyumluluğunu kontrol edin:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

3. TensorFlow'u GPU desteği ile yeniden kurun:
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

### Bellek Sorunları
Bellek yetersizliği hatası alırsanız:
- Optimizer yapılandırmalarında batch boyutunu azaltın
- CPU tabanlı stratejileri kullanın (--basic-strategy veya --random-search)

## Sanal Ortamı Devre Dışı Bırakma

İşiniz bittiğinde sanal ortamı devre dışı bırakın:
```bash
deactivate
```