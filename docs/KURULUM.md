# Kurulum Kılavuzu

Bu dokümanda Crypto Trading Strategy Optimizer'ın detaylı kurulum adımları anlatılmaktadır.

## 📋 Sistem Gereksinimleri

### Minimum Gereksinimler
- **İşletim Sistemi**: Ubuntu 20.04+ veya WSL2 (Windows)
- **Python**: 3.8 veya üzeri
- **RAM**: 8 GB
- **Disk Alanı**: 2 GB

### Önerilen Gereksinimler
- **GPU**: NVIDIA GPU (CUDA 12.0+ destekli)
- **RAM**: 16 GB
- **İşletim Sistemi**: WSL2 üzerinde Ubuntu 22.04

## 🚀 Otomatik Kurulum (Önerilen)

Tüm bağımlılıkları otomatik olarak kuran script'imizi kullanabilirsiniz:

```bash
# Repository'yi klonlayın
git clone https://github.com/wkaandemir/crypto-strategy-optimizer.git
cd crypto-strategy-optimizer

# Kurulum script'ini çalıştırın
chmod +x install.sh
./install.sh
```

Bu script otomatik olarak:
- ✅ GPU varlığını kontrol eder
- ✅ CUDA 12.6 ve cuDNN kurar (GPU varsa)
- ✅ Python sanal ortamını oluşturur
- ✅ TensorFlow'u GPU desteği ile kurar
- ✅ Tüm Python bağımlılıklarını kurar
- ✅ Kurulumu doğrular

## 🔧 Manuel Kurulum

### Adım 1: Repository'yi Klonlayın

```bash
git clone https://github.com/wkaandemir/crypto-strategy-optimizer.git
cd crypto-strategy-optimizer
```

### Adım 2: GPU Kurulumu (İsteğe Bağlı)

#### 2.1 GPU Kontrolü
```bash
nvidia-smi
```

GPU görünmüyorsa, Adım 3'e geçin.

#### 2.2 CUDA Toolkit Kurulumu
```bash
# CUDA keyring'i indir
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# CUDA 12.6'yı kur
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

#### 2.3 cuDNN Kurulumu
```bash
sudo apt-get install libcudnn9-cuda-12
```

#### 2.4 Ortam Değişkenleri
```bash
# PATH'e CUDA ekle
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Doğrulama
nvcc --version
```

### Adım 3: Python Sanal Ortamı

```bash
# Sanal ortam oluştur
python3 -m venv .venv_wsl

# Aktifleştir
source .venv_wsl/bin/activate

# pip'i güncelle
pip install --upgrade pip
```

### Adım 4: Bağımlılıkları Kurma

#### CPU Kurulumu
```bash
pip install -r requirements.txt
```

#### GPU Kurulumu
```bash
# Önce CPU TensorFlow'u kaldır (varsa)
pip uninstall tensorflow -y

# GPU destekli TensorFlow kur
pip install tensorflow[and-cuda]

# Diğer bağımlılıkları kur
pip install -r requirements.txt
```

### Adım 5: Kurulum Doğrulama

#### Python Paketleri
```bash
python -c "import pandas, numpy, tensorflow; print('✅ Tüm paketler yüklü')"
```

#### GPU Doğrulama
```bash
python -c "import tensorflow as tf; \
gpus = tf.config.list_physical_devices('GPU'); \
print(f'GPU Sayısı: {len(gpus)}'); \
if gpus: print('✅ GPU hazır:', gpus[0])"
```

## 🐧 WSL2 Kurulumu (Windows Kullanıcıları)

### WSL2'yi Etkinleştirme

1. PowerShell'i yönetici olarak açın:
```powershell
# WSL'i etkinleştir
wsl --install

# Ubuntu kur
wsl --install -d Ubuntu-22.04
```

2. Bilgisayarı yeniden başlatın

3. Ubuntu'yu açın ve kullanıcı oluşturun

### WSL2'de GPU Desteği

1. [NVIDIA WSL2 Driver](https://developer.nvidia.com/cuda/wsl) kurulumu yapın

2. WSL2 içinde GPU'yu kontrol edin:
```bash
nvidia-smi
```

## 🍎 macOS Kurulumu

> ⚠️ **Not**: macOS'ta GPU hızlandırma desteklenmez, sadece CPU modunda çalışır.

```bash
# Homebrew ile Python kur
brew install python@3.11

# Sanal ortam oluştur
python3 -m venv .venv_mac
source .venv_mac/bin/activate

# Bağımlılıkları kur
pip install -r requirements.txt
```

## 📦 Docker Kurulumu (Alternatif)

```bash
# Docker image'ı oluştur
docker build -t crypto-optimizer .

# Container'ı çalıştır
docker run --gpus all -it crypto-optimizer
```

## ✅ Kurulum Testi

Kurulumun başarılı olduğunu test etmek için:

```bash
# Basit bir backtest çalıştır
python tests/backtest_runner.py --rsi --basic-strategy
```

Çıktıda sonuçlar görünüyorsa kurulum başarılıdır!

## 🔄 Güncelleme

```bash
# Repository'yi güncelle
git pull

# Sanal ortamı aktifleştir
source .venv_wsl/bin/activate

# Bağımlılıkları güncelle
pip install -r requirements.txt --upgrade
```

## 🗑️ Kaldırma

```bash
# Sanal ortamı sil
rm -rf .venv_wsl

# CUDA'yı kaldır (isteğe bağlı)
sudo apt-get remove cuda-toolkit-12-6
sudo apt-get autoremove
```

## ❓ Yardım

Kurulum sırasında sorun yaşıyorsanız:
1. [SORUN_GIDERME.md](SORUN_GIDERME.md) dosyasına bakın
2. GitHub'da issue açın
3. Hata mesajını ve sistem bilgilerinizi paylaşın