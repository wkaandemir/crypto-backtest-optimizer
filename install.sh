#!/bin/bash

echo "================================================"
echo "Crypto Trading Strategy Optimizer - Kurulum"
echo "================================================"

# Renkleri tanımla
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# GPU kontrolü
echo -e "\n${YELLOW}🔍 GPU kontrolü yapılıyor...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ NVIDIA GPU algılandı${NC}"
        GPU_AVAILABLE=true
    else
        echo -e "${YELLOW}⚠️ GPU bulunamadı, CPU modunda devam ediliyor${NC}"
        GPU_AVAILABLE=false
    fi
else
    echo -e "${YELLOW}⚠️ nvidia-smi bulunamadı, CPU modunda devam ediliyor${NC}"
    GPU_AVAILABLE=false
fi

# APT sources'ı düzelt (Debian bookworm için)
echo -e "\n${YELLOW}📦 APT repository'leri düzeltiliyor...${NC}"
if grep -q "stable" /etc/apt/sources.list; then
    echo "APT sources düzeltiliyor (bookworm'a sabitleniyor)..."
    sudo sed -i 's/stable/bookworm/g' /etc/apt/sources.list
    sudo sed -i 's/oldstable/bullseye/g' /etc/apt/sources.list
fi

# Sistem paketlerini güncelle
echo -e "\n${YELLOW}📦 Sistem paketleri güncelleniyor...${NC}"
sudo apt-get update --allow-releaseinfo-change -y
sudo apt-get install -y python3 python3-pip python3-venv wget curl

# GPU varsa CUDA kur
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "\n${YELLOW}🚀 CUDA Toolkit kurulumu başlatılıyor...${NC}"
    
    # CUDA'nın kurulu olup olmadığını kontrol et
    if ! command -v nvcc &> /dev/null; then
        echo "CUDA Toolkit kuruluyor..."
        
        # CUDA keyring'i indir ve kur
        if [ ! -f cuda-keyring_1.1-1_all.deb ]; then
            wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        fi
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        
        # CUDA Toolkit 12.6'yı kur
        sudo apt-get update
        sudo apt-get -y install cuda-toolkit-12-6
        
        # cuDNN'i kur
        echo "cuDNN kuruluyor..."
        sudo apt-get -y install libcudnn9-cuda-12
        
        # Environment variables'ları ayarla
        echo -e "\n${YELLOW}⚙️ CUDA ortam değişkenleri ayarlanıyor...${NC}"
        echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        export PATH=/usr/local/cuda-12.6/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
        
        # Temizlik
        rm -f cuda-keyring_1.1-1_all.deb*
        
        echo -e "${GREEN}✅ CUDA kurulumu tamamlandı${NC}"
    else
        echo -e "${GREEN}✅ CUDA zaten kurulu${NC}"
        nvcc --version
    fi
fi

# Python sanal ortamını oluştur
echo -e "\n${YELLOW}🐍 Python sanal ortamı oluşturuluyor...${NC}"
if [ ! -d ".venv_wsl" ]; then
    python3 -m venv .venv_wsl
    echo -e "${GREEN}✅ Sanal ortam oluşturuldu${NC}"
else
    echo -e "${GREEN}✅ Sanal ortam zaten mevcut${NC}"
fi

# Sanal ortamı aktifleştir
source .venv_wsl/bin/activate

# pip'i güncelle
echo -e "\n${YELLOW}📦 pip güncelleniyor...${NC}"
pip install --upgrade pip

# TensorFlow kurulumu
echo -e "\n${YELLOW}📦 Python paketleri kuruluyor...${NC}"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "GPU destekli TensorFlow kuruluyor..."
    pip uninstall -y tensorflow tensorflow-cpu
    pip install tensorflow[and-cuda]
else
    echo "CPU-only TensorFlow kuruluyor..."
    pip install tensorflow
fi

# Diğer gereksinimleri kur
pip install -r requirements.txt

# Kurulumu doğrula
echo -e "\n${YELLOW}🔍 Kurulum doğrulanıyor...${NC}"
echo ""

# Python paketlerini kontrol et
python -c "import pandas, numpy, tensorflow, ta, matplotlib; print('✅ Tüm Python paketleri başarıyla kuruldu')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Bazı Python paketleri eksik${NC}"
else
    echo -e "${GREEN}✅ Python paketleri OK${NC}"
fi

# GPU kontrolü (varsa)
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "\n${YELLOW}🎮 GPU durumu kontrol ediliyor...${NC}"
    python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✅ TensorFlow GPU algıladı: {len(gpus)} GPU')
    for gpu in gpus:
        print(f'   - {gpu}')
else:
    print('⚠️ TensorFlow GPU algılayamadı')
" 2>/dev/null
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}✅ Kurulum tamamlandı!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Kullanım:"
echo "1. Sanal ortamı aktifleştirin:"
echo "   source .venv_wsl/bin/activate"
echo ""
echo "2. Backtest çalıştırın:"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "   python tests/backtest_runner.py --rsi --tensorflow-gpu"
else
    echo "   python tests/backtest_runner.py --rsi --basic-strategy"
fi
echo ""
echo "Tüm komutlar için README.md'ye bakın."