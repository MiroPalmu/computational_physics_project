# Dockerfile

```
FROM gcc:15.1


apt-get update && apt-get install -y \
    lsb-release \
    wget \
    software-properties-common \
    gnupg

apt-add-repository -y --component non-free
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 20

apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    libboost-fiber-dev \
    libboost-context-dev \
    libomp-20-dev \
    libclang-20-dev \
    clang-tools-20 \
    llvm-20-dev \
    lld-20

# Acpp
# clone acpp
cmake -DCMAKE_CXX_COMPILER=clang++-20 -GNinja -Bbuild .

#  cmake
wget https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-linux-x86_64.tar.gz
tar xf cmake-4.0.1-linux-x86_64.tar.gz

# Project
# clone project
CXX=clang++-20 ~/cmake-4.0.1-linux-x86_64/bin/cmake -Bbuild -GNinja -DCMAKE_CXX_FLAGS="--gcc-toolchain=/usr/local/ -fconstexpr-steps=10000000 -fPIC" .

```

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
