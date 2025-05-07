FROM gcc:15.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        lsb-release \
        wget \
        software-properties-common \
        gnupg \
    && apt-add-repository -y --component non-free \
    && apt-add-repository -y contrib non-free-firmware \
    && cd \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 20 \
    && apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        vim \
        ninja-build \
        libboost-fiber-dev \
        libboost-context-dev \
        libomp-20-dev \
        libclang-20-dev \
        clang-tools-20 \
        llvm-20-dev \
        lld-20 \
        dirmngr \
        ca-certificates \
        apt-transport-https \
        dkms \
        curl \
    && curl -fSsL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub \
        | gpg --dearmor \
        | tee /usr/share/keyrings/nvidia-drivers.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/nvidia-drivers.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' \
        | tee /etc/apt/sources.list.d/nvidia-drivers.list \
    && apt-get update && \
    apt-get install -y cuda-toolkit \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/AdaptiveCpp/AdaptiveCpp \
    && cd AdaptiveCpp \
    && cmake -DCMAKE_CXX_COMPILER=clang++-20 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DWITH_CUDA_BACKEND=ON -GNinja -Bbuild . \
    && cd build \
    && ninja \
    && ninja install \
    && cd \
    && rm -rf AdaptiveCpp \
    && wget https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-linux-x86_64.tar.gz \
    && tar xf cmake-4.0.1-linux-x86_64.tar.gz \
    && rm -f cmake-4.0.1-linux-x86_64.tar.gz \
    && git clone https://github.com/MiroPalmu/computational_physics_project \
    && cd computational_physics_project \
    && CXX=clang++-20 ../cmake-4.0.1-linux-x86_64/bin/cmake -Bdebug-build -GNinja -DCMAKE_CXX_FLAGS="--gcc-toolchain=/usr/local/ -fconstexpr-steps=10000000 -fPIC" -DCMAKE_BUILD_TYPE=Debug . \
    && CXX=clang++-20 ../cmake-4.0.1-linux-x86_64/bin/cmake -Brelease-build -GNinja -DCMAKE_CXX_FLAGS="--gcc-toolchain=/usr/local/ -fconstexpr-steps=10000000 -fPIC" -DCMAKE_BUILD_TYPE=Release .

CMD ["bash"]
