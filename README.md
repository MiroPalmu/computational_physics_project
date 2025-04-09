# Compilation

```shell
CXX=$HOME/data/scratch/clang-git/install/bin/clang++ cmake -Bbuild -GNinja -DAdaptiveCpp_DIR=$HOME/data/scratch/clang-git/acpp-git/install/lib/cmake/AdaptiveCpp .
cd build
ninja
LD_LIBRARY_PATH=~/data/scratch/clang-git/install/lib/x86_64-unknown-linux-gnu/ ./main
```

