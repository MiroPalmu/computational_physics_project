Report is located in `report.ipynb` and it only requires Numpy and Matplotlib.
Simulation code is scattered in different header and source files.

This document describes how to build Docker image in which
the simulation can be build, compiled and executed.

All the analysis in the report is done on the data from `data/` folder,
so compiling and running the simulation is not neccesseary for the report.

# Implementation detail
## idg

This project uses my previously written header only library called idg.
All of the functionality from idg is in `idg/~ folder`.
Main thing from it is the `einsum` functionality which allows me to write
einstein summation in C++.

Idg is written in C++26 which requires clang 20 or gcc 15.

## SYCL

Initially the code was supposed to be 3D and run on GPU using SYCL [1].
Due to `idg` being C++26 library I had to use AdpativeCpp with clang 20 [2].
However, due to my graphics card being too old for the features that AdaptiveCpp uses,
I had to switch to CPU computing only (*).

This meant that I had to switch from 3D simulation to 1D simulation
inorder to make the computational cost feasible.
Additionally there were some technical problems with my approach
which would have required a substantial refactoring.

Due to these circumstances, the code contains multiple unoptimal solutions
and is not very efficient.

[1] https://www.khronos.org/sycl/
[2] https://adaptivecpp.github.io/

(*) Theoretically simulation can run on newer Nvidia GPUs inside the provided image
    if docker is configured with Nvidia runtime:

    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

    `--runtime=nvidia --gpus all` is passed to `docker run` command
    and `sycl::cpu_selector_v` is switched to `sycl::gpu_selector_v` in `main.cpp`.


# Compiling the simulation

Project comes with `./Dockerfile` from which image 
containing all the required dependencies can be build.

Shell commands blow follow convention that commands starting with `#`
are executed as root privileges (downside of Docker).
They also assume that docker is installed and daemon is running.

To build the image run following command in `./image/` directory:

```shell
# docker build -t comp-phys-proj .
```

Build image can be run with:

```shell
# docker run --rm -it comp-phys-proj
```

This drops user to Bash prompt.
Now the simulation can be compiled with:

```shell
> cd /root/computational_physics_project/release-build
> ninja main
```

This produces `/root/computational_physics_project/release-build/main` executable.

# Running the simulation

`main` takes two command line arguments.
First is grid points in x-direction and second is number of time steps:

```shell
> ./main <Nx> <time_steps>
```

The simulation outputs to `./output` directory.
Note that some ouputs are appended instead of written over.

## Parameters

Some of the simulation parameters can be controlled with enviroment variables.
For example following uses 0.5 as a time time step:

```shell
> DT=0.5 ./main <Nx> <time_steps>
```

All options:

```
SUBSTEPS=<integer>             (default: 2)
    Number of backwards euler substeps.

W_CLAMP=<decimal>              (default: 0.0001)
    Minimum value for W. 

DT=<decimal>                   (default: 0.001)
    Time step.

KREISS_COEFF=<decimal>         (default: 0.25)
    Kreiss-Oliger dissipation coefficient (epsilon).

KM=<decimal>                   (default: 0.025)
    Momentum constraint damping coefficient (k_m).

MINKOWSKI=<0 or 1>             (default: 0)
    Use gauge wave (0) or Minkowski (1) initial conditions.

ALGEBRAIC_CONSTRAINTS=<0 or 1> (default: 1)
    Enforce algebraic constraints (1) or not (0).
```

## Getting data out of Docker image

To analyze ouput data in Jupyter notebook the data can be copied from
running image with following command:

```shell
# docker cp <CONTAINER>:/root/computational_physics_project/release-build/output <DIR>
```

where `<CONTAINER>` is the hostname of the container
and `<DIR>` is output directory on host machine.

`<CONTAINER>` can be read from the command prompt inside the image.
For example container `02da973db447` would have following prompt:

```shell
root@02da973db447:~/computational_physics_project/release-build#
```
