# CISD mini-chameleon

## Sujet

<https://solverstack.gitlabpages.inria.fr/mini-examples/mini-chameleon/>

## Code source

<https://gitlab.inria.fr/solverstack/mini-examples/mini-chameleon>

### Get code

```bash
git clone --recursive https://gitlab.inria.fr/solverstack/mini-examples/mini-chameleon.git
cd mini-chameleon
```

## guix

Define custom `guix` channels in `~/.config/guix/channels.scm`.

```bash
cp ./channels.scm ~/.config/guix/channels.scm
```

Pull project env.

```bash
# guix build hello
guix pull --allow-downgrades
hash guix
guix describe --format=channels
```

## Env for `mini-chameleon`

An env with `bash`, `emacs`, `nano` and `vim` that start with the command `bash --norc`. `--norc` disable the `.bashrc` execution.

```bash
guix shell --pure \
-D mini-chameleon --with-input=openblas=mkl \
bash emacs nano vim \
-- bash --norc
```

Build env

```bash
guix shell --pure \
bash emacs nano vim cmake coreutils gdb gcc-toolchain grep make mkl openmpi openssh sed valgrind \
-- bash --norc
```

Build env with vecto `gcc-toolchain`

```bash
guix shell --pure \
-D mini-chameleon --with-input=openblas=mkl \
bash emacs nano vim gcc-toolchain \
-- bash --norc
```

Build env with vecto `clang-toolchain`

```bash
guix shell --pure \
-D mini-chameleon --with-input=openblas=mkl \
bash emacs nano vim clang-toolchain \
-- bash --norc
```

## Guix ajout d'éditeur (vim et nano)

```bash
guix environment --pure \
chameleon --with-input=openblas=mkl --ad-hoc \
bash emacs nano vim \
-- bash --norc
```

## Use of basic intrinsics

```cmake
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=haswell")
```

## Guix notes

```bash
guix shell <moduleList> -- <startCmd>
```

## Build

```bash
mkdir -p build && cd build
cmake .. -DENABLE_MPI=ON -DENABLE_STARPU=ON
make -j
```
