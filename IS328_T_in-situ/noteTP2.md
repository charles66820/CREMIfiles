# Traitement in-situ TP2

## Init env & build

init :

```bash
$HOME/projects/TPs_In_Situ/exaStamp/scripts/configure-plafrim.sh
```

build :
> build path : `/home/cisd-goedefr/projects/TPs_In_Situ/build`

```bash
cd $HOME/projects/TPs_In_Situ/build
source setup-env.sh
salloc -proutage -n1 -c12 make -j24
```

Load env to RUN :

```bash
OLD_PWD=`pwd`
cd $HOME/projects/TPs_In_Situ/build
source setup-env.sh
cd $OLD_PWD
unset $OLD_PWD
```

## Exercice 1

1.1. `20` analyses and frequency is `1`.

```bash
salloc -proutage -n1 -c8 /bin/env OMP_NUM_THREADS=8 $HOME/projects/TPs_In_Situ/build/xstampv2 tutorial_insitu_histo2_par_freq.msp --profiling-vite traceHisto8.vite
```

1.2. ça vas plus vite

```bash
salloc -proutage -n1 -c12 /bin/env OMP_NUM_THREADS=12 $HOME/projects/TPs_In_Situ/build/xstampv2 tutorial_insitu_histo2_par_freq.msp --profiling-vite traceHisto12.vite
```

1.3. Il y a plus de check (les bars sont plus large)

1.4. Je sais pas

## Exercice 2

1.1.

```bash
salloc -proutage -n1 -c4 /bin/env OMP_NUM_THREADS=4 $HOME/projects/TPs_In_Situ/build/xstampv2 tutorial_insitu_histo2_space_sharing.msp --profiling-vite traceHistoSS4.vite
```

```bash
salloc -proutage -n1 -c8 /bin/env OMP_NUM_THREADS=8 $HOME/projects/TPs_In_Situ/build/xstampv2 tutorial_insitu_histo2_space_sharing.msp --profiling-vite traceHistoSS8.vite
```

1.2.

## Exercice 3

```bash
salloc -proutage -n1 -c8 /bin/env $HOME/projects/TPs_In_Situ/build/xstampv2 tutorial_insitu_histo_seq_naive_thread.msp --profiling-vite trace.vite
```
<!-- x86_64/bin/ustamp -->

## Exercice 4

```bash
salloc -proutage -n1 -c8 /bin/env OMP_NUM_THREADS=8 $HOME/projects/TPs_In_Situ/build/xstampv2 tutorial_insitu_histo_seq_3parts.msp --profiling-vite trace.vite
```
