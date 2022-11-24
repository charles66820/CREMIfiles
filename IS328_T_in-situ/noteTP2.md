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

run :

```bash
OLD_PWD=`pwd`
cd $HOME/projects/TPs_In_Situ/build
source setup-env.sh
cd $OLD_PWD
unset $OLD_PWD
```

## exec

```bash
salloc -proutage -n1 -c8 /bin/env OMP_NUM_THREADS=8 $HOME/projects/TPs_In_Situ/build/xstampv2 tutorial_insitu_histo2_par_freq.msp --profiling-vite traceHisto4.vite
```

## Exercice 1

1.1. 
