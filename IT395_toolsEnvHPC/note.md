# Notes

## Mount fs with ssh

```bash
mkdir plafrimProjects
sshfs plafrim_CISD:/home/cisd-goedefr/projects plafrimProjects
```

```bash
fusermount -u plafrimProjects
```

## PlaFRIM

> slurm

```bash
sinfo # nodes list
salloc -p <partition> -N <nbNodes> # alloc some nodes
sbatch <batch script> # alloc some nodes
squeue [-u login name] # show allocated nodes
scontrol show jobid <job id>
scancel <job id>
```

```bash
srun -C haswell --time=03:00:00 --pty bash -i
srun -N 1 -n 12 --exclusive --time=03:00:00 --pty bash -i
salloc -proutage --exclusive -N4 -n4 -c24 mpirun ./prog
salloc -proutage -n1 -c24 make -j24
```

- `-p, --partition <partitionName>` partition name routage
- `-N, --nodes <nbNodes>` number of nodes
- `-n, --ntasks <nbTasks>` number of MPI process
- `-c, --cpus-per-task <nbCores>` Number de coeurs par process MPI
- `-x` exclusive

## Environnement list

- module
- spack
- easybuild
- conda
- guix

## module

```bash
module anvil # ?
module list # current loaded modules
module load <moduleName>
module unload <moduleName>
module switch <moduleName1> <moduleName2>
module show <moduleName> # display module info
module purge # unload all modules
```
