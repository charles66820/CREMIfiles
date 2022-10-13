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
srun -N 1 --time=03:00:00 --pty bash -i
```

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
