# TP1 CUDA

## notes

`/net/cremi/amguermouche/coursCuda/exo1`

<https://cours-ag.gitlabpages.inria.fr/cisd-cuda/>

La latence est le délai entre le moment où une opération est initiée, et le moment où ses effets deviennent détectable.

Throughput (débit) est la quantité de travail effectué sur une durée.

## exo1

```bash
./deviceQuery
There is 1 device supporting CUDA

Device 0: "NVIDIA GeForce RTX 2070"
  CUDA Driver Version / Runtime Version          11.4 / 11.2
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 4066508800 bytes
MapSMtoCores for SM 7.5 is undefined.  Default to use 128 Cores/SM
MapSMtoCores for SM 7.5 is undefined.  Default to use 128 Cores/SM
  (36) Multiprocessors, (128) CUDA Cores/MP:     4608 CUDA Cores
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum number of threads per multiprocessor:  1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.71 GHz
  Memory clock rate:                             7.00 GHz
  Memory Bus Width:                              256 bits
  Concurrent copy and execution:                 Yes

Test PASSED
```

`nvcc helloWorld.cu -o helloWorld`

## exo2

Done.

## exo3

Done.

## exo4

Done.

## exo5
