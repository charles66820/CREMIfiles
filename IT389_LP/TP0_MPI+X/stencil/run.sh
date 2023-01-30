#!/bin/bash

# Path vars
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SCRIPT_NAME=$(basename -s .sh -- "$SCRIPT_PATH")

# Global vars
CSV_DIR=$SCRIPT_DIR/csv
ITER=20
SIZEs=(25,30 50,60 75,90 100,120 125,150 150,180 175,210 200,240 225,270 250,300 275,330 300,360 325,390 350,420 375,450 400,480 425,510 450,540 475,570 500,600 625,750 750,900 1000,1200)
BLOCS_SIZEs=(8 16 32 64 128 512)

# disable the turbo boost
# echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

make -j

# $ITER
CSV_FILENAME=seq.csv
echo "steps,timeInµSec,height,width,nbCells,fpOpByStep,gigaflops,cellByS" > $CSV_DIR/$CSV_FILENAME

# ./stencil_seq >> $CSV_DIR/$CSV_FILENAME
# ./stencil_seq >> $CSV_DIR/$CSV_FILENAME
# ./stencil_seq >> $CSV_DIR/$CSV_FILENAME
# ./stencil_seq >> $CSV_DIR/$CSV_FILENAME
# ./stencil_seq >> $CSV_DIR/$CSV_FILENAME
# ./stencil_seq >> $CSV_DIR/$CSV_FILENAME

IFS=',';
for i in "${SIZEs[@]}";
do
    set -- $i
    make clean -s
    STENCIL_SIZE_X=$1 STENCIL_SIZE_Y=$2 make stencil_seq && ./stencil_seq >> $CSV_DIR/$CSV_FILENAME
done

CSV_FILENAME=halos.csv
echo "steps,timeInµSec,height,width,tiledW,tiledH,nbCells,fpOpByStep,gigaflops,cellByS" > $CSV_DIR/$CSV_FILENAME
for TW in "${BLOCS_SIZEs[@]}";
do
  for TH in "${BLOCS_SIZEs[@]}";
  do
    make clean -s
    TILE_WIDTH=$TW TILE_HEIGHT=$TH make stencil_halos && ./stencil_halos >> $CSV_DIR/$CSV_FILENAME
  done
done
