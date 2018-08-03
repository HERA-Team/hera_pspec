#!/bin/bash
#PBS -q hera
#PBS -V
#PBS -j oe
#PBS -o pspec_batch.out
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -l vmem=128gb, mem=128gb

# start script
echo "starting power spectrum pipeline: $(date)"

# copy parameter files to more discrete names
# and put a readonly lock on them
cp pspec_pipe.yaml .pspec_pipe.yaml
chmod 444 .pspec_pipe.yaml

# run scripts
pspec_pipe.py .pspec_pipe.yaml

# unlock and clean-up parameter files
chmod 775 .pspec_pipe.yaml
rm .pspec_pipe.yaml

# end script
echo "ending power spectrum pipeline: $(date)"
