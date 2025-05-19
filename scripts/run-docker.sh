#!/bin/bash

export containerName=tfg2_$USER
docker run -d --gpus '"device=0"' --rm -it \
  --volume="/home/hhernandez/workspace:/workspace:rw" \
  --volume="/mnt/md1/datasets/ADL/:/dataset:ro" \
  --volume="/home/hhernandez/features:/features:rw" \
  --volume="/home/hhernandez/results:/results:rw" \
  --workdir="/workspace" \
  --shm-size=16g \
  --memory=16g \
  --name $containerName \
  hhernandez/tfg:latest bash
