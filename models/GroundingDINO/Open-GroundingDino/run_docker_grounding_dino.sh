#!/bin/bash

export containerName=tfg_$USER
docker run -d --gpus '"device=0"' --rm -it \
  --volume="/home/hhernandez/workspace:/workspace:rw" \
  --volume="/mnt/md1/datasets/ADL/:/dataset:ro" \
  --workdir="/workspace" \
  --shm-size=16g \
  --memory=16g \
  --name $containerName \
  hhernandez/open_grounding_dino:latest bash
