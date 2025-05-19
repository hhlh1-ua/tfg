#!/bin/bash
cd /workspace/GroundingDINO/Open-GroundingDino/models/GroundingDINO/ops/
python3 setup.py build install
python3 test.py
