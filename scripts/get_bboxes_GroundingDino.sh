#!/bin/bash
python3 ../create_file_gdino.py --config ../config/defaults.yaml --output ../models/GroundingDINO/Open-GroundingDino/ADL/Key_Frames.json
python3 ../models/GroundingDINO/Open-GroundingDino/ADL/create_coco.py --frames_file ../models/GroundingDINO/Open-GroundingDino/ADL/Key_Frames.json --catfile ../models/GroundingDINO/Open-GroundingDino/config/label_map_ADL.json  --outdir ../models/GroundingDINO/Open-GroundingDino/ADL/Key_frames_Annotated.json
cd ../models/GroundingDINO/Open-GroundingDino && bash get_ADL_bboxes.sh