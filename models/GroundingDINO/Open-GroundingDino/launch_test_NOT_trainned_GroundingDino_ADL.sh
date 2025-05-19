GPU_NUM=1

CFG="/workspace/GroundingDINO/Open-GroundingDino/config/cfg_ADL.py"
DATASETS="/workspace/GroundingDINO/Open-GroundingDino/config/dataset_test_ADL.json"
# OUTPUT_DIR="/workspace/GroundingDINO/Open-GroundingDino/resultados/pre_finetunning/Version_Oficial"
# OUTPUT_DIR="/workspace/GroundingDINO/Open-GroundingDino/resultados/pre_finetunning/Version_FineTunneada_en_COCO"
OUTPUT_DIR="/workspace/GroundingDINO/resultados_test/new_dist/pre_fine_tunning/Version_FineTunneada_en_COCO"
# /workspace/GroundingDINO/modelos/groundingdino_swint_ogc.pth
# /workspace/GroundingDINO/modelos/gdinot-coco-ft.pth
# /workspace/GroundingDINO/modelos/gdinot-1.8m-odvg.pth
# Eliminar --pretrain_model_path para ver como funciona antes de preentrenar sin checkpoint


NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python3 -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /workspace/GroundingDINO/modelos/gdinot-coco-ft.pth \
        --options text_encoder_type=/workspace/GroundingDINO/modelos/BERT \
        --save_results
