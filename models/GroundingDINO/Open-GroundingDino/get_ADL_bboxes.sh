GPU_NUM=1

CFG="/workspace/tfg_hhernandez/models/GroundingDINO/Open-GroundingDino/config/cfg_Best_GD_ADL.py"
DATASETS="/workspace/tfg_hhernandez/models/GroundingDINO/Open-GroundingDino/config/dataset_inference_ADL.json"
# OUTPUT_DIR="/workspace/GroundingDINO/Open-GroundingDino/resultados/pre_finetunning/Version_Oficial"
# OUTPUT_DIR="/workspace/GroundingDINO/Open-GroundingDino/resultados/pre_finetunning/Version_FineTunneada_en_COCO"
OUTPUT_DIR="/features/objects/GD/bboxes"
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
        --pretrain_model_path /workspace/GroundingDINO/modelos/new_dist_fine_tunning/1_8M_params/GROfficialStrategyBatch8/checkpoint_best_regular.pth \
        --save_results
