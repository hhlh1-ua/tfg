import os
import torch
import yaml
from dotmap import DotMap

def save_epoch(epoch, model, optimizer, work_dir, filename, config):
    # Guardar el checkpoint en un archivo .pt
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(work_dir, filename))
    
    # Guardar el archivo de configuración en formato YAML
    config_path = os.path.join(work_dir, 'config.yaml')  # Puedes cambiar el nombre si lo deseas
    with open(config_path, 'w') as f:
        yaml.safe_dump(config.toDict(), f, default_flow_style=False)

def save_best_res(result, loss, work_dir, filename="best_res.yaml"):
    best_info = {
        "best_precision": result,
        "loss": loss,
    }
    file_path = os.path.join(work_dir, filename)
    with open(file_path, 'w') as f:
        yaml.safe_dump(best_info, f, default_flow_style=False)
