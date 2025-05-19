import torch
import numpy as np
import os, sys
from dataset.ADL import ADLDataset

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.Model import Model
import math  # Necesario para el scheduler
import utils.save as sv
from utils.evaluate import evaluation  # Función de evaluación adaptada para multi-label
from utils.control_seed import worker_init_fn
import os
import wandb


def train_model(config, train_ds, val_ds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(config, input_dim=768, output_dim=32, dropout=config.train_params.dropout)

    



    


    if config.classifier.train_weights_path is not None:
        if os.path.isfile(config.classifier.train_weights_path):
            print("loading checkpoint '{}'".format(config.classifier.train_weights_path))
            checkpoint = torch.load(config.classifier.train_weights_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint

    model.to(device)
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.workers, 
        shuffle=True, 
        worker_init_fn=worker_init_fn
    )

    # Para multi-label se utiliza BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss()

    # Condición para aplicar lr decay: Si config.train_params.lr_decay es True se crea el scheduler.
    if hasattr(config.train_params, "lr_decay") and config.train_params.lr_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.train_params.lr_decay_every, 
            gamma=config.train_params.lr_decay_factor
        )
    else:
        scheduler = None


    val_loader = DataLoader(
        val_ds, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.workers, 
        shuffle=False, ### Si pongo esto a True obtengo un 60% de acierto con el MLP
        worker_init_fn=worker_init_fn
    )
    
    epochs = config.train_params.epochs
    best_metric_top1 = 0.0  # Se usará la top1 como métrica principal
    best_metric_top5 = 0.0  # Se usará la top5 como secundaria
    best_loss = float('inf')

    if config.train_params.optimizer=='adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.train_params.lr, 
            weight_decay=config.train_params.weight_decay
        )
    elif config.train_params.optimizer=='sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.train_params.lr, 
            momentum=0.9, 
            nesterov=True, 
            weight_decay=config.train_params.weight_decay
        )
    else:
        raise ValueError(f"Optimizer {config.model.train} not supported. Use 'adamw' or 'sgd'.")
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        current_noise_std = 0.2 #* (1 - e / epochs)
        for batch, (video_features, labels, frames) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            # Añadir ruido para reducir overfitting
            features_noisy = video_features + torch.randn_like(video_features) * current_noise_std
            
            # Convertir las etiquetas a float para BCEWithLogitsLoss
            labels = labels.to(device).float()

            output = model(features_noisy, frames)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            # Actualizar el scheduler si está configurado
            if scheduler is not None:
                scheduler.step()
            
            print(f"Training loss: {running_loss / len(train_loader)}")
            if not os.path.exists(config.save_dir):
                os.makedirs(config.save_dir)
            if (e+1) % config.train_params.save_every == 0:
                sv.save_epoch(e, model, optimizer, config.save_dir, f"epoch{e}.pt", config)

            avg_loss, top1_acc, top5_acc = evaluation(model, val_loader, criterion)
            wandb.log({"epoch": e, "top1_acc": top1_acc, "top5_acc": top5_acc, "loss": avg_loss})
            # Se utiliza top1_acc para determinar el mejor modelo
            if top1_acc > best_metric_top1 or (top1_acc == best_metric_top1 and top5_acc > best_metric_top5):
                best_metric_top1 = top1_acc
                best_metric_top5 = top5_acc
                best_loss = avg_loss
                
                sv.save_epoch(e, model, optimizer, config.save_dir, "best_epoch.pt", config)
            print(f"[{e}] Validation: Top1 ({top1_acc}), Top5 ({top5_acc}), Mean Loss ({avg_loss}), Best Top1 ({best_metric_top1}), Best Top5 ({best_metric_top5})")
    
    sv.save_best_res(best_metric_top1, best_loss, config.save_dir)

