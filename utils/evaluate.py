import torch
import numpy as np
import os, sys
from dataset.ADL import ADLDataset

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.Model import Model



@torch.no_grad()
def evaluation(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.0
    total_top1_correct = 0
    total_top5_correct = 0
    total_samples = 0

    for batch, (video_features, labels, frames) in enumerate(tqdm(dataloader)):

        labels = labels.to(device)  #  tensor multi-hot  [batch_size, num_classes]

        outputs = model(video_features, frames)  # [batch_size, num_classes]
        loss = criterion(outputs, labels)
        total_loss += loss.item() * video_features.size(0) ## multiplica el loss por el número de elementos dentro del batch que hay

        # 5 predicciones con mayor score para cada muestra
        _, pred_top5 = outputs.topk(5, dim=1)  # tamaño: [batch_size, 5]
        # print(pred_top5.shape)

        # Top1: Ver si la etiqueta con mayor score está entre las verdaderas (si hay 2 y la predicción está entre estas dos, se considera correcta)
        _, pred_top1 = outputs.topk(1, dim=1)  # tamaño: [batch_size, 1]
        # print(pred_top1)
        # print(labels)
        # Usamos gather para obtener el valor (0 o 1) de la etiqueta predicha. Se pone 1 porque en el tensor tenemos 32 columnas. Pred_top1 tiene el índice de cada columna a coger
        top1_matches = labels.gather(1, pred_top1) ## Si el valor de la columna es 0, es que no existe sa etiqueta. Si es 1 sí que existe.
        # print(top1_matches)
        total_top1_correct += (top1_matches > 0).sum().item() ## Solo se suma si es 1

        # Top5: Verificar si alguna de las 5 predicciones se encuentra en las verdaderas
        top5_matches = labels.gather(1, pred_top5)  # tamaño: [batch_size, 5]
        total_top5_correct += (top5_matches > 0).any(dim=1).sum().item() ## El any aquí se añade para evitar que sume como doble predicción correcta si en el top5 están las dos clases que pueden existir.

        total_samples += video_features.size(0)

    avg_loss = total_loss / total_samples
    top1_acc = total_top1_correct / total_samples
    top5_acc = total_top5_correct / total_samples

    return avg_loss, top1_acc, top5_acc





def test_model(config, test_ds):
    model=Model(config, input_dim=768, output_dim=32,dropout=config.train_params.dropout)
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss()
    if config.classifier.test_weights_path is not None:
        if os.path.isfile(config.classifier.test_weights_path):
            print(("loading checkpoint '{}'".format(config.classifier.test_weights_path)))
            checkpoint = torch.load(config.classifier.test_weights_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            raise Exception ("no checkpoint found at '{}'".format(config.classifier.test_weights_path))

    model.to(device)
    return evaluation(model, test_loader, criterion) 


    