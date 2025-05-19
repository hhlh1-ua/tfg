import torch.nn as nn
import torch.nn.functional as F
import torch
from models.Classifier import Classifier

class Fusion_Class(nn.Module):
    def __init__(self, config, video_encoder_dim=768, object_encoder_dim=768):
        super().__init__()
        self.fusion_strategy = config.model.fusion_strategy
        self.obj_embedding_size = config.object_detector.embedding_size
        self.linear_video = nn.Sequential(nn.Linear(video_encoder_dim, 768), nn.LayerNorm(768))

        if self.fusion_strategy == 'concat':
            self.object_recopilation_strategy = config.object_detector.object_recopilation_strategy
            if self.object_recopilation_strategy == 'all_frames':
                self.classifier_input_dim = video_encoder_dim * ((config.video_model.block_size//2) * config.object_detector.max_detected_objects + 1)
            elif self.object_recopilation_strategy in ['1stframe', 'middleframe']:
                self.classifier_input_dim = video_encoder_dim * (config.object_detector.max_detected_objects+ 1)
            else:
                raise ValueError("Estrategia de recogilación de objetos no soportada")
        elif self.fusion_strategy in ['mean', 'sum']:

            self.classifier_input_dim = video_encoder_dim
            self.object_recopilation_strategy = config.object_detector.object_recopilation_strategy
        else:
            raise ValueError("Estrategia de fusión no soportada: use 'concat', 'mean' o 'sum'.")

    def forward(self, visual_features, object_features):
        # visual_features: [batch, video_encoder_dim]
        # object_features puede tener distinta forma dependiendo de la estrategia de recogida:
        # Por ejemplo: para 'all_frames' se espera [batch, n_frames, max_objects, object_encoder_dim]
        # Para las demás, [batch, max_objects, object_encoder_dim]

        # Expandir visual_features a dimensión secuencial
        visual_features = self.linear_video(visual_features)
        visual_features = visual_features.unsqueeze(1)  # [batch, 1, video_encoder_dim]

        if self.object_recopilation_strategy == 'all_frames':
            object_features = object_features.view(object_features.size(0), -1, object_features.size(-1))
            # [batch, n_frames * max_objects, object_encoder_dim]

        features = torch.cat((visual_features, object_features), dim=1)  # [batch, n_total, encoder_dim]

        if self.fusion_strategy == 'concat':
            final_tensor = features.view(features.size(0), -1)  # [batch, n_total * encoder_dim]
        elif self.fusion_strategy == 'mean':
            final_tensor = features.mean(dim=1)  # [batch, encoder_dim]
        elif self.fusion_strategy == 'sum':
            final_tensor = features.sum(dim=1)  # [batch, encoder_dim]
        else:
            raise ValueError("Estrategia de fusión no implementada.")

        return final_tensor
