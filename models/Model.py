import torch.nn as nn
import torch.nn.functional as F
import torch
from models.Classifier import Classifier
from models.Fusion import Fusion_Class
from models.ObjectEmbeddingsGetter import ObjectEmbeddingsGetter
from models.ObjectDetector import ObjectDetectorCreator
from transformers import ViTImageProcessorFast, ViTModel
from PIL import Image

class Model(nn.Module):
    def __init__(self, config, input_dim=768, output_dim=32, dropout=0.2):
        super().__init__()
        self.multimodal = config.model.multimodal
        mlp_hidden_dims=config.get('classifier', {}).get('hidden_dims', [256, 128, 64])
        if not self.multimodal:
            self.MLP = Classifier(hidden_dims=mlp_hidden_dims, input_dim=768, output_dim=output_dim, dropout=config.train_params.dropout)
        else:
            self.fusion_strategy = Fusion_Class(config)
            self.max_detected_objs=config.object_detector.max_detected_objects
            self.MLP = Classifier(hidden_dims=mlp_hidden_dims, input_dim=self.fusion_strategy.classifier_input_dim, output_dim=output_dim, dropout=config.train_params.dropout)

            self. object_detector = ObjectDetectorCreator.instatiate_ObjectDetector(config)

            self.ObjFTExtractor=ObjectEmbeddingsGetter(config, object_detector=self.object_detector)

    def forward(self, x, frames):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        
        if not self.multimodal:
            return self.MLP(x)
        else:
            object_features=self.ObjFTExtractor(frames,max_objects=self.max_detected_objs)
            fused_features=self.fusion_strategy(x,object_features)
            return self.MLP(fused_features)