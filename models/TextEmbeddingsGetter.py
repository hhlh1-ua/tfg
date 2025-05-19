import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel,DistilBertTokenizer, DistilBertModel,RobertaTokenizer, RobertaModel
from models.ObjectDetector import ObjectDetectorCreator
import os
import pickle
class TextEmbeddingsGetter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = config.text_encoder.text_model

        if self.text_encoder != 'Bert' and self.text_encoder != 'DistilBert' and self.text_encoder != 'Roberta':
            raise ValueError(f"Modelo de texto no soportado: {self.text_encoder}")


        
        self.obj_detector = ObjectDetectorCreator.instatiate_ObjectDetector(config)
        self.obj_detect_str = config.object_detector.object_model
        additional_object_features_file_name = f"/features/objects/text/{self.obj_detect_str}/{self.text_encoder}.pkl"
        if not hasattr(self, "additional_objects_features") or self.additional_objects_features is None:
            with open(additional_object_features_file_name, "rb") as f:
                self.additional_objects_features = pickle.load(f)

    def forward(self, key):
        """
        Args:
            key (str): Clave del diccionario de características que representa el objeto.
        
        Returns:
            Tensor: Representación del embedding del token [CLS] o equivalente para cada bbox.
        """
        cls_embedding = self.additional_objects_features.get(key)
        if cls_embedding is None:
            raise ValueError(f"Key {key} not found in additional objects features")
        
        return cls_embedding
