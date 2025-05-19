import torch.nn.functional as F
from transformers import ViTImageProcessorFast, ViTModel, SwinModel
from torch import nn
import torch
from models.Cropper import Cropper
from models.ObjectDetector import ObjectDetectorCreator
from models.TextFeatureExtractor import TextFeatureExtractor
import os
import pickle


class ObjectFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.object_encoder = config.object_detector.object_encoder

        if self.object_encoder == "swin":
            self.visual_encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
            self.image_processor = ViTImageProcessorFast.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.cls_token = lambda x: x['pooler_output']
        elif self.object_encoder == "vit":
            self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.image_processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224-in21k")
            self.cls_token = lambda x: x['last_hidden_state'][:, 0, :]
        else:
            raise ValueError(f"Modelo de objeto no soportado: {self.object_encoder}")

        self.obj_detector = ObjectDetectorCreator.instatiate_ObjectDetector(config)
        self.obj_detect_str = config.object_detector.object_model
        self.cropper = Cropper()

    def forward(self, datasets):
        print("Extracting object features...")
        self.visual_encoder.to(self.device)
        self.obj_detector.to(self.device)
        self.obj_detector.eval()
        self.visual_encoder.eval()
        fts_dict = {}

        # Desactivar cálculo de gradientes para la inferencia y ahorrar memoria
        with torch.no_grad():
            for dataset in datasets:
                for s in dataset.sequences:
                    video, start, end, actions, block_frames, posicion_subsegmento = s
                    print(f"video: {video}, start frame:{start}, end frame:{end}, actions: {actions}, selected frames:{len(block_frames)}, block number: {posicion_subsegmento}")
                    for i,frame_path in enumerate(block_frames):
                        if i%2 != 0:
                            continue
                        bboxes = self.obj_detector(frame_path)
                        if len(bboxes) > 0:
                            for bbox in bboxes:
                                # Crop the image using the bounding boxes
                                crops = self.cropper.get_crops(frame_path, [bbox])
                                inputs = self.image_processor(images=crops, return_tensors="pt")
                                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                                # Get the features from the visual encoder
                                outputs = self.visual_encoder(**inputs)
                                # Use the CLS token or equivalent as the feature representation
                                cls_embedding = self.cls_token(outputs)
                                # Mover el tensor a CPU para liberar memoria en la GPU
                                frame = os.path.splitext(os.path.basename(frame_path))[0]
                                obj_representation= cls_embedding.detach().cpu().numpy().squeeze()
                                # print(f"obj_representation: {obj_representation.shape}")
                                fts_dict[f"X1{bbox[0]}_Y1{bbox[1]}_X2{bbox[2]}_Y2{bbox[3]}_CLS{bbox[-1]}_F{frame}_V{video}"] = obj_representation


        file_name = f"/features/objects/{self.obj_detect_str}/object_features/{self.object_encoder}.pkl"
        dir_path = os.path.dirname(file_name)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(fts_dict, f)
        print("Extraction completed !")
