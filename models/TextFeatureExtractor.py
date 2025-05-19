import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel,DistilBertTokenizer, DistilBertModel,RobertaTokenizer, RobertaModel
from models.ObjectDetector import ObjectDetectorCreator
import os
import pickle
class TextFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = config.text_encoder.text_model
        
        if self.text_encoder == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
            self.textModel = BertModel.from_pretrained('bert-base-uncased')
            
        elif self.text_encoder == 'DistilBert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.textModel = DistilBertModel.from_pretrained('distilbert-base-uncased')
            
        elif self.text_encoder == 'Roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.textModel = RobertaModel.from_pretrained('roberta-base')
        
        else:
            raise ValueError(f"Modelo de texto no soportado: {self.text_encoder}")
        self.obj_detector = ObjectDetectorCreator.instatiate_ObjectDetector(config)
        self.obj_detect_str = config.object_detector.object_model

    def forward(self, datasets):
        print("Extracting additional object features...")
        """
        Args:
            bboxes (List[tuple]): Lista de bounding boxes en formato (x1, y1, x2, y2, clase).
        
        Returns:
            Tensor: Representación del embedding del token [CLS] o equivalente para cada bbox.
        """
        self.textModel.to(self.device)
        self.obj_detector.to(self.device)
        self.textModel.eval()
        self.obj_detector.eval()
        fts_dict = {}
        
        # Desactivar cálculo de gradientes para la inferencia y ahorrar memoria
        with torch.no_grad():
            for dataset in datasets:
                print(f"Extracting additional object features for the {dataset.split} split ...")
                for s in dataset.sequences:
                    video, start, end, actions, block_frames, posicion_subsegmento = s
                    print(f"video: {video}, start frame:{start}, end frame:{end}, actions: {actions}, selected frames:{len(block_frames)}, block number: {posicion_subsegmento}")
                    for i,frame_path in enumerate(block_frames):
                        if i%2 != 0:
                            continue
                        bboxes = self.obj_detector(frame_path)
                        if len(bboxes) == 0:
                            continue
                        # Se construye un prompt concatenando las coordenadas y la clase
                        for bbox in bboxes:
                            prompt_text = [f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}"]
                            encoded_input = self.tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True)
                            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                            outputs_text = self.textModel(**encoded_input)
                            # Para Bert y RoBERTa, usualmente se utiliza el token [CLS] (o <s> en RoBERTa)
                            cls_embedding = outputs_text.last_hidden_state[:, 0, :]  # [len_bboxes, hidden_size]
                            frame = os.path.splitext(os.path.basename(frame_path))[0]
                            additional_objs_representation= cls_embedding.detach().cpu().numpy().squeeze()
                            fts_dict[f"X1{bbox[0]}_Y1{bbox[1]}_X2{bbox[2]}_Y2{bbox[3]}_CLS{bbox[-1]}_F{frame}_V{video}"] = additional_objs_representation


        file_name = f"/features/objects/text/{self.obj_detect_str}/{self.text_encoder}.pkl"
        dir_path = os.path.dirname(file_name)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(fts_dict, f)
        print("Extraction completed !")
