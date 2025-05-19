import torch.nn.functional as F
from transformers import ViTImageProcessorFast, ViTModel, SwinModel
from torch import nn
import torch

from models.ObjectDetector import ObjectDetectorCreator
from models.TextEmbeddingsGetter import TextEmbeddingsGetter
import pickle
import os



class ObjectEmbeddingsGetter(nn.Module):
    def __init__(self, config,object_detector=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.object_encoder = config.object_detector.object_encoder

        self.get_text_features = config.model.get_text_features
        self.get_object_features = config.model.get_object_features

        if self.object_encoder != "swin" and self.object_encoder != "vit":
            raise ValueError(f"Modelo de objeto no soportado: {self.object_encoder}")



        self.hidden_size = 768

        if self.get_object_features:
            self.obj_embedding_size = config.object_detector.embedding_size
            self.linear_object = nn.Sequential(nn.Linear(self.obj_embedding_size, 768), nn.LayerNorm(768))


        if self.get_text_features:
            self.text_feature_extractor = TextEmbeddingsGetter(config)
            self.text_embedding_size = config.text_encoder.embedding_size
            self.linear_text = nn.Sequential(nn.Linear(self.text_embedding_size , 768), nn.LayerNorm(768))
            
        if self.get_object_features and self.get_text_features:
            self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=2), num_layers=2)

        if not self.get_object_features and not self.get_text_features:
            raise ValueError("No se han solicitado características de objeto ni de texto.")
            

        self.obj_detector = object_detector
        self.object_recopilation_strategy = config.object_detector.object_recopilation_strategy
        self.obj_detect_str = config.object_detector.object_model
        objects_file_name = f"/features/objects/{self.obj_detect_str}/object_features/{self.object_encoder }.pkl"
        if not hasattr(self, "objects_features") or self.objects_features is None:
            with open(objects_file_name, "rb") as f:
                self.objects_features = pickle.load(f)

        
        
    def forward(self, frames, max_objects=4):
        
        """
        frames: secuencia (lista) de 16 tuplas,
                cada tupla es (path_img_batch0, path_img_batch1, ..., path_img_batchN).
        max_objects: número máximo de crops a considerar por muestra.
        """
        if (self.object_recopilation_strategy == 'all_frames'):
            object_features = self.objects_in_all_frames(frames_list=frames, max_objects=max_objects)
 
        if (self.object_recopilation_strategy == '1stframe'):
            object_features = self.objects_in_one_frame(frames_list=frames, frame_num=0,max_objects=max_objects)
            
        if (self.object_recopilation_strategy == 'middleframe'):
            middle_frame = len(frames) // 2
            object_features = self.objects_in_one_frame(frames_list=frames, frame_num=middle_frame,max_objects=max_objects)

        # if (self.object_recopilation_strategy == 'lastframe'):
        #     object_features = self.objects_in_one_frame(frames_list=frames, frame_num=-1,max_objects=max_objects)

        # print (object_features.shape)
        return object_features

    
    def objects_in_all_frames(self, frames_list=None, max_objects=4):

        batch_size = len(frames_list[0])
        final_features_per_video = [[] for _ in range(batch_size)]

        for i, frames in enumerate(zip(*frames_list)):
            bboxes_in_video = []
            object_history = {}
            
            # Recorremos todos los frames para obtener las bboxes y actualizar el contador por clase
            for j, frame in enumerate(frames):
                if j % 2 != 0:
                    continue
                bboxes = self.obj_detector(frame)
                bboxes_in_video.append(bboxes)
                for bbox in bboxes:
                    # Extraemos la clase (último elemento de la tupla)
                    label = bbox[-1]
                    object_history[label] = object_history.get(label, 0) + 1

            # Obtenemos las top 4 clases más comunes
            top_classes_tuples = sorted(object_history.items(), key=lambda x: x[1], reverse=True)[:max_objects]
            # Extraemos solo los nombres de las clases
            top_class_names = [item[0] for item in top_classes_tuples]

            # Filtramos las bbox de cada frame para quedarnos con solo una por cada clase de top 4
            filtered_bboxes = []
            for bboxes_in_frame in bboxes_in_video:
                seen = set()  # para controlar que solo se incluya una bbox por clase en este frame
                frame_filtered = []
                for bbox in bboxes_in_frame:
                    label = bbox[-1]
                    if label in top_class_names and label not in seen:
                        frame_filtered.append(bbox)
                        seen.add(label)
                filtered_bboxes.append(frame_filtered)
            j=0
            for bboxes_in_frame in filtered_bboxes:
                
                if self.get_object_features:
                    object_features = torch.empty(0, self.hidden_size, device=self.device)
                if self.get_text_features:
                    text_features = torch.empty(0, self.hidden_size, device=self.device)

                if len(bboxes_in_frame) > 0:
                    if self.get_object_features:
                        features_list = []
                    if self.get_text_features:
                        additional_features_list = []

                    for bbox in bboxes_in_frame:
                        frame = os.path.splitext(os.path.basename(frames[j]))[0]
                        key="X1" + str(bbox[0]) + "_Y1" + str(bbox[1]) + "_X2" + str(bbox[2]) + "_Y2" + str(bbox[3]) + "_CLS" + str(bbox[-1]) + "_F" + str(frame) + "_V" + str(frames[0].split("/")[-2])
                        
                        if self.get_object_features:
                        
                            object_feat=self.objects_features.get(key)#[hidden_size,1]
                            if object_feat is not None:
                                object_feat = torch.tensor(object_feat).squeeze().to(self.device) #[1,hidden_size]
                                features_list.append(object_feat)
                            else:
                                raise ValueError("No se encontró la característica para el bbox:", bbox)


                        if self.get_text_features:
                            text_ft=self.text_feature_extractor(key)
                            if text_ft is not None:
                                text_ft = torch.tensor(text_ft).squeeze().to(self.device)
                                additional_features_list.append(text_ft)
                            else:
                                raise ValueError("No se encontró la característica para el bbox:", bbox)

                    
                    if self.get_object_features:
                        object_features = torch.stack(features_list, dim=0)
                    if self.get_text_features:
                        text_features = torch.stack(additional_features_list, dim=0)
                j+=2

                
                
                
                
                if self.get_object_features and self.get_text_features:
                    
                    if text_features.size(0) == 0 and object_features.size(0) == 0:
                        fused_representation = torch.zeros(max_objects, self.hidden_size, device=self.device)

                    if len(object_features) > 0 and len(text_features) > 0:

                        self.linear_object.to(self.device)
                        self.linear_text.to(self.device)

                        object_features = self.linear_object(object_features)
                        text_features = self.linear_text(text_features)

                        self.transformer_encoder.to(self.device)
                        fusion_input = torch.stack([object_features, text_features], dim=0)  # [2, max_objects, feature_dim]
                        # print("fusion_input shape: ", fusion_input.shape)
                        fusion_output = self.transformer_encoder(fusion_input)
                        fused_representation = fusion_output.mean(dim=0)  # [max_objects, feature_dim]
                        # print("fused_representation shape: ", fused_representation.shape)

                    if len(fused_representation) < max_objects:
                        pad_size = max_objects - len(fused_representation)
                        padding = torch.zeros(pad_size, self.hidden_size, device=self.device)                    
                        fused_representation = torch.cat([fused_representation, padding], dim=0)

                    final_features_per_video[i].append(fused_representation)
                elif self.get_object_features:
                    if object_features.size(0) == 0:
                        object_features = torch.zeros(max_objects, self.hidden_size, device=self.device)
                        final_features_per_video[i].append(object_features)
                    else:
                        self.linear_object.to(self.device)
                        object_features = self.linear_object(object_features)
                        if len(object_features) < max_objects:
                            pad_size = max_objects - len(object_features)
                            padding = torch.zeros(pad_size, self.hidden_size, device=self.device)                    
                            object_features = torch.cat([object_features, padding], dim=0)
                        final_features_per_video[i].append(object_features)
                elif self.get_text_features:
                    if text_features.size(0) == 0:
                        text_features = torch.zeros(max_objects, self.hidden_size, device=self.device)
                        final_features_per_video[i].append(text_features)
                    else:
                        self.linear_text.to(self.device)
                        text_features = self.linear_text(text_features)
                        if len(text_features) < max_objects:
                            pad_size = max_objects - len(text_features)
                            padding = torch.zeros(pad_size, self.hidden_size, device=self.device)                    
                            text_features = torch.cat([text_features, padding], dim=0)
                        final_features_per_video[i].append(text_features)



                
        
        video_tensors = [torch.stack(video_feats, dim=0) for video_feats in final_features_per_video] # (N_frames, max_objects, feature_dim)
        # print(video_tensors[0].shape)
        final_object_features = torch.stack(video_tensors, dim=0) #[batch_size, N_frames, max_objects, feature_dim]
        # print(batch_features.shape)
        # raise ValueError("batch_features shape: ", batch_features.shape, "batch_features: ", batch_features)
        # raise ValueError("final_object_features shape: ", final_object_features.shape)

        return final_object_features


    def objects_in_one_frame(self, frames_list=None, frame_num=0, max_objects=4):
        """
        Procesa únicamente un frame (el indicado por frame_num) de cada video y extrae hasta
        'max_objects' objetos, filtrando para que en el frame solo se incluya una bbox por cada
        una de las clases más comunes (las de mayor frecuencia en ese frame).

        Args:
            frames_list: Lista de tuplas, donde cada tupla contiene los paths de imagen para cada video.
            frame_num : Índice del frame del que extraer los objetos.
            max_objects: Número máximo de objetos a considerar (número de bboxes filtradas).
        
        Returns:
            final_features: Tensor con forma [batch_size, max_objects, feature_dim] que contiene 
                            la representación fusionada (o las features individuales en caso de contar 
                            solo con object o text features) para cada video.
        """
        final_features_per_video = []
        
        # Iteramos sobre cada video; cada video se representa como una tupla de frames.
        for video_frames in zip(*frames_list):
            # Seleccionamos el frame indicado (p. ej.: el primer frame, si frame_num == 0)
            frame = video_frames[frame_num]

            # Detectamos los objetos en el frame
            bboxes = self.obj_detector(frame)

            # Creamos un histograma de las ocurrencias de cada clase en el frame
            object_history = {}
            for bbox in bboxes:
                label = bbox[-1]
                object_history[label] = object_history.get(label, 0) + 1

            # Obtenemos las 'max_objects' clases más frecuentes
            top_classes_tuples = sorted(object_history.items(), key=lambda x: x[1], reverse=True)[:max_objects]
            top_class_names = [item[0] for item in top_classes_tuples]

            # Filtramos las bounding boxes: se incluye solo una bbox por cada clase del top, en orden de aparición
            filtered_bboxes = []
            seen = set()
            for bbox in bboxes:
                label = bbox[-1]
                if label in top_class_names and label not in seen:
                    filtered_bboxes.append(bbox)
                    seen.add(label)

            # Inicializamos tensores vacíos para las features de objeto y de texto
            if self.get_object_features:
                object_features = torch.empty(0, self.hidden_size, device=self.device)
            if self.get_text_features:
                text_features = torch.empty(0, self.hidden_size, device=self.device)

            # Si hay alguna bbox filtrada, extraemos las features correspondientes
            if len(filtered_bboxes) > 0:
                features_list = []
                additional_features_list = []
                for bbox in filtered_bboxes:
                    # Se genera una clave única basada en las coordenadas de la bbox, la clase, el frame y el ID del video
                    frame_name = os.path.splitext(os.path.basename(frame))[0]
                    video_id = video_frames[0].split("/")[-2]
                    key = (
                        f"X1{bbox[0]}_Y1{bbox[1]}_X2{bbox[2]}_Y2{bbox[3]}"
                        f"_CLS{bbox[-1]}_F{frame_name}_V{video_id}"
                    )
                    
                    if self.get_object_features:
                        object_feat = self.objects_features.get(key)
                        if object_feat is not None:
                            object_feat = torch.tensor(object_feat).squeeze().to(self.device)
                            features_list.append(object_feat)
                        else:
                            raise ValueError("No se encontró la característica para el bbox:", bbox)
                            
                    if self.get_text_features:
                        text_ft = self.text_feature_extractor(key)
                        if text_ft is not None:
                            text_ft = torch.tensor(text_ft).squeeze().to(self.device)
                            additional_features_list.append(text_ft)
                        else:
                            raise ValueError("No se encontró la característica para el bbox:", bbox)
                
                if self.get_object_features and features_list:
                    object_features = torch.stack(features_list, dim=0)
                if self.get_text_features and additional_features_list:
                    text_features = torch.stack(additional_features_list, dim=0)
            
            # Fusionamos las features si se tienen ambas (objetos y texto)
            if self.get_object_features and self.get_text_features:
                if object_features.size(0) == 0 and text_features.size(0) == 0:
                    fused_representation = torch.zeros(max_objects, self.hidden_size, device=self.device)
                if object_features.size(0) > 0 and text_features.size(0) > 0:
                    self.linear_object.to(self.device)
                    self.linear_text.to(self.device)
                    object_features = self.linear_object(object_features)
                    text_features = self.linear_text(text_features)

                    self.transformer_encoder.to(self.device)
                    fusion_input = torch.stack([object_features, text_features], dim=0)  # [2, #objetos, feature_dim]
                    fusion_output = self.transformer_encoder(fusion_input)
                    fused_representation = fusion_output.mean(dim=0)  # [#objetos, feature_dim]

                # Si la cantidad de features fusionadas es menor a max_objects, se realiza padding
                if fused_representation.size(0) < max_objects:
                    pad_size = max_objects - fused_representation.size(0)
                    padding = torch.zeros(pad_size, self.hidden_size, device=self.device)
                    fused_representation = torch.cat([fused_representation, padding], dim=0)

                final_features_per_video.append(fused_representation)

            # Caso en que solo se extraigan object features
            elif self.get_object_features:
                if object_features.size(0) == 0:
                    object_features = torch.zeros(max_objects, self.hidden_size, device=self.device)
                else:
                    self.linear_object.to(self.device)
                    object_features = self.linear_object(object_features)
                    if object_features.size(0) < max_objects:
                        pad_size = max_objects - object_features.size(0)
                        padding = torch.zeros(pad_size, self.hidden_size, device=self.device)
                        object_features = torch.cat([object_features, padding], dim=0)
                final_features_per_video.append(object_features)

            # Caso en que solo se extraigan text features
            elif self.get_text_features:
                if text_features.size(0) == 0:
                    text_features = torch.zeros(max_objects, self.hidden_size, device=self.device)
                else:
                    self.linear_text.to(self.device)
                    text_features = self.linear_text(text_features)
                    if text_features.size(0) < max_objects:
                        pad_size = max_objects - text_features.size(0)
                        padding = torch.zeros(pad_size, self.hidden_size, device=self.device)
                        text_features = torch.cat([text_features, padding], dim=0)
                final_features_per_video.append(text_features)
        
        # Se agrupan las features de cada video en un único tensor: [batch_size, max_objects, feature_dim]
        final_features = torch.stack(final_features_per_video, dim=0)
        return final_features



                        


