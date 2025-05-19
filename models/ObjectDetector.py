import os
import torch.nn as nn
import torch
import json


class ObjectDetectorCreator():
    @staticmethod
    def instatiate_ObjectDetector(config):
        if config.object_detector.object_model == 'GT':
            return GTdetector(config)
        elif config.object_detector.object_model == 'GDino':
            return GDinoDetector(config)
        else:
            raise ValueError(f"Modelo de detección de objetos no soportado: {config.object_detector.object_model}")



class GTdetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.annotations_dir = os.path.join(config.data.annotations_path, 'object_annotation')
        # Cache para evitar leer varias veces el mismo archivo
        self.cache = {}

    def load_annotations(self, video_id):
        """
        Carga las anotaciones de 'object_annot_<video_id>.txt'
        y almacena en un diccionario:
            frame_to_bboxes[frame_id] = [ (x1, y1, x2, y2, clase), ... ]
        """
        if video_id in self.cache:
            return self.cache[video_id]
        
        annot_path = os.path.join(self.annotations_dir, f"object_annot_{video_id}.txt")
        frame_to_bboxes = {}
        with open(annot_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                
                x1, y1, x2, y2 = map(int, parts[1:5])
                frame_id = parts[5]
                clase = parts[-1]
                
                # Ajustar el frame_id removiendo los dos primeros dígitos
                frame_id = frame_id[2:].zfill(6)
                
                frame_to_bboxes.setdefault(frame_id, []).append((x1, y1, x2, y2, clase))
        
        self.cache[video_id] = frame_to_bboxes
        return frame_to_bboxes

    def forward(self, frame_path):
        """
        Dado el path de una imagen, extrae el video_id y el número de frame
        y retorna una lista con los objetos detectados. Si no hay anotaciones
        para el frame, usa las del frame anterior más cercano.
        """
        parts = frame_path.split(os.sep)
        video_id = parts[-2]        # Ejemplo: "P_01"
        frame_file = parts[-1]      # Ejemplo: "000000.jpg"
        frame_num = os.path.splitext(frame_file)[0]  # "000000"
        
        # Cargar las anotaciones para este video
        annotations = self.load_annotations(video_id)
        
        # Intentar obtener directamente las bboxes
        bboxes = annotations.get(frame_num, [])
        if not bboxes:
            # Si no existen, buscar el frame anterior más cercano
            # Convertimos las claves a enteros y filtramos < frame_num
            current = int(frame_num)
            prev_frames = [int(f) for f in annotations.keys() if int(f) < current]
            # print(prev_frames)
            if prev_frames:
                closest = str(max(prev_frames)).zfill(len(frame_num))
                # print(closest)
                bboxes = annotations[closest]
        # print(bboxes)
        return bboxes



class GDinoDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        bboxes_path = f"/features/objects/GDino/bboxes/results-block{config.video_model.block_size}.pkl"
        annotations_path = f"/workspace/tfg_hhernandez/models/GroundingDINO/Open-GroundingDino/ADL/Key_frames_Annotated_Block{config.video_model.block_size}.json"
        label_map_path = f"/workspace/tfg_hhernandez/models/GroundingDINO/Open-GroundingDino/config/label_map_ADL.json"

        with open(label_map_path, 'r') as file:
            self.label_map = json.load(file)

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        self.image_dict = {img["file_name"]: img["id"] for img in annotations.get("images", [])}

        if not hasattr(self, "bboxes") or self.objects_features is None:
           self.bboxes=torch.load(bboxes_path)

    def forward(self, frame_path):
        """
        Dado el path de una imagen, extrae el video_id y el número de frame
        y retorna una lista con los objetos detectados.
        """
        parts = frame_path.split(os.sep)
        video_id = parts[-2]
        frame_file = parts[-1]
        file_name = os.path.join(video_id, frame_file)
        idx = self.image_dict.get(file_name)
        # print(idx)
        if idx is None:
            raise ValueError(f"Imagen {file_name} no encontrada en el diccionario de imágenes.")
        bboxes = []
        for bbox_data in self.bboxes['res_info'][idx]:
            x1, y1, x2, y2, confidence, label = bbox_data
            if confidence < 0.35:
                continue
            
            # Convertir el índice de la etiqueta a su nombre
            label = self.label_map[str(int(label))]
            # print(f"bbox_data: {bbox_data} {label}")
            bboxes.append((int(x1), int(y1), int(x2), int(y2),float(confidence), label))
        
        return bboxes