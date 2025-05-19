import torch
import os, sys
from torch.utils.data import Dataset
import re
import pickle
import random

class ADLDataset(Dataset):
    # Tasa de frames del video (constante definida)
    FRAME_RATE = 29.97

    def __init__(self, config, split=None):
        """
        Inicializa el dataset leyendo las rutas de los datos, configuraciones del video,
        y el split (train, val, test) si se especifica.
        """
        # Rutas y parámetros principales a partir de la configuración
        self.anno_path = config.data.annotations_path
        self.data_path = config.data.data_path
        self.frames_path = os.path.join(self.data_path, 'rgb_frames')
        self.video_encoder = config.video_model.name
        self.down_sampling_rate = config.data.down_sampling_rate
        self.video_block_size = config.video_model.block_size
        self.object_encoder = config.object_detector.object_encoder
        self.obj_detect_str = config.object_detector.object_model

        # Determinar qué videos se utilizarán en función del split indicado
        if split is not None:
            split_attr = getattr(config.data, split, None)
            if isinstance(split_attr, list):
                self.split_videos = split_attr
                self.split = split
            else:
                raise ValueError(f'You must specify the videos for {split}')
        else:
            self.split_videos = None

        # Repetición del bloque anterior (puede ser redundante, revisar duplicidad)
        if split is not None:
            split_attr = getattr(config.data, split, None)
            if isinstance(split_attr, list):
                self.split_videos = split_attr
                self.split = split
            else:
                raise ValueError(f'You must specify the videos for {split}')
        else:
            self.split_videos = None

        # Procesa todas las secuencias y anotaciones
        self.process_sequences()


    def get_action_class_dictionary(self):
        """
        Lee el archivo 'action_list.txt' y crea un diccionario que mapea cada ID de acción
        a su etiqueta correspondiente.
        """
        action_list_file = os.path.join(self.anno_path, 'action_annotation', 'action_list.txt')
        action_classes = {}
        # Patrón para extraer el ID y la etiqueta de la acción
        pattern = r"actions\((\d+)\)\.txt\s*=\s*'(.+?)';"
        
        with open(action_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(pattern, line)
                if match:
                    action_id = int(match.group(1))
                    action_label = match.group(2)
                    action_classes[action_id] = action_label
        return action_classes
    

    def get_action_annotations(self):
        """
        Lee las anotaciones de acción para cada video desde los archivos de anotación.
        Filtra los videos según el split (si está definido) y retorna un diccionario
        donde la clave es el nombre del video y el valor es una lista de anotaciones.
        """
        anno_dir = os.path.join(self.anno_path, 'action_annotation')
        video_annotations = {}
        
        # Se recorren los archivos de anotación ordenadamente
        for anno_file in sorted(os.listdir(anno_dir)):
            # Solo se consideran archivos que empiecen con "P_"
            if not anno_file.startswith("P_"):
                continue
            video_name = anno_file.split('.')[0]
            # Filtra videos si se ha definido un split específico
            if self.split_videos is not None and video_name not in self.split_videos:
                continue

            file_path = os.path.join(anno_dir, anno_file)
            annotations = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    tokens = line.split()
                    if len(tokens) < 3:
                        continue
                    
                    start_time = tokens[0]
                    end_time = tokens[1]
                    
                    try:
                        action_id = int(tokens[2])
                    except ValueError:
                        continue

                    action_name = self.action_classes.get(action_id, "Unknown")
                    if action_name == "Unknown":
                        continue

                    # Caso especial para un video en particular (P_19.txt)
                    if anno_file == "P_19.txt":
                        if end_time == "20:18" and start_time == "01:58":
                            if action_name == "drinking coffee/tea":
                                continue
                    # Agrega la anotación a la lista
                    annotations.append({
                        'start': start_time,
                        'end': end_time,
                        'action': action_name
                    })

            video_annotations[video_name] = annotations
            
        return video_annotations

    
    def convert_to_seconds(self, time_str):
        """
        Convierte una cadena de tiempo en formato mm:ss a segundos.
        """
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return 0.0

    
    def split_segment(self, frames, start_frame, end_frame, action, video_name):
        """
        Divide un segmento de video en bloques (subsecuencias) de tamaño fijo.
        Si la cantidad de frames es mayor al tamaño del bloque y el split es 'train',
        se realiza un muestreo aleatorio para eliminar frames sobrantes.
        Retorna una lista de secuencias, cada una con sus correspondientes datos.
        """
        sequences = []
        posicion_subsegmento = 0
        # Si no hay suficientes frames para un bloque, se descarta la secuencia
        if len(frames) < self.video_block_size:
            return []
        # Si hay más frames de los necesarios para un bloque
        if len(frames) > self.video_block_size:
            if self.split == 'train':
                remainder = len(frames) % self.video_block_size
                if remainder != 0:
                    # Calcula el número de bloques completos
                    blocks = len(frames) // self.video_block_size
                    # Selecciona aleatoriamente un bloque para eliminar los frames sobrantes
                    start_block = random.randint(0, blocks)
                    end_index = self.video_block_size * start_block
                    start_index = end_index + remainder 
                    # Combina frames antes y después de los sobrantes
                    frames = frames[:end_index] + frames[start_index:]
            else:
                # En validación o test, se selecciona un bloque aleatorio
                start_idx = random.randint(0, len(frames) - self.video_block_size)
                end_idx = start_idx + self.video_block_size
                frames = frames[start_idx:end_idx]
        # Para el split 'train' se generan múltiples bloques si es posible
        if self.split == 'train':
            for i in range(0, len(frames), self.video_block_size):
                block_frames = frames[i:i+self.video_block_size]
                # Se obtiene el primer y último frame (para referencia, aunque no se usan aquí)
                primer_frame = os.path.splitext(os.path.basename(block_frames[0]))[0]
                ultimo_frame = os.path.splitext(os.path.basename(block_frames[-1]))[0]
                sequences.append((video_name, start_frame, end_frame, action, block_frames, posicion_subsegmento))
                posicion_subsegmento += 1
        else:
            # Para validación o test se retorna un único bloque
            primer_frame = os.path.splitext(os.path.basename(frames[0]))[0]
            ultimo_frame = os.path.splitext(os.path.basename(frames[-1]))[0]
            sequences.append((video_name, start_frame, end_frame, action, frames, posicion_subsegmento))
        return sequences

            
    def get_sequeces(self):
        """
        Genera y retorna una lista de secuencias de video a partir de las anotaciones.
        Para cada anotación de acción, se calculan los frames correspondientes y se
        subdividen en bloques utilizando 'split_segment'.
        """
        sequences = []
        for video_name, annotations in self.action_annotations.items():
            for ann in annotations:
                start_str = ann['start']
                end_str = ann['end']
                start_sec = self.convert_to_seconds(start_str)
                end_sec = self.convert_to_seconds(end_str)
                # Si la duración es menor o igual a cero se omite
                if end_sec - start_sec <= 0:
                    continue
                # Calcular los índices de los frames a partir del tiempo y la tasa de frames
                start_frame = int(start_sec * ADLDataset.FRAME_RATE)
                end_frame = int(end_sec * ADLDataset.FRAME_RATE)
                frames = []
                frames_path = os.path.join(self.frames_path, video_name)
                # Selecciona frames aplicando down-sampling (se empieza en start+3 y luego se salta según down_sampling_rate)
                for frame in range(start_frame + 3, end_frame, self.down_sampling_rate):
                    frame_str = f"{frame:06}.jpg"  # Los nombres de los frames tienen 6 dígitos
                    frame_path = os.path.join(frames_path, frame_str)
                    frames.append(frame_path)
                # Se obtiene la secuencia dividida en bloques
                sequence = self.split_segment(frames, start_frame, end_frame, ann['action'], video_name)
                sequences.extend(sequence)
        return sequences

    

    def split_overlapping_annotations(self):
        """
        Procesa las anotaciones de cada video para dividir los intervalos en segmentos disjuntos
        donde se solapan las acciones. Solo se incluyen aquellos segmentos en los que dos o más
        anotaciones se superponen.
        
        Devuelve un diccionario en el que cada clave es un video y su valor es una lista de intervalos,
        cada uno con 'start', 'end' y 'action' (lista de acciones solapadas en ese intervalo).
        """
        new_annotations = {}

        for video, ann_list in self.action_annotations.items():
            # Convertir cada anotación a un intervalo en segundos
            intervals = []
            for ann in ann_list:
                start_sec = self.convert_to_seconds(ann['start'])
                end_sec = self.convert_to_seconds(ann['end'])
                intervals.append({
                    'start': start_sec,
                    'end': end_sec,
                    'action': ann['action']
                })
            
            # Recopilar todos los límites (inicios y finales) de los intervalos
            boundaries = set()
            for interval in intervals:
                boundaries.add(interval['start'])
                boundaries.add(interval['end'])
            sorted_boundaries = sorted(boundaries)

            # Crear segmentos basados en los límites y asignar las acciones activas en cada segmento
            segments = []
            for i in range(len(sorted_boundaries) - 1):
                seg_start = sorted_boundaries[i]
                seg_end = sorted_boundaries[i + 1]
                active_actions = []
                for interval in intervals:
                    # Una acción es activa en el segmento si su intervalo la cubre completamente
                    if interval['start'] <= seg_start and interval['end'] >= seg_end:
                        active_actions.append(interval['action'])
                # Solo se consideran segmentos con solapamiento (más de una acción)
                if len(active_actions) > 1:
                    # Eliminar duplicados conservando el orden
                    deduped_actions = sorted(list(dict.fromkeys(active_actions)))
                    seg = {
                        'start': self.seconds_to_time_str(seg_start),
                        'end': self.seconds_to_time_str(seg_end),
                        'action': deduped_actions
                    }
                    segments.append(seg)

            # Fusionar segmentos consecutivos con la misma lista de acciones
            merged_segments = []
            for seg in segments:
                if merged_segments and merged_segments[-1]['action'] == seg['action']:
                    merged_segments[-1]['end'] = seg['end']
                else:
                    merged_segments.append(seg)

            new_annotations[video] = merged_segments

        return new_annotations

    
    def seconds_to_time_str(self, seconds):
        """
        Convierte segundos a un string con formato mm:ss.
        """
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02}:{s:02}"


    def distribute_annotations_with_overlapping(self):
        """
        Combina las anotaciones originales con los segmentos de solapamiento para obtener la distribución final
        de anotaciones en cada video.
        
        Para cada video:
          1. Se recopilan los límites de ambas fuentes de anotación.
          2. Se generan intervalos a partir de estos límites.
          3. Se determina la(s) acción(es) activas en cada intervalo.
          4. Se fusionan intervalos consecutivos con las mismas acciones.
        
        Devuelve un diccionario en el que cada clave es el nombre del video y su valor es una lista de segmentos,
        cada uno con 'start', 'end' y 'action'.
        """
        final_annotations = {}

        for video in self.action_annotations:
            # Recuperar las anotaciones originales y los segmentos de solapamiento para el video
            orig_segments = self.action_annotations[video]
            overlap_segments = self.overlapping_annotations.get(video, [])

            # Recopilar todos los límites de tiempo de ambas fuentes
            boundaries = set()
            for seg in orig_segments:
                boundaries.add(self.convert_to_seconds(seg['start']))
                boundaries.add(self.convert_to_seconds(seg['end']))
            for seg in overlap_segments:
                boundaries.add(self.convert_to_seconds(seg['start']))
                boundaries.add(self.convert_to_seconds(seg['end']))
            sorted_boundaries = sorted(boundaries)

            final_segments = []
            # Dividir la línea de tiempo en intervalos basados en los límites
            for i in range(len(sorted_boundaries) - 1):
                t_start = sorted_boundaries[i]
                t_end = sorted_boundaries[i + 1]
                
                # Verificar primero si el intervalo está completamente contenido en algún solapamiento
                overlap_annotation = None
                for o_seg in overlap_segments:
                    o_start = self.convert_to_seconds(o_seg['start'])
                    o_end = self.convert_to_seconds(o_seg['end'])
                    if t_start >= o_start and t_end <= o_end:
                        overlap_annotation = o_seg['action']
                        break

                if overlap_annotation is not None:
                    annotation = overlap_annotation
                else:
                    # Sino, determinar qué anotación(s) originales cubren el intervalo
                    active_actions = []
                    for seg in orig_segments:
                        seg_start = self.convert_to_seconds(seg['start'])
                        seg_end = self.convert_to_seconds(seg['end'])
                        if t_start >= seg_start and t_end <= seg_end:
                            active_actions.append(seg['action'])
                    # Si solo hay una acción se usa directamente, si hay más se eliminan duplicados
                    if len(active_actions) == 1:
                        annotation = [active_actions[0]]
                    elif len(active_actions) > 1:
                        annotation = list(dict.fromkeys(active_actions))
                    else:
                        continue

                segment = {
                    'start': self.seconds_to_time_str(t_start),
                    'end': self.seconds_to_time_str(t_end),
                    'action': annotation
                }
                final_segments.append(segment)

            # Fusionar segmentos consecutivos con la misma anotación
            merged_segments = []
            for seg in final_segments:
                if merged_segments and merged_segments[-1]['action'] == seg['action']:
                    merged_segments[-1]['end'] = seg['end']
                else:
                    merged_segments.append(seg)
            
            final_annotations[video] = merged_segments

        return final_annotations

    
    def process_sequences(self):
        """
        Procesa todas las anotaciones y secuencias:
          1. Obtiene el diccionario de clases de acción.
          2. Crea un mapeo de acción a índice.
          3. Lee las anotaciones originales y las solapadas.
          4. Combina las anotaciones para obtener la distribución final.
          5. Genera las secuencias de video a partir de los frames.
        """
        self.action_classes = self.get_action_class_dictionary()
        self.action_to_idx = {self.action_classes[k]: idx for idx, k in enumerate(sorted(self.action_classes.keys()))}
        self.action_annotations = self.get_action_annotations()
        self.overlapping_annotations = self.split_overlapping_annotations()
        self.action_annotations = self.distribute_annotations_with_overlapping()
        self.sequences = self.get_sequeces()

    
    def __len__(self):
        """
        Retorna la cantidad de secuencias disponibles en el dataset.
        """
        return len(self.sequences)

    
    def __getitem__(self, index):
        """
        Retorna la secuencia (video_features y etiquetas) correspondiente al índice dado.
        
        - Carga las características de video (si no se han cargado previamente) desde un archivo pickle.
        - Busca la clave de características en el diccionario.
        - Crea el tensor multi-hot para las etiquetas.
        """
        file_name = f"/features/video/{self.video_encoder}_{self.split}.pkl"
        # Verifica si el atributo 'video_features' no existe o es None; en ese caso, se carga desde el archivo
        if not hasattr(self, "video_features") or self.video_features is None:
            with open(file_name, "rb") as f:
                self.video_features = pickle.load(f)
        

        video_name, start_frame, end_frame, actions, frames, block_idx = self.sequences[index]

        first_frame = os.path.splitext(os.path.basename(frames[0]))[0]
        last_frame = os.path.splitext(os.path.basename(frames[-1]))[0]
        feature_key = f"NA{len(actions)}_S{first_frame}_E{last_frame}_V{video_name}"

        video_features = self.video_features.get(feature_key)
        if video_features is None:
            raise KeyError(f"Feature key '{feature_key}' not found in the video_features dictionary")
        
        # Convierte las acciones a índices y crea un tensor multi-hot
        labels_ids = [self.action_to_idx[action] for action in actions]
        labels = torch.nn.functional.one_hot(torch.tensor(labels_ids), num_classes=len(self.action_classes)).float()
        multi_hot_tensor = labels.sum(dim=0)
        multi_hot_tensor = (multi_hot_tensor > 0).float()

        return video_features, multi_hot_tensor,frames