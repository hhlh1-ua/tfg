import torch
import  os
from torch.utils.data import Dataset
import re
import pickle
import random

class ADLDataset(Dataset):
    FRAME_RATE=29.97
    def __init__(self, config, split=None):
        self.anno_path = config.data.annotations_path
        self.data_path=config.data.data_path
        self.frames_path=os.path.join(self.data_path,'rgb_frames')
        video_encoder=config.video_model.name
        self.down_sampling_rate=config.data.down_sampling_rate
        self.video_block_size=config.video_model.block_size
        if split is not None:
            split_attr = getattr(config.data, split, None)
            if isinstance(split_attr, list):
                self.split_videos = split_attr
                self.split=split
            else:
                raise ValueError(f'You must specify the videos for {split}') 
        else:
            self.split_videos = None

        
        if split is not None:
            split_attr = getattr(config.data, split, None)
            if isinstance(split_attr, list):
                self.split_videos = split_attr
                self.split=split
                if config.video_model.extract_features == False:
                    file_name = f"/features/video/{video_encoder}_{split}.pkl"
                    with open(file_name, "rb") as f:
                        self.features = pickle.load(f)
            else:
                raise ValueError(f'You must specify the videos for {split}') 
        else:
            self.split_videos = None

        self.process_sequences()
            


    
    def get_action_class_dictionary(self):
        action_list_file = os.path.join(self.anno_path, 'action_annotation', 'action_list.txt')
        action_classes = {}
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


        anno_dir = os.path.join(self.anno_path, 'action_annotation')
        
        video_annotations = {}
        
        for anno_file in sorted(os.listdir(anno_dir)):
            if not anno_file.startswith("P_"):
                continue
            video_name = anno_file.split('.')[0]
            # Si se ha definido un split, filtramos por la lista self.split_videos
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

                    annotations.append({
                        'start': start_time,
                        'end': end_time,
                        'action': action_name
                    })

            print(f"{video_name}: {annotations}")
            video_annotations[video_name] = annotations
            
        return video_annotations

    def convert_to_seconds(self,time_str):
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return 0.0
    
    def split_segment(self,frames,start_frame,end_frame,action,video_name):
        sequences = []
        posicion_subsegmento=0
        if len(frames) < self.video_block_size:
            return []
        if len(frames) > self.video_block_size:
            if self.split == 'train':
                remainder = len(frames) % self.video_block_size
                if remainder != 0:
                    blocks=len(frames)// self.video_block_size ## Vemos la cantidad de bloques X bytes que se pueden obtener
                    start_block= random.randint(0, blocks) ## Devuelve entre que bloques de X frames hay que eliminar los frames sobrantes
                    end_index = self.video_block_size * start_block ## Se calculan los frames que se van a coger al principio 
                    start_index = end_index + remainder 
                    ##Ej si hay 32 frames, se coge del 7 al 31 si start_block = 0. si start_block = 1 se coge del 0 al 15 y del 22 al 37
                    frames = frames[:end_index] + frames [start_index:]
            else:
                start_idx= random.randint(0, len(frames)-self.video_block_size)
                end_idx=start_idx+self.video_block_size
                frames = frames[start_idx:end_idx]
        if self.split=='train':
                for i in range(0, len(frames), self.video_block_size):
                    block_frames = frames[i:i+self.video_block_size]
                    primer_frame = os.path.splitext(os.path.basename(block_frames[0]))[0]
                    ultimo_frame = os.path.splitext(os.path.basename(block_frames[-1]))[0]
                    sequences.append((video_name,start_frame,end_frame,action,block_frames,posicion_subsegmento))
                    posicion_subsegmento+=1
        else:
            primer_frame = os.path.splitext(os.path.basename(frames[0]))[0]
            ultimo_frame = os.path.splitext(os.path.basename(frames[-1]))[0]
            sequences.append((video_name,start_frame,end_frame,action,frames,posicion_subsegmento))
        return sequences

            

    def get_sequeces(self):
        sequences =[]
        for video_name,annotations  in self.action_annotations.items():
            print(video_name)
            for ann in annotations:
                start_str = ann['start']
                end_str = ann['end']
                start_sec = self.convert_to_seconds(start_str)
                end_sec = self.convert_to_seconds(end_str)
                if end_sec - start_sec <= 0:
                    continue # Skip too short sequences
                start_frame= int(start_sec*ADLDataset.FRAME_RATE)
                end_frame= int(end_sec*ADLDataset.FRAME_RATE)
                frames = []
                frames_path=os.path.join(self.frames_path,video_name)
                # Selecciona el tercer frame de cada grupo de 5:
                # Comienza en start + 3 y luego añade 5 en cada iteración.
                for frame in range(start_frame+3, end_frame, self.down_sampling_rate):
                    frame_str = f"{frame:06}.jpg"  # Los frames se guardan con 6 dígitos
                    frame_path = os.path.join(frames_path, frame_str)
                    frames.append(frame_path)

                sequence = self.split_segment(frames,start_frame,end_frame,ann['action'],video_name)                      
                sequences.extend(sequence)


        return sequences

    def get_action_classes(self):
        return self.action_classes
        
    def process_sequences(self):
        self.action_classes=self.get_action_class_dictionary()
        self.action_to_idx = {self.action_classes[k]: idx for idx, k in enumerate(sorted(self.action_classes.keys()))}
        self.action_annotations=self.get_action_annotations()
        self.sequences=self.get_sequeces()

    
    def __len__ (self):

        return len((self.sequences))

    def __getitem__(self, index):

        video_name,start_frame,end_frame,action, _ =self.sequences[index]

        feature_key=f"A{action}_S{start_frame}_E{end_frame}_V{video_name}"
        features = self.features.get(feature_key)

        if features is None:
            raise KeyError(f"Feature key '{feature_key}' not found in the features dictionary")
        if action not in self.action_to_idx:
            raise KeyError(f"Action '{action}' not found in action_classes {feature_key}")

        label_id = self.action_to_idx[action]
        label = torch.tensor(label_id, dtype=torch.long)
        # label = torch.nn.functional.one_hot(torch.tensor(label_id), num_classes=len(self.action_classes)).float()
        

        return features,label


        
        
                    
