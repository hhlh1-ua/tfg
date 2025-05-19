import torch
import  os
from transformers import TimesformerModel,AutoImageProcessor,VivitModel, VideoMAEModel,VivitImageProcessor
from PIL import Image
import torch.nn as nn
import wandb
import pickle
import random  
class VideoFeaturesExtractor():
    def __init__(self, config):

        
        self.block_size=config.video_model.block_size


        if config.video_model.name == "vivit":
            self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
            self.cls_token = lambda x: x['last_hidden_state'][:, 0, :]
            self.model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")  
            if self.block_size % 32 != 0:
                raise ValueError("Invalid block size. Number must be a multiple of 32") 


        elif config.video_model.name == "videomae":
            # self.image_processor = AutoImageProcessor.from_pretrained('MCG-NJU/videomae-base')
            # self.model           = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            self.image_processor = AutoImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
            self.model           = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            

            self.cls_token =lambda x: torch.mean(x['last_hidden_state'],dim=1)

            if self.block_size % 16 != 0:
                raise ValueError("Invalid block size. Number must be a multiple of 16") 


        elif config.video_model.name == "timesformer":
            self.image_processor = AutoImageProcessor.from_pretrained('MCG-NJU/videomae-base')
            self.cls_token = lambda x: x['last_hidden_state'][:, 0, :]
            self.model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
            if self.block_size % 8 != 0:
                raise ValueError("Invalid block size. Number must be a multiple of 8") 
        else:
            raise ValueError("Invalid video transformer model") 
        
        self.pktlFilename=config.video_model.name
        
        self.down_sampling_rate = config.data.down_sampling_rate
        

    def get_features(self,dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("===== WARNING =====")
            print("Running on CPU")
            print("==================")

        fts_dict = {}
        base_frames_path = os.path.join(dataset.data_path, 'rgb_frames')
        print(f"Extracting video features for the {dataset.split} split ...")
        for s in dataset.sequences:
            video, start, end, actions, block_frames, posicion_subsegmento = s #(<video>, <start_frame>, <end_frame>, <action>, [frames after downsampling])(e.g., ('P_19', 13830, 14190, 'adjusting thermostat'))
            print(f"video: {video}, start frame:{start}, end frame:{end}, actions: {actions}, selected frames:{len(block_frames)}, block number: {posicion_subsegmento}")
            if len(block_frames)< self.block_size:
                continue
            first_frame = os.path.splitext(os.path.basename(block_frames[0]))[0]
            last_frame = os.path.splitext(os.path.basename(block_frames[-1]))[0]


            frames = [Image.open(frame_path).convert('RGB') for frame_path in block_frames]
            inputs = self.image_processor(images=frames, return_tensors="pt")
            inputs.to(device)
            self.model.to(device)
            output = self.model(**inputs)
            video_representation = self.cls_token(output).detach().cpu().numpy().squeeze()
            fts_dict[f"NA{len(actions)}_S{first_frame}_E{last_frame}_V{video}"] = video_representation

        file_name = f"/features/video/{self.pktlFilename}_{dataset.split}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(fts_dict, f)
        print("Extraction completed !")
        
