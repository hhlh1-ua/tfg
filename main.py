import os
import yaml
import shutil
import argparse
from dotmap import DotMap
from dataset.ADL import ADLDataset
from models.VideoFeaturesExtractor import VideoFeaturesExtractor
from models.ObjectFeatureExtractor import ObjectFeatureExtractor
from models.TextFeatureExtractor import TextFeatureExtractor
from models.Classifier import Classifier
from utils.evaluate import test_model
from utils.train import train_model
import utils.control_seed as ctrl_seed
import pickle
import wandb





def get_config():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    args = parser.parse_args()
    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.full_load(f)
    config = DotMap(config)

    return config



def main():
     
    config = get_config()
    ctrl_seed.set_seed(config.seed)
    if config.video_model.extract_features:
        name = f"{config.video_model.name}_{config.video_model.block_size}"
        wandb.init(
            # set the wandb project where this run will be logged
            project="VideoFeatures",
            # track hyperparameters and run metadata
            config=config,
            name=name  
        )
    if config.object_detector.extract_features:
        name = f"{config.object_detector.object_encoder}_{config.object_detector.embedding_size}"
        wandb.init(
            # set the wandb project where this run will be logged
            project="ObjectFeatures",
            # track hyperparameters and run metadata
            config=config,
            name=name  
        )
    elif config.model.multimodal:
        if config.model.get_text_features and config.model.get_object_features:
            name = f"{config.text_encoder.text_model}_{config.object_detector.object_encoder}_{config.video_model.name}_{config.object_detector.object_recopilation_strategy}"
            wandb.init(
                project="objs_and_additional",
                # set the wandb project where this run will be logged
                name=name,
                # track hyperparameters and run metadata
                config=config  
            )
        elif config.model.get_object_features:
            if config.object_detector.object_encoder == "vit":
                name = f"{config.object_detector.object_encoder}_{config.video_model.name}_{config.object_detector.object_recopilation_strategy}"
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="ViTOnly",
                    # track hyperparameters and run metadata
                    config=config,
                    name=name 
                )
            elif config.object_detector.object_encoder == "swin":
                name = f"{config.object_detector.object_encoder}_{config.video_model.name}_{config.object_detector.object_recopilation_strategy}"
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="ActionRecognition",
                    # track hyperparameters and run metadata
                    config=config,
                    name=name  

                )
        elif config.text_encoder.text_model == "Bert":
            name = f"{config.text_encoder.text_model}_{config.video_model.name}_{config.object_detector.object_recopilation_strategy}"
            wandb.init(
                # set the wandb project where this run will be logged
                project="BertOnly",
                # track hyperparameters and run metadata
                config=config,
                name=name  
            )
        elif config.text_encoder.text_model == "DistilBert":
            name = f"{config.text_encoder.text_model}_{config.video_model.name}_{config.object_detector.object_recopilation_strategy}"
            wandb.init(
                # set the wandb project where this run will be logged
                project="DistilBertOnly",
                # track hyperparameters and run metadata
                config=config,
                name=name  
            )
        elif config.text_encoder.text_model == "Roberta":
            name = f"{config.text_encoder.text_model}_{config.video_model.name}_{config.object_detector.object_recopilation_strategy}"
            wandb.init(
                # set the wandb project where this run will be logged
                project="RobertaOnly",
                # track hyperparameters and run metadata
                config=config,
                name=name  
            )
        else:
            print("***** Multimodal *****")
            wandb.init(
                # set the wandb project where this run will be logged
                project="ActionRecognition",
                # track hyperparameters and run metadata
                config=config  
            )
    elif not config.model.multimodal:
        name = f"{config.video_model.name}_{config.video_model.block_size}"
        wandb.init(
            # set the wandb project where this run will be logged
            project="BasicMLP",
            # track hyperparameters and run metadata
            config=config,
            name=name  
        )
    

        

    train_ds  = ADLDataset(config,'train')
    val_ds  = ADLDataset(config,'val')
    test_ds  = ADLDataset(config,'test')
    if config.video_model.extract_features:
        
        video_model = VideoFeaturesExtractor(config)
        video_model.get_features(train_ds)
        video_model.get_features(val_ds)
        video_model.get_features(test_ds)

    if config.object_detector.extract_features:
        object_model = ObjectFeatureExtractor(config)
        object_model([train_ds,val_ds,test_ds])

    if config.text_encoder.extract_features:
        text_model = TextFeatureExtractor(config)
        text_model([train_ds,val_ds,test_ds])

        
    if config.classifier.train:
        train_model(config,train_ds,val_ds)

    if config.classifier.test:
        avg_loss,top1,top5 = test_model(config,test_ds)
        print("***** Train Stats *******")
        print(f"Top 1 Accuracy: {top1}")
        print(f"Top 5 Accuracy: {top5}")
        print(f"Mean Loss: {avg_loss}")
    


    

                
if __name__ == "__main__":
    main()



