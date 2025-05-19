import os
import yaml
import argparse
from dotmap import DotMap
from dataset.ADL import ADLDataset
import utils.control_seed as ctrl_seed
import json


def get_config():
    """
    Parsea los argumentos de línea de comando y carga el archivo de configuración YAML.
    Además, añade el nombre del archivo de salida para el JSON.
    """
    parser = argparse.ArgumentParser(
        description="Genera un archivo JSON con todos los frames extraídos de las secuencias"
    )
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo de configuración YAML")
    parser.add_argument("--output", type=str, default="frames.json", help="Archivo JSON de salida")
    args = parser.parse_args()

    # Cargar la configuración desde el YAML
    with open(args.config, "r") as f:
        config = yaml.full_load(f)
    config = DotMap(config)

    # Guardar el nombre del archivo de salida en el config
    config.output_file = args.output
    return config


def save_frames_to_json(frames, output_file):
    """
    Guarda la lista de frames en un archivo JSON.
    """
    with open(output_file, "w") as f:
        json.dump(frames, f, indent=4)
    print(f"Archivo JSON guardado en: {output_file}")


def extract_frames_flat(sequences):
    """
    Dado un listado de secuencias, extrae y aplana la parte de los frames.
    Se asume que cada secuencia es una tupla y que la lista de frames se encuentra en la posición 4.
    """
    # Usamos comprensión de listas para extraer cada frame individualmente
    return [frame for sequence in sequences if len(sequence) > 4 for frame in sequence[4]]


def main():
    # Cargar la configuración
    config = get_config()

    # Establecer la semilla para la reproducibilidad
    ctrl_seed.set_seed(config.seed)

    # Cargar los datasets: train, val y test
    train_ds = ADLDataset(config, 'train')
    val_ds   = ADLDataset(config, 'val')
    test_ds  = ADLDataset(config, 'test')
    # Extraer los frames de cada split (aplanados en una sola lista)
    frames_train = extract_frames_flat(train_ds.sequences)
    frames_val   = extract_frames_flat(val_ds.sequences)
    frames_test  = extract_frames_flat(test_ds.sequences)

    # Combinar todos los frames en una sola lista
    all_frames = frames_train + frames_val + frames_test

    # Guardar la lista total de frames en el archivo JSON
    save_frames_to_json(all_frames, config.output_file)


if __name__ == "__main__":
    main()
