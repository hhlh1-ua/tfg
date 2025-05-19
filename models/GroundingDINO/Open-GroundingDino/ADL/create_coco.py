import argparse
import json
import os
from PIL import Image

def read_categories(categories_path):
    with open(categories_path, 'r') as file:
        categories = json.load(file)
    # Invertir el mapeo para usar nombres de categorías directamente
    return {v: int(k) for k, v in categories.items()}

def main():
    parser = argparse.ArgumentParser(
        description="Script para crear anotaciones COCO a partir de un archivo JSON con una lista de frames"
    )
    parser.add_argument("--frames_file", help="Ruta al archivo JSON que contiene la lista de frames", required=True)
    parser.add_argument("--outdir", help="Ruta del archivo de salida JSON con las anotaciones COCO", required=True)
    parser.add_argument("--catfile", help="Ruta del archivo JSON con las categorías", required=True)
    args = parser.parse_args()

    # Cargar la lista de archivos a partir del archivo JSON
    with open(args.frames_file, 'r') as f:
        files = json.load(f)

    # Ordenar la lista para mantener un orden consistente
    files = sorted(files)
    if not files:
        print("No se encontraron archivos en el archivo JSON proporcionado.")
        return
    files = files[::2] # Seleccionar frames de dos en dos (ejemplo: se procesará el 0, 2, 4, 6, …)
    # Inicialización de la estructura COCO
    categories = read_categories(args.catfile)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k, "supercategory": "none"} for k, v in categories.items()]
    }

    # Se define el prefijo a eliminar de cada path
    prefix = "/dataset/rgb_frames/"
    
    # Procesar TODOS los frames de la lista
    for idx, file_path in enumerate(files):
        # Eliminar el prefijo para dejar el formato "P_XX/xxxxxx.jpg"
        if file_path.startswith(prefix):
            short_file = file_path[len(prefix):]
        else:
            short_file = file_path

        image_id = idx  # Se utiliza el índice del frame como ID
        coco["images"].append({
            "id": image_id,
            "file_name": short_file,
            "width": 640,
            "height": 480
        })
        coco["annotations"].append({
            "id": image_id,
            "image_id": image_id,
            "category_id": categories.get("bed", 0),  # Se utiliza la categoría 'bed' por defecto; si no, se asigna 0.
            "bbox": [0, 0, 0, 0],
            "area": 640 * 480,
            "iscrowd": 0 
        })

    # Escribir el JSON con el formato COCO en el archivo de salida especificado
    with open(args.outdir, 'w') as out_file:
        json.dump(coco, out_file, indent=4)
    print(f"Anotaciones COCO guardadas en {args.outdir}")

if __name__ == "__main__":
    main()
