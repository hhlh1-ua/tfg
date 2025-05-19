import os
import json, glob
import argparse
import re

ORIGINAL_HEIGTH = 960
ORIGINAL_WIDTH = 1280
# En el servidor las imágenes tienen la mitad de tamaño que lo indicado en el paper
HEIGTH = 480
WIDTH = 640

category_names = {}  # Diccionario para asignar ID a categorías
categories = []

def create_label_map_extended(test, categories):
    filename = "label_map_ADL.json"
    path = os.path.join('..', 'config', filename)
    inverted = {value: key for key, value in categories.items()}
    inverted_str = {str(key): inverted[key] for key in sorted(inverted.keys())}
    with open(path, 'w') as f:
        json.dump(inverted_str, f, indent=2)

def update_coco2ovdgADL(categories):
    coco2ovdg_ADL_file = os.path.join('..', 'tools', 'coco2odvg_ADL.py')
    new_id_map = {value: value for value in categories.values()}  # Cada ID mapea a sí mismo
    new_id_map = str(new_id_map)  # Convertir a cadena, si es necesario

    inverted = {value: key for key, value in categories.items()}
    inverted_str = {str(key): inverted[key] for key in sorted(inverted.keys())}
    inverted_str = json.dumps(inverted_str, indent=4)
    # Leer el contenido del archivo
    with open(coco2ovdg_ADL_file, 'r') as file:
        content = file.read()

    # Reemplazar id_map en el archivo
    content = re.sub(r'id_map\s*=\s*\{[^\}]*\}', f'id_map = {new_id_map}', content)

    # Reemplazar ori_map en el archivo
    content = re.sub(r'ori_map\s*=\s*\{[^\}]*\}', f'ori_map = {inverted_str}', content)

    # Escribir el contenido actualizado de vuelta al archivo
    with open(coco2ovdg_ADL_file, 'w') as file:
        file.write(content)

def updat_cfg(file, categories):
    sorted_categories = sorted(categories.items(), key=lambda x: x[1])
    label_list = [cat_name for cat_name, cat_id in sorted_categories]
    new_label_list_str = json.dumps(label_list)
    with open(file, 'r') as f:
        content = f.read()
    pattern = r'(label_list\s*=\s*)\[[^\]]*\]'
    new_content = re.sub(pattern, r'\1' + new_label_list_str, content, flags=re.DOTALL)
    with open(file, 'w') as f:
        f.write(new_content)

def getCategories(video_list=None):
    """
    Genera el diccionario de categorías a partir de los archivos de anotaciones.
    Si se proporciona video_list, se itera sobre esa lista; de lo contrario, se usan
    los videos P_01 a P_20.
    """
    global category_names, categories
    category_id_counter = 0
    if video_list is None:
        video_list = [f'P_{i:02d}' for i in range(1, 21)]
    for video in video_list:
        filename = f'object_annot_{video}.txt'
        filedir = os.path.join('annotations', 'object_annotation', filename)
        with open(filedir, "r") as file:
            for line in file:
                line_vals = line.split()
                category_name = line_vals[-1]  # La última palabra es la categoría
                if category_name not in category_names:
                    category_names[category_name] = category_id_counter
                    categories.append({
                        "id": category_id_counter,
                        "name": category_name,
                        "supercategory": "none"
                    })
                    category_id_counter += 1

def createCOCO_from_list(outfile, video_list):
    """
    Crea el archivo COCO a partir de una lista de videos.
    Cada video se asume que tiene dos archivos:
      - object_annot_<video>.txt
      - object_annot_<video>_annotated_frames.txt
    """
    images = []       # Lista de imágenes con IDs
    annotations = []  # Lista de anotaciones
    images_dict = {}  # Diccionario para asignar ID a imágenes
    image_id_counter = 0
    annotation_id_counter = 1
    errores = []
    
    for video in video_list:
        filename = f'object_annot_{video}.txt'
        filedir = os.path.join('annotations', 'object_annotation', filename)
        annotated_frames = f'object_annot_{video}_annotated_frames.txt'
        filedir_annotated_frames = os.path.join('annotations', 'object_annotation', annotated_frames)
        print(f'Obteniendo anotaciones para frames en el video {video}')
        
        # Procesar archivo de frames anotados
        with open(filedir_annotated_frames, "r") as file:
            for line in file:
                line_vals = line.split()
                image_name = line_vals[0][2:]  # Se asume que se omiten los dos primeros caracteres
                image_name = os.path.join(video, image_name + '.jpg')
                if image_name not in images_dict:
                    images_dict[image_name] = image_id_counter
                    images.append({
                        "id": image_id_counter,
                        "width": WIDTH,
                        "height": HEIGTH,
                        "file_name": image_name
                    })
                    image_id_counter += 1
        
        # Procesar archivo de anotaciones
        with open(filedir, "r") as file:
            for line in file:
                line_vals = line.split()
                category_name = line_vals[-1]  # Última palabra (categoría)
                image_name = line_vals[5][2:]  # Se omiten los dos primeros caracteres
                image_name = os.path.join(video, image_name + '.jpg')
                if image_name in images_dict:
                    x1 = int(line_vals[1])
                    y1 = int(line_vals[2])
                    x2 = int(line_vals[3])
                    y2 = int(line_vals[4])
                    w_bbox = x2 - x1
                    h_bbox = y2 - y1
                    bbox = [x1, y1, w_bbox, h_bbox]
                    annotation = {
                        "id": annotation_id_counter,
                        "image_id": images_dict[image_name],
                        "category_id": category_names[category_name],
                        "segmentation": [],
                        "area": w_bbox * h_bbox,
                        "bbox": bbox,
                        "iscrowd": 0,
                        "ignore": 0
                    }
                    annotations.append(annotation)
                    annotation_id_counter += 1
                else:
                    errores.append({
                        "video": video,
                        "frame": image_name
                    })
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    outdir = os.path.join('annotations', 'coco_format', outfile)
    with open(outdir, 'w') as out_file:
        json.dump(coco, out_file, indent=4)

def main():
    # Definición de las listas para train, validation y test
    # train_list = ['P_14', 'P_12', 'P_18', 'P_19', 'P_04', 'P_10', 'P_03', 'P_13', 'P_05', 'P_06', 'P_20', 'P_09']
    train_list = ['P_03','P_04','P_05', 'P_06','P_09','P_10','P_12','P_13','P_14','P_18', 'P_19','P_20']
    val_list   = ['P_01', 'P_02', 'P_16', 'P_17']
    test_list  = ['P_07', 'P_08', 'P_11', 'P_15']
    # Unión de todas las listas para extraer las categorías (se asume que cubren todos los videos)
    all_list = train_list + val_list + test_list

    getCategories(all_list)
    createCOCO_from_list(outfile='annotations_train.json', video_list=train_list)
    update_coco2ovdgADL(category_names)
    create_label_map_extended(test=False, categories=category_names)
    createCOCO_from_list(outfile='annotations_test.json', video_list=test_list)
    create_label_map_extended(test=True, categories=category_names)
    createCOCO_from_list(outfile='annotations_val.json', video_list=val_list)
    cfg_file = os.path.join('..', 'config', 'cfg_ADL.py')
    updat_cfg(cfg_file, category_names)

if __name__ == '__main__':
    main()
