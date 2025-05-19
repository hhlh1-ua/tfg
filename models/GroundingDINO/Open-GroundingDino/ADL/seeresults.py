import cv2
import torch
import json
import random

# Cargar la imagen
image_path = '/dataset/rgb_frames/P_11/000000.jpg'
image = cv2.imread(image_path)

# Cargar el label map
with open('../config/label_map_ADL.json', 'r') as file:
    label_map = json.load(file)

# Cargar el objeto con las cajas detectadas
model_path = '../resultados/pre_finetunning/Version_Oficial/results-0.pkl'
object = torch.load(model_path)

# Iterar por cada bounding box
for bbox_data in object['res_info'][0]:
    # Se espera que bbox_data tenga el formato: [x1, y1, x2, y2, confidence, label]
    x1, y1, x2, y2, confidence, label = bbox_data
    if confidence > 0.35:
        label_str = label_map[str(int(label))]
        print(f'Label: {label_str}, Confidence: {confidence}')
        
        # Generar un color aleatorio (en formato BGR, como lo usa OpenCV)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Dibujar la caja y agregar el texto usando el mismo color
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label_str, (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Guardar la imagen con las cajas
cv2.imwrite('../resultados/pre_finetunning/version_GR_oficial.jpg', image)
