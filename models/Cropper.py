from PIL import Image


class Cropper:
    def get_crops(self, image_path, bboxes):
        """
        Dado el path de una imagen y un listado de bounding boxes en el formato
        (x1, y1, x2, y2, clase), abre la imagen y retorna una lista de recortes (crops)


        Parámetros:
            image_path: Ruta de la imagen a abrir.
            bboxes: Lista de tuplas (x1, y1, x2, y2, clase).
            max_crops: Número máximo de recortes a generar.

        Retorna:
            Una crops que es la lista de imágenes recortadas.
        """

        
        # Abrir la imagen dentro del método
        image = Image.open(image_path).convert("RGB")
        crops = []
        
        # Iteramos cada bounding box y, si no se supera max_crops, generamos el recorte
        for (x1, y1, x2, y2, clase) in bboxes:



            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        
        return crops