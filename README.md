# TFG hhernandez
##
Para entrenar ir a `config/defaults.yaml` y poner  `classifier.train` a True. Si quieres que se vaya ejecutando la evaluación, poner `classifier.eval` a True también. Si quieres solo split de Test, poner `classifier.train` a False
## DONE:
- [x] **Extracción de features de vídeo con TimeSformer**

  - Utilizar TimeSformer para extraer features de los vídeos.
  - Seleccionar el *class token* ya que contiene una representación consolidada de los demás tokens.
  - Guardar las features en alguno de los siguientes formatos: `.pt`, `.npy` o `.pickle`.
- [x] **Extracción de features de vídeo con ViViT y VideoMAE**

## TODO:
- [ ] **Extracción de features de vídeo con TimeSformer**

  - Ver si las features extraídas son correctas

- [ ] **Seguir la estructura de GitHub pasado**

- [ ] **Extracción de features de texto de los objetos**

  - Extraer features textuales que describan la ubicación y la clase de cada objeto presente.
  - Utilizar un encoder de texto (utilizando alguno de Hugging Face) para procesar las features textuales de los objetos.


- [ ] **Fusión de features mediante concatenación**

  - Combinar las features extraídas de vídeo y texto utilizando un MLP (Multi-Layer Perceptron) para la concatenación.
  - Si en un futuro va bien mirar Transformer
