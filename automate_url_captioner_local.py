import os
import glob
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration #Modelos Blip2

# Cargar el procesador y el modelo preentrenados
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Especificar el directorio donde están tus imágenes
image_dir = "/path/to/your/images"
image_exts = ["jpg", "jpeg", "png"]  # especificar las extensiones de archivo de imagen a buscar

# Abrir un archivo para escribir los subtítulos
with open("captions.txt", "w") as caption_file:
    # Iterar sobre cada archivo de imagen en el directorio
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            # Cargar tu imagen
            raw_image = Image.open(img_path).convert('RGB')

            # No necesitas una pregunta para la generación de subtítulos de imágenes
            inputs = processor(raw_image, return_tensors="pt")

            # Generar un subtítulo para la imagen
            out = model.generate(**inputs, max_new_tokens=50)

            # Decodificar los tokens generados a texto
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Escribir el subtítulo en el archivo, precedido por el nombre del archivo de imagen
            caption_file.write(f"{os.path.basename(img_path)}: {caption}\n")