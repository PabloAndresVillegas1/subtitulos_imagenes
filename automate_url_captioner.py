import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Cargar el procesador y el modelo preentrenados
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL de la página a raspar
url = "https://en.wikipedia.org/wiki/IBM"

# Descargar la página
response = requests.get(url)
# Analizar la página con BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Encontrar todos los elementos img
img_elements = soup.find_all('img')

# Abrir un archivo para escribir los subtítulos
with open("captions.txt", "w") as caption_file:
    # Iterar sobre cada elemento img
    for img_element in img_elements:
        img_url = img_element.get('src')

        # Saltar si la imagen es un SVG o demasiado pequeña (probablemente un ícono)
        if 'svg' in img_url or '1x1' in img_url:
            continue

        # Corregir la URL si está mal formada
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue  # Saltar URLs que no comienzan con http:// o https://

        try:
            # Descargar la imagen
            response = requests.get(img_url)
            # Convertir los datos de la imagen a una imagen PIL
            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:  # Saltar imágenes muy pequeñas
                continue

            raw_image = raw_image.convert('RGB')

            # Procesar la imagen
            inputs = processor(raw_image, return_tensors="pt")
            # Generar un subtítulo para la imagen
            out = model.generate(**inputs, max_new_tokens=50)
            # Decodificar los tokens generados a texto
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Escribir el subtítulo en el archivo, precedido por la URL de la imagen
            caption_file.write(f"{img_url}: {caption}\n")
        except Exception as e:
            print(f"Error procesando la imagen {img_url}: {e}")
            continue