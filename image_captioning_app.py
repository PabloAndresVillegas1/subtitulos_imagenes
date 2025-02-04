import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Cargar el procesador y modelo preentrenados
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convertir el arreglo numpy a una imagen PIL y convertir a RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Procesar la imagen
    inputs = processor(raw_image, return_tensors="pt")

    # Generar un título para la imagen
    out = model.generate(**inputs,max_length=50)

    # Decodificar los tokens generados a texto
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Generación de Títulos para Imágenes",
    description="Esta es una aplicación web simple para generar títulos para imágenes utilizando un modelo entrenado."
)

iface.launch()