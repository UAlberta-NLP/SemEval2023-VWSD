from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor
from utils import custom_processor

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

pil = Image.open('semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/image.0.jpg')
alpha_image = custom_processor([pil])
beta_image = processor(images=[pil], return_tensors='pt').pixel_values

assert np.array_equal(alpha_image, beta_image)
