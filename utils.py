import torch
import numpy as np
import os
from PIL import Image
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from multiprocessing import cpu_count

def cos_sim(img_e, txt_e) -> torch.Tensor:
  if len(txt_e.shape) == 1:
    txt_e = txt_e.reshape(txt_e.size(0), 1)
  if len(img_e.shape) == 1:
    img_e = img_e.reshape(1, img_e.size(0))
  image_norm = torch.linalg.norm(img_e, dim=1, keepdim=True)
  text_norm = torch.linalg.norm(txt_e, dim=0, keepdim=True)
  cosine_similarity = ((img_e @ txt_e) / (image_norm @ text_norm)).mT
  # if 1 in cosine_similarity.shape:
  #   cosine_similarity = cosine_similarity.flatten()
  return cosine_similarity

def cos_sim_softmax(img_e, txt_e) -> torch.Tensor:
  cosine_similarity = cos_sim(img_e, txt_e)
  sm = cosine_similarity.softmax(dim=-1)
  assert sm.shape == cosine_similarity.shape
  return sm

def dot_prod_sim(img_e, txt_e) -> torch.Tensor:
  dot_similarity = (img_e @ txt_e).T
  return dot_similarity

__replace_map = {
    'jpeg': 'jpg',
    'tiff': 'tif',
    'asp': 'aspx',
}

def clean_ext(ext: str):
    ext = ext.lower()
    for k, v in __replace_map.items():
        if ext.endswith(k):
            ext = ext.replace(k, v)
    if '?' in ext:
        ext = ext.split('?')[0]
    return ext

cosine_sim = cosine_similarity = cos_sim

image_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
image_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def normalize(image: np.ndarray):
  return (image - image_mean[:, None, None]) / image_std[:, None, None]

def centre_crop(image: Image, size: int) -> Image:
  size = (size, size)
  image_shape = (image.size[1], image.size[0])
  top = (image_shape[0] - size[0]) // 2
  bottom = top + size[0]  # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
  left = (image_shape[1] - size[1]) // 2
  right = left + size[1]  # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
  # For PIL Images we have a method to crop directly.
  return image.crop((left, top, right, bottom))

def resize(image: Image, size: int, resample: Image.Resampling) -> Image:
  width, height = image.size
  short, long = (width, height) if width <= height else (height, width)
  requested_new_short = size if isinstance(size, int) else size[0]
  if short == requested_new_short:
    return image
  new_short, new_long = requested_new_short, int(requested_new_short * long / short)
  size = (new_short, new_long) if width <= height else (new_long, new_short)
  return image.resize(size, resample=resample)

def custom_processor(images: list) -> torch.Tensor:
  processed_images = torch.stack([
    torch.from_numpy(
      normalize(
        (
          np.array(
            centre_crop(
              resize(image.convert("RGB"), size=224, resample=Image.Resampling.BICUBIC),
              size=224
            ),
            dtype=np.float32
          ) * (1 / 255.0)
        )
        .transpose(2, 0, 1)
      )
    )
    for image in images
  ], dim=0)
  return processed_images

class ParallelLoader:
  def __init__(self, data, fn, max_workers=cpu_count() - 1, save_dir='/local/storage/ogezi/v-wsd/.cache/') -> None:
    super().__init__()
    self.__shared_data = {}
    self.data = data
    self.max_workers = max_workers
    self.fn = fn
    self.save_loc = os.path.join(save_dir, 'cache_bfloat16.pt')

  def load(self) -> dict:
    if len(self.__shared_data) > 0:
      return False

    if os.path.exists(self.save_loc):
      print(f'Loading cached data from {self.save_loc}...')
      self.__shared_data = torch.load(self.save_loc)
      return False
    
    kwargs = {
      'total':  len(self.data),
      'desc': "Loading data into memory...",
    }
    if self.max_workers > 1:
      print(f'Loading original data with {self.max_workers} workers...')
      self.__shared_data = dict(process_map(
        self.fn, 
        self.data, 
        max_workers=self.max_workers, 
        **kwargs
      ))
      return True

    print(f'Loading original data with main process...')
    self.__shared_data = dict([self.fn(instance) for instance in tqdm(self.data, **kwargs)])
    return True

  def save(self) -> None:
    torch.save(self.shared_data, self.save_loc) if not os.path.exists(self.save_loc) else None

  @property
  def shared_data(self):
    return self.__shared_data
