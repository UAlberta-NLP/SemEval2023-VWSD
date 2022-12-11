import torch

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