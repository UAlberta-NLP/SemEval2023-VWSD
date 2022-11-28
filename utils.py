import torch

def cos_sim(img_e, txt_e) -> torch.Tensor:
  image_norm = torch.linalg.norm(img_e, dim=1, keepdim=True)
  text_norm = torch.linalg.norm(txt_e, dim=0, keepdim=True)
  cosine_similarity = ((img_e @ txt_e) / (image_norm @ text_norm)).T
  return cosine_similarity

cosine_sim = cosine_similarity = cos_sim