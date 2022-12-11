'''
  We caption all the option images and do a text similarity comparison between those results and the context
'''

from email.mime import image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import argparse
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from utils import cos_sim
import termcolor
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser()
parser.add_argument('--sense-key-file', '-s', default='data/text-wsd/gold.txt')
parser.add_argument('--data', default='data/trial.data.txt')
parser.add_argument('--gold', default='data/trial.gold.txt')
parser.add_argument('--image-dir', default='data/all_images')
parser.add_argument('--model', default='openai/clip-vit-base-patch32')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

clip_model = CLIPModel.from_pretrained(args.model).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
clip_tokenizer = CLIPTokenizer.from_pretrained(args.model)

def line_to_tuple(line):
  splits = line.strip().split('\t')
  return tuple(splits)

sense_keys = [l.strip().split(' ')[1] for l in open(args.sense_key_file).readlines()]
image_data = [line_to_tuple(l) for l in open(args.data).readlines()]
gold_data = [l.strip() for l in open(args.gold).readlines()]
assert len(sense_keys) == len(image_data)
assert len(image_data) == len(gold_data)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = captioning_model.generate(pixel_values, **gen_kwargs)

  preds = captioning_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

total = 0
correct = 0
for dat, gold in zip(image_data, gold_data):
  word = dat[0]
  context = dat[1]
  gold = os.path.join(args.image_dir, gold)
  image_paths = [os.path.join(args.image_dir, i) for i in dat[2:]]
  dummy_image = Image.open('data/all_images/image.139.jpg')
  gold_idx = image_paths.index(gold)
  image_paths = sorted(image_paths, key=lambda x: (x.split('.')[1]))
  preds = predict_step(image_paths)
  print('\n'.join(preds))
  print('\n'.join(image_paths))
  # exit()
  text = [context] + preds
  clip_inputs = processor(text=text, images=[dummy_image], return_tensors="pt", padding=True).to(device)
  clip_outputs = clip_model(**clip_inputs)
  txt_e = clip_outputs.text_embeds
  txt_e = (txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)).T
  sim = cos_sim(txt_e.T, txt_e)
  # print(sim.shape, sim, sim[0, 1:])
  best = image_paths[sim[0, 1:].argmax()]
  color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
  print(word, f'"{preds}"', best, f'{gold}/{preds[gold_idx]}', '->', color, end='\n\n')
  if gold == best:
    correct += 1
  total += 1

print(f'\nAccuracy: {correct / total}')