from nltk.corpus import wordnet as wn
import emoji

path = '/home/ogezi/ideas/v-wsd/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt'
output_path = '/home/ogezi/ideas/v-wsd/semeval-2023-task-1-V-WSD-train-v1/train_v1/not_in_wn_train.data.v1.txt'
emoji_path = '/home/ogezi/ideas/v-wsd/semeval-2023-task-1-V-WSD-train-v1/train_v1/emoji_train.data.v1.txt'

f = open(path).readlines()
ok = [i for i in f if wn.synsets(i.split('\t')[0]) != []]
not_ok = [f'{idx + 1}\t{i}' for idx, i in enumerate(f) if wn.synsets(i.split('\t')[0]) == []]
emojis = [f'{idx + 1}\t{i}' for idx, i in enumerate(f) if emoji.is_emoji(i.split('\t')[0])]

print(f'In WN count: {len(ok)}; coverage: {len(ok) / len(f) * 100.}%')
print(f'Not in WN count: {len(not_ok)}; coverage: {len(not_ok) / len(f) * 100.}%')
print(f'Emojis count: {len(emojis)}; coverage: {len(emojis) / len(f) * 100.}%')

open(output_path, 'w').write(''.join(not_ok))
open(emoji_path, 'w').write(''.join(emojis))