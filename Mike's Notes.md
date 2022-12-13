Pure Text+Image Baseline
------------------------
- Run similarity search between the choice images and the text contexts
```
image.132.jpg image.86.jpg wrong
image.14.jpg image.70.jpg wrong
image.131.jpg image.107.jpg wrong
image.34.jpg image.64.jpg wrong
image.18.jpg image.18.jpg right
image.156.jpg image.156.jpg right
image.11.jpg image.54.jpg wrong
image.102.jpg image.28.jpg wrong
image.124.jpg image.124.jpg right
image.118.jpg image.118.jpg right
image.71.jpg image.71.jpg right
image.134.jpg image.134.jpg right
image.25.jpg image.25.jpg right
image.119.jpg image.149.jpg wrong
image.85.jpg image.85.jpg right
image.103.jpg image.103.jpg right

Accuracy: 0.5625
```

Image Generation Baseline
-------------------------
- Generate an image from the context then cosine similarity between the generated image(s) and the choice images
```
andromeda image.86.jpg image.86.jpg -> right
angora image.91.jpg image.70.jpg -> wrong
anteater image.131.jpg image.107.jpg -> wrong
bank image.64.jpg image.64.jpg -> right
router image.18.jpg image.18.jpg -> right
stick image.156.jpg image.156.jpg -> right
swing image.113.jpg image.54.jpg -> wrong
tube image.28.jpg image.28.jpg -> right
venus image.124.jpg image.124.jpg -> right
wheel image.140.jpg image.118.jpg -> wrong
white image.71.jpg image.71.jpg -> right
acrobatics image.134.jpg image.134.jpg -> right
adalia image.44.jpg image.25.jpg -> wrong
administration image.15.jpg image.149.jpg -> wrong
amber image.85.jpg image.85.jpg -> right
ambrosia image.103.jpg image.103.jpg -> right

Accuracy: 0.625
```

Note To Self:
-------------
Doing `n_images = [[]] * len(data)` cause multiple copies of the **same** array to be created, thus, when adding to any of the subarrays, we add to all of them. Instead, we should do `n_images = [[] for i in range(len(data))]` then each subarray is independent.

1 Accuracy: 0.6875
2 Accuracy: 0.625
3 Accuracy: 0.5625
4 Accuracy: 0.625
5 Accuracy: 0.625
6 Accuracy: 0.5625
7 Accuracy: 0.5625
8 Accuracy: 0.6875
9 Accuracy: 0.625

Text-only WSD
-------------
- Talgat used consec to disambiguate the words in context
- Ning did the same by hand
- Michael did the same by hand

### I used the scorer from Raganato et al, 2017 to measure aggreement between ConSec, Ning and Michael (P = R = F1)
- ConSec and Ning: 43.8%
- ConSec and Michael: 56.3%
- Ning and Michael: 56.3%

These scores imply that these words are highly amibiguous
