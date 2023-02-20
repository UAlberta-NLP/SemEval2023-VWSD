--------------------------------------------------------------------------------------------------
         SemEval-2023 Task-1 - V-WSD: Visual Word Sense Disambiguation (V-WSD)


  Alessandro Raganato, Iacer Calixto, Jose Camacho-Collados, Asahi Ushio, Mohammad Taher Pilehvar
--------------------------------------------------------------------------------------------------


Task: Given a potentially ambiguous word and some limited textual context, the task is to select among a set of ten candidate images the one which corresponds to the intended meaning of the target word.


This package contains the test data for three languages, English (en), Farsi (fa), and Italian (it). 


The test dataset contains a folder named "test_images", and one file (each line corresponds to an instance):


    * {en,fa,it}.test.data.txt: This is a tab-separated file containing the following items: 


        target_word <tab> full_phrase <tab> image_1 <tab> image_2 <tab> ... <tab> image_10


            - "target_word": the potentially ambiguous target word.


               - "full_phrase": limited textual context containing the target_word.


            - "image_i": corresponds to the "i"th image. In this version all images are in the test_images folder. 


	* {en,fa,it}.test.gold.txt: This file contains the gold labels. Each line contains the name of the gold image for each corresponding data instance.


Following the Codalab link for the evaluation:
https://codalab.lisn.upsaclay.fr/competitions/8190


For further details, please see https://raganato.github.io/vwsd/