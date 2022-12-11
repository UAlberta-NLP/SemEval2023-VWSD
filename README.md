# V-WSD

## disambiguate.py
Use this script to disambiguate target words in context.

Installation:

- Install pytorch following instructions from the [official website](https://pytorch.org/get-started/locally/).
- Install dependencies
```bash consec/setup.sh```
- Download the ConSec "consec_wngt_best.ckpt" checkpoint from [here](https://drive.google.com/file/d/1dwzQ7QDwe8hH4pGBBe-5g4N_BI2eLDfA/view?usp=sharing)
- Move "consec_wngt_best.ckpt" to consec/checkpoints
- Run disambigute.py in the following way
```PYTHONPATH=$(PWD)/consec python3 disambiguate.py --data_path data/trial.data.txt --output consec.tsv```
where --data_path is the SE23 dataset path and --output is the path where you would like to save the output