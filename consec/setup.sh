#!/bin/bash
# install python requirements
pip install -r $(dirname -- "$0")/requirements.txt

# download nltk wordnet
python -c "import nltk; nltk.download('wordnet')"

# download spacy
python -m spacy download en_core_web_sm

