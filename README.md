# Concept Extraction Using Pointer-Generator Networks and Distant Supervision for Data Augmentation

## Overview
This repository contains the code for running concept extraction using pretrained models described in the paper "Concept Extraction Using Pointer-Generator Networks and Distant Supervision for Data Augmentation". The preview of the dataset used for training and evaluation of concept extraction models is included. The whole dataset can be downloaded from Google Drive (~600 MB): [concept_extraction_dataset.zip](https://drive.google.com/file/d/1p992LgEVV71vTXpH_vOekjHpndCnoj92/view?usp=sharing). Details about the models and the dataset can be found in the paper.

Citation: \
Shvets, A. and Wanner, L. 2020. Concept Extraction Using Pointer-Generator Networks and Distant Supervision for Data Augmentation. In International Conference on Knowledge Engineering and Knowledge Management. Springer, Cham.

![MSA](static/architecture.png)
*The architecture of the proposed model.*

## Installation
Install into Anaconda Python Environment (recommended) \
Step 1. Download and install Anaconda ([Windows](https://repo.anaconda.com/archive/Anaconda2-5.3.0-Windows-x86_64.exe), [Mac OS X](https://repo.anaconda.com/archive/Anaconda2-5.3.0-MacOSX-x86_64.pkg), [Linux](https://repo.anaconda.com/archive/Anaconda2-5.3.0-Linux-x86_64.sh)) \
Step 2. Create conda environment:
```
# Create a new conda environment
conda create -n ce-env python=3.6

# Activate the conda environment
source activate ce-env
```

Step 3. Ensure pip is up-to-date:
```
conda update pip
```

Step 4. Run setup.py to install necessary dependencies for Python and download pretrained models:
```
cd ConceptExtraction/
python setup.py
```
or run manually:
```
pip install -r requirements.txt --user
python download_models.py
```

Step 5. Install torch from https://pytorch.org/

Step 6. Clone and install submodule dependencies
```
git submodule update --init --recursive
cd OpenNMT-py/
python setup.py install
# Copy modified translate.py to OpenNMT-py root folder
cp ../translate.py ../OpenNMT-py/translate.py
```

## Extract concepts
Run the extractor that uses pretrained models:
```
python run_concept_extraction.py -i static/example_text.txt -odir output
# In case you have an error with "torch.div", replace "torch.div" with "torch.floor_divide" in the file "OpenNMT-py/onmt/translate/beam_search.py", line 164
```
To check all available options, simply run:
```
python run_concept_extraction.py --help
```
The input file should contain one text by line (apparently, it might be a paragraph or a sentence by line).
