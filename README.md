# Indic TrOCR - Transformer based OCR for Indian Languages

TrOCR is an OCR (Optical Character Recognition) model proposed by Minghao Li et al. in their paper titled <a href="https://arxiv.org/abs/2109.10282">TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models</a>. This model is composed of an image Transformer encoder and an autoregressive text Transformer decoder, enabling it to accurately perform OCR.

In this repository, you will find TrOCR, an OCR model specifically developed for recognizing handwritten Indian documents in various languages including Hindi, Tamil, Malayalam, Bengali, and more. The TrOCR model has been designed to accurately detect and convert text in these languages from images of handwritten documents, making it a valuable tool for various applications such as digitizing old documents, extracting information from scanned documents, and more.

## Installation

```
git clone https://github.com/iitb-research-code/indic-trocr.git
cd indic-trocr
virtualenv trocr_env
source trocr_env/bin/activate
pip install requirements.txt
```

## Train for a new Language

1. Download the Word level Handwritten dataset for that language from the [IIIT-HW Datasets](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data). This folder contains the train, test and val word-level images and their corresponding text labels in train.txt, test.txt and val.txt files.
2. Change the train, test, val text file and root directory paths in ``train.py``.
3. We need a language specific RoBERTa decoder model for training TrOCR. Find a RoBERTa model for that language on [Hugging Face](http://huggingface.co). ([Kannada Bert](https://huggingface.co/RahulRaman/Kannada-LM-RoBERTa))
4. Copy the model name from Hugging Face and change the ``decode`` variable on ``line 80`` in ``train.py``.
5. Change the ``training_args`` on ``line 113`` in ``train.py`` if required.
6. The model is ready for training. Run the following command:
```
python train.py
```



