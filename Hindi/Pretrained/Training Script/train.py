import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.environ["WANDB_DISABLED"] = "true"


with open('../input/devanagiri-dataset/train.txt') as f:
    train = f.readlines()
counter = 0

train_list = []
for i in range(len(train)):
    if counter > 5000:
        break
    image_id = train[i].split("\n")[0].split(' ')[0].strip()
#     vocab_id = int(train[i].split(",")[1].strip())
    text = train[i].split("\n")[0].split(' ')[1].strip()
    row = [image_id, text]
    train_list.append(row)
    counter += 1

train_df = pd.DataFrame(train_list, columns=['file_name', 'text'])
# train_df.head()

with open('../input/devanagiri-dataset/test.txt') as f:
    test = f.readlines()

counter = 0
test_list = []
for i in range(len(test)):
    if counter > 2000:
        break
    image_id = test[i].split("\n")[0].split(' ')[0].strip()
#     vocab_id = int(train[i].split(",")[1].strip())
    text = test[i].split("\n")[0].split(' ')[1].strip()
    row = [image_id, text]
    test_list.append(row)
    counter += 1

test_df = pd.DataFrame(test_list, columns=['file_name', 'text'])
# test_df.head()

with open('../input/devanagiri-dataset/val.txt') as f:
    val = f.readlines()
counter = 0
val_list = []
for i in range(len(val)):
    if counter > 2000:
        break
    image_id = val[i].split("\n")[0].split(' ')[0].strip()
#     vocab_id = int(train[i].split(",")[1].strip())
    text = val[i].split("\n")[0].split(' ')[1].strip()
    row = [image_id, text]
    val_list.append(row)
    counter += 1
    
val_df = pd.DataFrame(val_list, columns=['file_name', 'text'])
# val_df.head()

print(f"Train, Test & Val Shape{train_df.shape, test_df.shape, val_df.shape}")

import torch
from torch.utils.data import Dataset
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
#         print(encoding)
        return encoding
    
from transformers import ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor, PreTrainedTokenizerFast
from transformers import VisionEncoderDecoderModel
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import TrOCRProcessor

encode = 'google/vit-base-patch16-224-in21k'
decode = 'flax-community/roberta-hindi'

feature_extractor=ViTFeatureExtractor.from_pretrained(encode)
tokenizer = RobertaTokenizer.from_pretrained(decode)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

train_dataset = IAMDataset(root_dir='../input/devanagiri-dataset/HindiSeg/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir='../input/devanagiri-dataset/HindiSeg/',
                           df=test_df,
                           processor=processor)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode, decode)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
print(processor.tokenizer.pad_token_id)
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
# config_decoder.is_decoder = True
# config_decoder.add_cross_attention = True

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir="./",
    logging_steps=2,
    save_steps=200,
    eval_steps=100,
)

from datasets import load_metric
cer_metric = load_metric("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

trainer.train()