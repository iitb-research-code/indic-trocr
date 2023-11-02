import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from datasets import load_metric
from config import *
os.environ["WANDB_DISABLED"] = "true"
# torch.cuda.empty_cache()


# Access a specific decoder by language
language = 'Hindi'  # Replace with the desired language
decoder = DECODERS.get(language, None)

# directory and file paths, change accordingly
train_text_file = "D:/Documents/iitb/indic-trocr/indic-trocr/hindi/train.txt"
test_text_file = "D:/Documents/iitb/indic-trocr/indic-trocr/hindi/test.txt"
val_text_file = "D:/Documents/iitb/indic-trocr/indic-trocr/hindi/val.txt"
root_dir = "D:/Documents/iitb/indic-trocr/indic-trocr/hindi/"

def dataset_generator_lang_set1(data_path):
    with open(data_path) as f:
        dataset = f.readlines()
    # counter = 0 

    dataset_list = []
    for i in range(len(dataset)):
        # if counter > 199: 
            # break           
        image_id = dataset[i].split("\n")[0].split(' ')[0].strip()
        # vocab_id = int(dataset[i].split(",")[1].strip())
        text = dataset[i].split("\n")[0].split(' ')[1].strip()
        row = [image_id, text]
        dataset_list.append(row)
        # counter += 1 

    dataset_df = pd.DataFrame(dataset_list, columns=['file_name', 'text'])
    # dataset_df.head()
    return dataset_df

def dataset_generator_lang_set2(data_path, set_name):
    with open(data_path) as f:
        dataset = f.readlines()

    with open("/content/sample_data/bengali/vocab.txt") as f:
        vocab = f.readlines()

    for j in range(len(vocab)):
        vocab[j] = vocab[j].split("\n")[0].strip()

    dataset_list = []
    for i in range(len(dataset)):
        image_id = dataset[i].split("\n")[0].split(',')[0].strip()
        image_id = set_name + "/" + image_id
        vocab_id = int(dataset[i].split("\n")[0].split(',')[1].strip())
        text = vocab[vocab_id]
        row = [image_id, text]
        dataset_list.append(row)

    dataset_df = pd.DataFrame(dataset_list, columns=['file_name', 'text'])
    return dataset_df
    
if language in lang_set1:
    train_df = dataset_generator_lang_set1(train_text_file)
    test_df = dataset_generator_lang_set1(test_text_file)
    val_df = dataset_generator_lang_set1(val_text_file)
elif language in lang_set2:
    train_df = dataset_generator_lang_set2(train_text_file, 'train')
    test_df = dataset_generator_lang_set2(test_text_file, 'test')
    val_df = dataset_generator_lang_set2(val_text_file, 'val')
else:
    print('language not supported')


print(f"Train, Test & Val shape: {train_df.shape, test_df.shape, val_df.shape}")

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
        # print(encoding)
        return encoding


encode = ENCODER
decode = decoder

feature_extractor=ViTFeatureExtractor.from_pretrained(encode)
tokenizer = RobertaTokenizer.from_pretrained(decode)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

train_dataset = IAMDataset(root_dir=root_dir,
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir=root_dir,
                           df=test_df,
                           processor=processor)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode, decode)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
print(f"processor.tokenizer.pad_token_id: {processor.tokenizer.pad_token_id}")
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

training_args = Seq2SeqTrainingArguments(
    num_train_epochs=50,
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    output_dir="./checkpoints/",
    logging_steps=2,
    save_steps=2000,
    eval_steps=100,
)

cer_metric = load_metric("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

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

os.makedirs("model/")
model.save_pretrained("model/")
