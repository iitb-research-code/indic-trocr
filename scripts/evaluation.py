from transformers import VisionEncoderDecoderModel
from transformers import ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image
import pandas as pd
import fastwer
import os

# directory and file paths
test_text_file = "/home/venkat/trocr_hindi/dataset/test.txt"
root_dir = "/home/venkat/trocr_hindi/dataset/"

def dataset_generator(data_path):
    with open(data_path) as f:
        dataset = f.readlines()

    dataset_list = []
    for i in range(len(dataset)):
        image_id = dataset[i].split("\n")[0].split(' ')[0].strip()
        # vocab_id = int(dataset[i].split(",")[1].strip())
        text = dataset[i].split("\n")[0].split(' ')[1].strip()
        row = [image_id, text]
        dataset_list.append(row)

    dataset_df = pd.DataFrame(dataset_list, columns=['file_name', 'text'])
    return dataset_df
    
test_df = dataset_generator(test_text_file)
print(f"Test Shape: {test_df.shape}")


encode = 'google/vit-base-patch16-224-in21k'
decode = 'flax-community/roberta-hindi'

feature_extractor=ViTFeatureExtractor.from_pretrained(encode)
tokenizer = RobertaTokenizer.from_pretrained(decode)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained("checkpoints/checkpoint-53600")

wer = []
cer = []

for i in range(len(test_df)):
    image_path = os.path.join(root_dir, test_df['file_name'][i])
    ground_truth = test_df['text'][i]
    
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    wer_score = fastwer.score([generated_text], [ground_truth])
    wer.append(wer_score)
    cer_score = fastwer.score([generated_text], [ground_truth], char_level=True)
    cer.append(cer_score)
    
print(f"WER SCORE : {sum(wer)/len(wer)}")
print(f"CER SCORE : {sum(cer)/len(cer)}")