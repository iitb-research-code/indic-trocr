from transformers import VisionEncoderDecoderModel
from transformers import ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image
import matplotlib.pyplot as plt

encode = 'google/vit-base-patch16-224-in21k'
decode = 'd42kw01f/Tamil-RoBERTa'

feature_extractor=ViTFeatureExtractor.from_pretrained(encode)
tokenizer = RobertaTokenizer.from_pretrained(decode)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained("/home/pageocr/trocr/tamil/checkpoints/checkpoint-44000")

def preview(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # plt.imshow(image)
    file = open("/home/pageocr/trocr/tamil/results/result.txt", "a")
    text = generated_text + ',' + img
    file.write(text)
    print(generated_text, ",", img)

with open("/data/BADRI/IHTR/testset_small/tamil/test.txt") as f:
    for line in f:
        img = line[5:]
        image_path = "/data/BADRI/IHTR/testset_small/tamil/images/{}".format(img.strip('\n'))
        preview(image_path=image_path)
