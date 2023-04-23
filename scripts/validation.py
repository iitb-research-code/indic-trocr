from transformers import VisionEncoderDecoderModel
from transformers import ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image
import matplotlib.pyplot as plt

encode = 'google/vit-base-patch16-224-in21k'
decode = 'RahulRaman/Telugu-LM-RoBERTa'

feature_extractor=ViTFeatureExtractor.from_pretrained(encode)
tokenizer = RobertaTokenizer.from_pretrained(decode)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained("/home/pageocr/trocr/telugu/checkpoints/checkpoint-16000")

def preview(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # plt.imshow(image)
    file = open('/home/pageocr/trocr/telugu/results/val.txt','a')
    text = 'test/' + img + ' ' +  generated_text + '\n'
    file.write(text)
    print(text)

with open("/data/BADRI/IHTR/validationset_small/telugu/val.txt") as f:
    for line in f:
        line = line.split()[0]
        img = line[5:]
        image_path = "/data/BADRI/IHTR/validationset_small/telugu/images/{}".format(img.strip('\n'))
        preview(image_path=image_path)
