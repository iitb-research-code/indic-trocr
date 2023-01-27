from transformers import VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_pretrained("checkpoint-800")

from PIL import Image
import matplotlib.pyplot as plt
image = Image.open("/kaggle/input/devanagiri-dataset/HindiSeg/HindiSeg/test/11/1/1.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
plt.imshow(image)
print(generated_text)
