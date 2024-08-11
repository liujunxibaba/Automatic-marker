from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("braintacles/brainblip").to("cpu")

image_path_or_url = r"C:\Users\33051\Desktop\lib\20240604161837.PNG"
raw_image = Image.open(requests.get(image_path_or_url, stream=True).raw) if image_path_or_url.startswith("http") else Image.open(image_path_or_url)
inputs = processor(raw_image, return_tensors="pt").to("cpu")
out = model.generate(**inputs, min_length=40, max_new_tokens=75, num_beams=5, repetition_penalty=1.40)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)