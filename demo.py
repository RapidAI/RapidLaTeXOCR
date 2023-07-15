# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapid_latex_ocr import LatexOCR

image_resizer_path = "models/image_resizer.onnx"
encoder_path = "models/encoder.onnx"
decoder_path = "models/decoder.onnx"
tokenizer_json = "models/tokenizer.json"
model = LatexOCR(
    image_resizer_path=image_resizer_path,
    encoder_path=encoder_path,
    decoder_path=decoder_path,
    tokenizer_json=tokenizer_json,
)

img_path = "tests/test_files/6.png"
with open(img_path, "rb") as f:
    data = f.read()

res, elapse = model(data)

print(res)
print(elapse)
