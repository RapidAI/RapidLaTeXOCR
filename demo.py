# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapid_latex_ocr import LatexOCR

img_path = "tests/test_files/6.png"
model = LatexOCR()
with open(img_path, "rb") as f:
    data = f.read()
print(model(data))
