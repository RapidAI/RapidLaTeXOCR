# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))
import pytest
from rapid_latex_ocr import LatexOCR

model = LatexOCR()

img_dir = cur_dir / "test_files"


@pytest.mark.parametrize(
    "img_path, gt",
    [
        (
            "1.png",
            r"\exp\left[\int d^{4}x g\phi\bar{\psi}\psi\right]=\sum_{n=0}^{\infty}\frac{g^{n}}{n!}\left(\int d^{4}x\phi\bar{\psi}\psi\right)^{n}.",
        ),
        ("5.png", r"x={\frac{-b\pm{\sqrt{b^{2}-4a c\ }}}{2a}}"),
        ("2.png", r"x^{2}+y^{2}=1"),
        ("6.png", r"{\frac{x^{2}}{a^{2}}}-{\frac{y^{2}}{b^{2}}}=1"),
    ],
)
def test(img_path, gt):
    img_path = img_dir / img_path
    # img = Image.open(str(img_path))
    res = model(img_path)
    print(res)
    assert res == gt
