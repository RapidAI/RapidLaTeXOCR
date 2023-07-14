# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import re
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import yaml
from PIL import Image

from .models import EncoderDecoder
from .utils import PreProcess, TokenizerCls
from .utils_load import InputType, LoadImage, OrtInferSession

cur_dir = Path(__file__).resolve().parent
DEFAULT_CONFIG = cur_dir / "config.yaml"

model_dir = cur_dir / "models"
IMAGE_RESIZER_PATH = model_dir / "image_resizer.onnx"
ENCODER_PATH = model_dir / "encoder.onnx"
DECODER_PATH = model_dir / "decoder.onnx"
TOKENIZER_JSON = model_dir / "tokenizer.json"


class LatexOCR:
    def __init__(self, config_path: Union[str, Path] = DEFAULT_CONFIG):
        with open(config_path, "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        self.max_dims = [args.get("max_width"), args.get("max_height")]
        self.min_dims = [args.get("min_width", 32), args.get("min_height", 32)]
        self.temperature = args.get("temperature", 0.25)

        self.load_img = LoadImage()

        self.pre_pro = PreProcess(max_dims=self.max_dims, min_dims=self.min_dims)

        self.image_resizer = OrtInferSession(IMAGE_RESIZER_PATH)

        self.encoder_decoder = EncoderDecoder(
            encoder_path=ENCODER_PATH,
            decoder_path=DECODER_PATH,
            bos_token=args["bos_token"],
            eos_token=args["eos_token"],
            max_seq_len=args["max_seq_len"],
        )
        self.tokenizer = TokenizerCls(TOKENIZER_JSON)

    def __call__(self, img: InputType) -> str:
        img = self.load_img(img)
        resizered_img = self.loop_image_resizer(img)
        dec = self.encoder_decoder(resizered_img, temperature=self.temperature)
        decode = self.tokenizer.token2str(dec)
        pred = self.post_process(decode[0])
        return pred

    def loop_image_resizer(self, img: np.ndarray) -> np.ndarray:
        pillow_img = Image.fromarray(img)
        pad_img = self.pre_pro.pad(pillow_img)
        input_image = self.pre_pro.minmax_size(pad_img).convert("RGB")
        r, w, h = 1, input_image.size[0], input_image.size[1]
        for _ in range(10):
            h = int(h * r)
            final_img, pad_img = self.pre_process(input_image, r, w, h)

            resizer_res = self.image_resizer([final_img])[0]
            argmax_idx = int(np.argmax(resizer_res, axis=-1))
            w = (argmax_idx + 1) * 32
            if w == pad_img.size[0]:
                break

            r = w / pad_img.size[0]
        return final_img

    def pre_process(
        self, input_image: Image.Image, r, w, h
    ) -> Tuple[np.ndarray, Image.Image]:
        if r > 1:
            resize_func = Image.Resampling.BILINEAR
        else:
            resize_func = Image.Resampling.LANCZOS

        resize_img = input_image.resize((w, h), resize_func)
        pad_img = self.pre_pro.pad(self.pre_pro.minmax_size(resize_img))
        cvt_img = np.array(pad_img.convert("RGB"))

        gray_img = self.pre_pro.to_gray(cvt_img)
        normal_img = self.pre_pro.normalize(gray_img)
        final_img = self.pre_pro.transpose_and_four_dim(normal_img)
        return final_img, pad_img

    @staticmethod
    def post_process(s: str) -> str:
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed image
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Only img path of the formula.")
    args = parser.parse_args()

    engine = LatexOCR()
    result = engine(args.img_path)
    print(result)


if __name__ == "__main__":
    main()
