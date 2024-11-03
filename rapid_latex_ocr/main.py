# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import yaml
from PIL import Image

from .models import EncoderDecoder
from .utils import DownloadModel, PreProcess, TokenizerCls, get_file_encode
from .utils_load import InputType, LoadImage, LoadImageError, OrtInferSession

cur_dir = Path(__file__).resolve().parent
DEFAULT_CONFIG = cur_dir / "config.yaml"


@dataclass
class LaTeXOCRInput:
    max_width: int = 672
    max_height: int = 192
    min_height: int = 32
    min_width: int = 32
    bos_token: int = 1
    max_seq_len: int = 512
    eos_token: int = 2
    temperature: float = 0.00001


class LaTeXOCR:
    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG,
        image_resizer_path: Union[str, Path] = None,
        encoder_path: Union[str, Path] = None,
        decoder_path: Union[str, Path] = None,
        tokenizer_json: Union[str, Path] = None,
    ):
        self.image_resizer_path = image_resizer_path
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.tokenizer_json = tokenizer_json

        self.get_model_path()

        file_encode = get_file_encode(config_path)
        with open(config_path, "r", encoding=file_encode) as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        input_params = LaTeXOCRInput(**args)

        self.max_dims = [input_params.max_width, input_params.max_height]
        self.min_dims = [input_params.min_width, input_params.min_height]
        self.temperature = input_params.temperature

        self.load_img = LoadImage()

        self.pre_pro = PreProcess(max_dims=self.max_dims, min_dims=self.min_dims)

        self.image_resizer = OrtInferSession(self.image_resizer_path)

        self.encoder_decoder = EncoderDecoder(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            bos_token=input_params.bos_token,
            eos_token=input_params.eos_token,
            max_seq_len=input_params.max_seq_len,
        )
        self.tokenizer = TokenizerCls(self.tokenizer_json)

    def get_model_path(
        self,
    ) -> Tuple[str]:
        def try_download(file_name):
            save_path = default_model_dir / file_name
            if save_path.exists() or downloader(file_name):
                return save_path
            raise FileNotFoundError(f"{file_name} must not be None.")

        downloader = DownloadModel()
        decoder_name = "decoder.onnx"
        encoder_name = "encoder.onnx"
        resizer_name = "image_resizer.onnx"
        tokenizer_name = "tokenizer.json"

        default_model_dir = cur_dir / "models"

        if self.image_resizer_path is None:
            self.image_resizer_path = try_download(resizer_name)

        if self.encoder_path is None:
            self.encoder_path = try_download(encoder_name)

        if self.decoder_path is None:
            self.decoder_path = try_download(decoder_name)

        if self.tokenizer_json is None:
            self.tokenizer_json = try_download(tokenizer_name)

    def __call__(self, img: InputType) -> Tuple[str, float]:
        s = time.perf_counter()

        try:
            img = self.load_img(img)
        except LoadImageError as exc:
            error_info = traceback.format_exc()
            raise LoadImageError(
                f"Load the img meets error. Error info is {error_info}"
            ) from exc

        try:
            resizered_img = self.loop_image_resizer(img)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ValueError(
                f"image resizer meets error. Error info is {error_info}"
            ) from e

        try:
            dec = self.encoder_decoder(resizered_img, temperature=self.temperature)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ValueError(
                f"EncoderDecoder meets error. Error info is {error_info}"
            ) from e

        decode = self.tokenizer.token2str(dec)
        pred = self.post_process(decode[0])

        elapse = time.perf_counter() - s
        return pred, elapse

    def loop_image_resizer(self, img: np.ndarray) -> np.ndarray:
        pillow_img = Image.fromarray(img)
        pad_img = self.pre_pro.pad(pillow_img)
        input_image = self.pre_pro.minmax_size(pad_img).convert("RGB")
        r, w, h = 1, input_image.size[0], input_image.size[1]
        for _ in range(10):
            h = int(h * r)
            final_img, pad_img = self.pre_process(input_image, r, w, h)

            resizer_res = self.image_resizer([final_img.astype(np.float32)])[0]

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
        noletter = r"[\W_^\d]"
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
    parser.add_argument("-img_resizer", "--image_resizer_path", type=str, default=None)
    parser.add_argument("-encdoer", "--encoder_path", type=str, default=None)
    parser.add_argument("-decoder", "--decoder_path", type=str, default=None)
    parser.add_argument("-tokenizer", "--tokenizer_json", type=str, default=None)
    parser.add_argument("img_path", type=str, help="Only img path of the formula.")
    args = parser.parse_args()

    engine = LaTeXOCR(
        image_resizer_path=args.image_resizer_path,
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        tokenizer_json=args.tokenizer_json,
    )

    result, elapse = engine(args.img_path)
    print(result)
    print(f"cost: {elapse:.5f}")


if __name__ == "__main__":
    main()
