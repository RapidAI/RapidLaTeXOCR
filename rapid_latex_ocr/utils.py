# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import List, Union, Optional

import cv2
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import BPE
import requests
import tqdm
import io


class PreProcess:
    def __init__(self, max_dims: List[int], min_dims: List[int]):
        self.max_dims, self.min_dims = max_dims, min_dims
        self.mean = np.array([0.7931, 0.7931, 0.7931]).astype(np.float32)
        self.std = np.array([0.1738, 0.1738, 0.1738]).astype(np.float32)

    @staticmethod
    def pad(img: Image.Image, divable: int = 32) -> Image.Image:
        """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

        Args:
            img (PIL.Image): input image
            divable (int, optional): . Defaults to 32.

        Returns:
            PIL.Image
        """
        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)

        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b : b + h, a : a + w]
        im = Image.fromarray(rect).convert("L")
        dims: List[Union[int, int]] = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))

        padded = Image.new("L", tuple(dims), 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def minmax_size(
        self,
        img: Image.Image,
    ) -> Image.Image:
        """Resize or pad an image to fit into given dimensions

        Args:
            img (Image): Image to scale up/down.

        Returns:
            Image: Image with correct dimensionality
        """
        if self.max_dims is not None:
            ratios = [a / b for a, b in zip(img.size, self.max_dims)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                img = img.resize(tuple(size.astype(int)), Image.BILINEAR)

        if self.min_dims is not None:
            padded_size: List[Union[int, int]] = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, self.min_dims)
            ]

            new_pad_size = tuple(padded_size)
            if new_pad_size != img.size:  # assert hypothesis
                padded_im = Image.new("L", new_pad_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def normalize(self, img: np.ndarray, max_pixel_value=255.0) -> np.ndarray:
        mean = self.mean * max_pixel_value
        std = self.std * max_pixel_value
        denominator = np.reciprocal(std, dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

    @staticmethod
    def to_gray(img) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def transpose_and_four_dim(img: np.ndarray) -> np.ndarray:
        return img.transpose(2, 0, 1)[:1][None, ...]


class TokenizerCls:
    def __init__(self, json_file: Union[Path, str]):
        self.tokenizer = Tokenizer(BPE()).from_file(str(json_file))

    def token2str(self, tokens) -> List[str]:
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]

        dec = [self.tokenizer.decode(tok.tolist()) for tok in tokens]
        return [
            "".join(detok.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
            for detok in dec
        ]


class DownloadModel:
    """Modified from https://github.com/lukas-blecher/LaTeX-OCR/blob/1781514fb8c92ea9f94057295fdae0e683f4648e/pix2tex/model/checkpoints/get_latest_checkpoint.py"""

    def __init__(self) -> None:
        self.url = "https://github.com/RapidAI/RapidLaTeXOCR/releases/download/v0.0.0"
        self.cur_dir = Path(__file__).resolve().parent

    def __call__(self, file_name: str) -> bool:
        save_dir = self.cur_dir / "models"
        save_dir.mkdir(parents=True, exist_ok=True)

        full_path = f"{self.url}/{file_name}"
        print(f"Download {full_path} to {self.cur_dir}/models")

        try:
            file = self.download_as_bytes_with_progress(full_path, file_name)
            save_file_path = save_dir / file_name
            self.save_file(save_file_path, file)
        except Exception:
            return False
        return True

    @staticmethod
    def download_as_bytes_with_progress(url: str, name: Optional[str] = None) -> bytes:
        resp = requests.get(url, stream=True, allow_redirects=True)
        total = int(resp.headers.get("content-length", 0))
        bio = io.BytesIO()
        with tqdm.tqdm(
            desc=name, total=total, unit="b", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                bar.update(len(chunk))
                bio.write(chunk)
        return bio.getvalue()

    @staticmethod
    def save_file(save_path: Union[str, Path], file: bytes):
        with open(save_path, "wb") as f:
            f.write(file)


if __name__ == "__main__":
    downloader = DownloadModel()
    downloader("decoder.onnx")
