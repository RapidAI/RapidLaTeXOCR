## rapid_latex_ocr
<p align="left">
    <a href="https://huggingface.co/spaces/SWHL/RapidLatexOCR" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pepy.tech/project/rapid_latex_ocr"><img src="https://static.pepy.tech/personalized-badge/rapid_latex_ocr?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
    <a href="https://pypi.org/project/rapid_latex_ocr/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid_latex_ocr"></a>
    <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


- `rapid_latex_ocr` is a tool to convert formula images to latex format.
- **The reasoning code in the repo is modified from [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR), the model has all been converted to ONNX format, and the reasoning code has been simplified, Inference is faster and easier to deploy.**
- The repo only has codes based on `ONNXRuntime` or `OpenVINO` inference in onnx format, and does not contain training model codes. If you want to train your own model, please move to [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR).
- If it helps you, please give a little star ⭐ or sponsor a cup of coffee (click the link in Sponsor at the top of the page)
- Welcome all friends to actively contribute to make this tool better.


### Installation
1. pip install `rapid_latext_ocr` library. Because packaging the model into the whl package exceeds the pypi limit (100M), the model needs to be downloaded separately.
    ```bash
    pip install rapid_latex_ocr
    ```
2. Download the model ([Google Drive](https://drive.google.com/drive/folders/1e8BgLk1cPQDSZjgoLgloFYMAQWLTaroQ?usp=sharing) | [Baidu NetDisk](https://pan.baidu.com/s/1rnYmmKp2HhOkYVFehUiMNg?pwd=dh72)), when initializing, just specify the model path, see the next part for details.

    |model name|size|
    |---:|:---:|
    |`image_resizer.onnx`|37.1M|
    |`encoder.onnx`|84.8M|
    |`decoder.onnx`|48.5M|


### Usage
- Used by python script:
    ```python
    from rapid_latex_ocr import LatexOCR

    image_resizer_path = 'models/image_resizer.onnx'
    encoder_path = 'models/encoder.onnx'
    decoder_path = 'models/decoder.onnx'
    tokenizer_json = 'models/tokenizer.json'
    model = LatexOCR(image_resizer_path=image_resizer_path,
                    encoder_path=encoder_path,
                    decoder_path=decoder_path,
                    tokenizer_json=tokenizer_json)

    img_path = "tests/test_files/6.png"
    with open(img_path, "rb") as f:
        data = f. read()

    result, elapse = model(data)

    print(result)
    # {\frac{x^{2}}{a^{2}}}-{\frac{y^{2}}{b^{2}}}=1

    print(elapse)
    # 0.4131628000000003
    ```
- Used by command line.
    ```bash
    $ rapid_latex_ocr -h
    usage: rapid_latex_ocr [-h] [-img_resizer IMAGE_RESIZER_PATH]
                        [-encdoer ENCODER_PATH] [-decoder DECODER_PATH]
                        [-tokenizer TOKENIZER_JSON]
                        img_path

    positional arguments:
    img_path Only img path of the formula.

    optional arguments:
    -h, --help show this help message and exit
    -img_resizer IMAGE_RESIZER_PATH, --image_resizer_path IMAGE_RESIZER_PATH
    -encdoer ENCODER_PATH, --encoder_path ENCODER_PATH
    -decoder DECODER_PATH, --decoder_path DECODER_PATH
    -tokenizer TOKENIZER_JSON, --tokenizer_json TOKENIZER_JSON

    $ rapid_latex_ocr tests/test_files/6.png \
        -img_resizer models/image_resizer.onnx \
        -encoder models/encoder.onnx \
        -dedocer models/decoder.onnx \
        -tokenizer models/tokenizer.json
    # ('{\\frac{x^{2}}{a^{2}}}-{\\frac{y^{2}}{b^{2}}}=1', 0.47902780000000034)
    ```

### See details for [RapidLatexOCR](https://github.com/RapidAI/RapidLatexOCR)