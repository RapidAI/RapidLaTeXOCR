简体中文 | [English](https://github.com/RapidAI/RapidLatexOCR/blob/main/README.md)

## Rapid Latex OCR

<p align="left">
    <a href="https://swhl-rapidlatexocrdemo.hf.space" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
    <a href="https://www.modelscope.cn/studios/liekkas/RapidLatexOCRDemo/summary" target="_blank"><img src="https://img.shields.io/badge/ModelScope-Demo-blue"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pepy.tech/project/rapid_latex_ocr"><img src="https://static.pepy.tech/personalized-badge/rapid_latex_ocr?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
    <a href="https://pypi.org/project/rapid_latex_ocr/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid_latex_ocr"></a>
    <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


- `rapid_latex_ocr`是一个将公式图像转为latex格式的工具。
- **仓库中的推理代码改自[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)，模型已经全部转为ONNX格式，并对推理代码做了精简，推理速度更快，更容易部署。**
- 仓库只有基于`ONNXRuntime`或者`OpenVINO`推理onnx格式的代码，不包含训练模型代码。如果想要训练自己的模型，请移步[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)。
- 如果有帮助到您的话，请给个小星星⭐或者赞助一杯咖啡（点击页面最上面的Sponsor中链接）
- 欢迎各位小伙伴积极贡献，让这个工具更好。
- ☆ [模型转换笔记](https://github.com/RapidAI/RapidLatexOCR/wiki/Model-Conversion-Notes)

### [Demo](https://swhl-rapidlatexocrdemo.hf.space)

<div align="center">
    <img src="https://github.com/RapidAI/RapidLatexOCR/releases/download/v0.0.0/demo.gif" width="100%" height="100%">
</div>

### TODO
- [ ] 基于`rapid_latex_ocr`，重写GUI版本
- [x] 在hugging face上添加在线demo
- [ ] 整合其他更优的模型进来
- [ ] 添加对OpenVINO的支持

### 使用
1. 安装
    1. pip安装`rapid_latext_ocr`库。因将模型打包到whl包中超出pypi限制（100M），因此需要单独下载模型。
        ```bash
        pip install rapid_latex_ocr
        ```
    2. 下载模型（[Google Drive](https://drive.google.com/drive/folders/1e8BgLk1cPQDSZjgoLgloFYMAQWLTaroQ?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1rnYmmKp2HhOkYVFehUiMNg?pwd=dh72)），初始化时，指定模型路径即可，详细参见下一部分。

          |模型名称|大小|
          |---:|:---:|
          |`image_resizer.onnx`|37.1M|
          |`encoder.onnx`|84.8M|
          |`decoder.onnx`|48.5M|

2. 使用
    - 脚本使用：
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
            data = f.read()

        result, elapse = model(data)

        print(result)
        # {\frac{x^{2}}{a^{2}}}-{\frac{y^{2}}{b^{2}}}=1

        print(elapse)
        # 0.4131628000000003
        ```
    - 命令行使用
        ```bash
        $ rapid_latex_ocr -h
        usage: rapid_latex_ocr [-h] [-img_resizer IMAGE_RESIZER_PATH]
                            [-encdoer ENCODER_PATH] [-decoder DECODER_PATH]
                            [-tokenizer TOKENIZER_JSON]
                            img_path

        positional arguments:
        img_path              Only img path of the formula.

        optional arguments:
        -h, --help            show this help message and exit
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
3. 输入输出说明
   - **输入(`Union[str, Path, bytes]`)**：只含有公式的图像。
   - **输出(`Tuple[str, float]`)**： `(识别结果, 耗时)`， 具体参见下例：
       ```python
       (
          '{\\frac{x^{2}}{a^{2}}}-{\\frac{y^{2}}{b^{2}}}=1',
          0.47902780000000034
       )
       ```

### 👫 Community
我们将[RapidLatexOCR Discussions](https://github.com/RapidAI/RapidLatexOCR/discussions) 作为社区交流平台。 所有的想法和问题都可以在这里提出和讨论。

### 更新日志
- 2023-07-15 v0.0.1 update:
  - 首次发版
