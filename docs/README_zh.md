简体中文 | [English](https://github.com/RapidAI/RapidLatexOCR/blob/main/README.md)

## Rapid Latex OCR

<p align="left">
    <a href="https://huggingface.co/spaces/SWHL/RapidLatexOCR" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapid_latex_ocr/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid_latex_ocr"></a>
    <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


- `rapid_latex_ocr`是一个将公式图像转为latex格式的工具。
- **仓库中的推理代码来自修改自[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)，模型已经全部转为ONNX格式，并对推理代码做了精简，推理速度更快，更容易部署。**
- 仓库只有基于`ONNXRuntime`或者`OpenVINO`推理onnx格式的代码，不包含训练模型代码。如果想要训练自己的模型，请移步[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)。
- 如果有帮助到您的话，请给个小星星⭐或者赞助一杯咖啡（点击页面最上面的Sponsor中链接）
- 欢迎各位小伙伴积极贡献，让这个工具更好。

### TODO
- [ ] 基于`rapid_latex_ocr`，重写GUI版本
- [ ] 整合其他更优的模型进来

### 使用
1. 安装
    - 方法一：（推荐）pip安装`rapid_latext_ocr`库
       ```bash
       pip install rapid_latex_ocr
       ```
       - ⚠️注意：因为已经将3个模型达到了whl包，因此包较大 **(156M)**，下载时请耐心等待。各部分模型情况：
            |模型名称|大小|
            |---:|:---:|
            |`image_resizer.onnx`|37.1M|
            |`encoder.onnx`|84.8M|
            |`decoder.onnx`|48.5M|
    - 方法二： 源码安装
      1. 下载仓库源码
      2. 单独下载模型（[Google Drive](https://drive.google.com/drive/folders/1e8BgLk1cPQDSZjgoLgloFYMAQWLTaroQ?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1rnYmmKp2HhOkYVFehUiMNg?pwd=dh72)），放到`rapid_latex_ocr/models`即可。

2. 使用
    - 脚本使用：
        ```python
        from rapid_latex_ocr import LatexOCR

        model = LatexOCR()

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
        usage: rapid_latex_ocr [-h] img_path

        positional arguments:
        img_path    Only img path of the formula.

        optional arguments:
        -h, --help  show this help message and exit

        $ rapid_latex_ocr tests/test_files/6.png
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

### 更新日志
- 2023-07-15 v0.0.1 update:
  - 首次发版