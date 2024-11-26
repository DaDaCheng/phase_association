from setuptools import setup, find_packages

setup(
    name="harpa",              # 包名
    version="0.1.0",                       # 版本号
    packages=find_packages(),              # 自动查找所有模块
    install_requires=[                     # 指定依赖库
        # "numpy",                         # 如果有依赖项，取消注释并填写依赖包
    ],
    author="Cheng Shi",
    author_email="cheng.shi@unibas.ch",
    description="This is the source code for paper Harpa: High-Rate Phase Association with Travel Time Neural Fields.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DaDaCheng/phase_association",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)