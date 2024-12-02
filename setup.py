from setuptools import setup, find_packages

setup(
    name="harpa",              
    version="0.1.0",                       
    packages=find_packages(),              
    install_requires=[                     
        
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