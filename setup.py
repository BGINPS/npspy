from setuptools import setup, find_packages


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

exec(open('npspy/version.py').read())

setup(
    name="npspy",
    version=__version__,
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    #packages=["npspy"],
    packages=find_packages(),
    author="Hailin Pan, Ji Wang",
    author_email="panhailin@genomics.cn, wangji1@genomics.cn",
    description="npspy is a general toolkit for processing protein electrical signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD 2-Clause License",
    url="https://github.com/HailinPan/npspy",
)
