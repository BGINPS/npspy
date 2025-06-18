# npspy
In addition to mass spectrometry, protein sequences can also be sequenced through electrical signals. npspy is a general python toolkit for processing protein electrical signals.

# Installation
1. create conda env:
```
conda create -n npspy_env python=3.12.5 -y && conda activate npspy_env
```

2. install pytorch according to [pytorch.org](https://pytorch.org/get-started/locally/) and your system environment, such as (v2.7.0):
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. install npspy:
```
pip install git+https://gitlab.genomics.cn/panhailin/npspy.git
```


