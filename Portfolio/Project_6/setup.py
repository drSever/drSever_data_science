from setuptools import setup, find_packages


REQUIRED_PACKAGES = [
    'appnope==0.1.4',
    'asttokens==2.4.1',
    'Brotli==1.0.9',
    'certifi==2024.7.4',
    'charset-normalizer==3.3.2',
    'colorama==0.4.6',
    'comm==0.2.2',
    'contourpy==1.2.0',
    'cycler==0.11.0',
    'debugpy==1.8.5',
    'decorator==5.1.1',
    'exceptiongroup==1.2.2',
    'executing==2.0.1',
    'filelock==3.13.1',
    'fonttools==4.51.0',
    'gmpy2==2.1.2',
    'idna==3.7',
    'importlib_metadata==8.2.0',
    'ipykernel==6.29.5',
    'ipython==8.26.0',
    'jedi==0.19.1',
    'Jinja2==3.1.4',
    'jupyter_client==8.6.2',
    'jupyter_core==5.7.2',
    'kiwisolver==1.4.4',
    'MarkupSafe==2.1.3',
    'matplotlib==3.9.1',
    'matplotlib-inline==0.1.7',
    'mpmath==1.3.0',
    'nest_asyncio==1.6.0',
    'networkx==3.3',
    'numpy==1.26.0',
    'opencv-python==4.10.0',
    'opencv-python-headless==4.10.0',
    'packaging==24.1',
    'parso==0.8.4',
    'path==17.0.0',
    'pexpect==4.9.0',
    'pickleshare==0.7.5',
    'pillow==10.4.0',
    'pip==24.0',
    'platformdirs==4.2.2',
    'prompt_toolkit==3.0.47',
    'psutil==6.0.0',
    'ptyprocess==0.7.0',
    'pure_eval==0.2.3',
    'Pygments==2.18.0',
    'pyparsing==3.0.9',
    'PySocks==1.7.1',
    'python-dateutil==2.9.0.post0',
    'PyYAML==6.0.1',
    'pyzmq==26.1.0',
    'requests==2.32.3',
    'setuptools==72.1.0',
    'six==1.16.0',
    'stack-data==0.6.2',
    'sympy==1.12',
    'torch==2.4.0',
    'torchvision==0.19.0',
    'tornado==6.4.1',
    'tqdm==4.66.5',
    'traitlets==5.14.3',
    'typing_extensions==4.11.0',
    'unicodedata2==15.1.0',
    'urllib3==2.2.2',
    'wcwidth==0.2.13',
    'wheel==0.43.0',
    'zipp==3.19.2'
]


setup(
    name='myproject',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages()
)
