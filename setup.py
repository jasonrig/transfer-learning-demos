import importlib.util

from setuptools import setup, find_packages

setup(
    name='transfer-learning-demos',
    version='0.1',
    packages=find_packages(),
    url='',
    license='',
    author='Jason Rigby',
    author_email='',
    description='',
    install_requires=[
        'appdirs>=1.4.0,<1.5',
        'Markdown>=2.6.0,<2.7.0',
        'requests>=2.18.0,<2.19.0',
        'beautifulsoup4>=4.6.0,<4.7.0'
    ]
)

# Check if TensorFlow is available
tf = importlib.util.find_spec("tensorflow")
if tf is None:
    print("Warning: A working TensorFlow installation is required but none could be found.")
    print("Please run `pip install tensorflow` or `pip install tensorflow-gpu` before using this package.")
