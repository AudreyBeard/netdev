from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='netdev',
      version='0.0.1',
      description='A collection of tools to make neural network development faster, easier, and more readable',
      long_description=long_description,
      author='Joshua Beard',
      author_email='joshuabeard92@gmail.com',
      packages=find_packages(),
      install_requires=['numpy',
                        'torch',
                        'ubelt',
                        'torchvision'],
      url='https://github.com/JoshuaBeard/netdev',
      changelog={'0.0.0': 'Beta',
                 '0.0.1': 'New training network system'
                 }
      )
