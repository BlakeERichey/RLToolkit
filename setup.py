from setuptools import setup, find_packages

setup(name='rltoolkit',
      version='0.0.1',
      packages=find_packages(),
      description='RL Toolkit',
      author = 'Blake Richey',
      author_email='blake.e.richey@gmail.com',
      install_requires=['keras>=2.3.1', 'gym>=0.16.0', 'tensorflow>=1.14', 'matplotlib>=3.2.1'],
      )