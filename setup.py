from setuptools import setup

setup(
    name='pytorch-cnn-visualization',
    version='0.0.1',
    packages=['cnn_visualization'],
    license='MIT',
    description='Several CNN visualization implemented in Pytorch',
    long_description=open('README.md').read(),
    install_requires=['torch==0.3.1', 'torchvision==0.2.0', 'scipy', 'matplotlib'],
    url='https://github.com/limyunkai19/pytorch-cnn-visualization',
    author='Lim Yun Kai',
    author_email='yunkai96@hotmail.com'
)
