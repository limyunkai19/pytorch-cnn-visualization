from setuptools import setup

setup(
    name='pytorch-cnn-visualization',
    version='0.2.3',
    packages=['pytorch_cnn_visualization'],
    package_dir={'pytorch_cnn_visualization': 'pytorch_cnn_visualization'},
    package_data={'pytorch_cnn_visualization': ['data/*.json', 'misc_scripts/*.py']},
    license='MIT',
    description='Several CNN visualization implemented in Pytorch',
    long_description=open('README.md').read(),
    install_requires=['torch==0.3.1', 'torchvision==0.2.0', 'scipy', 'matplotlib'],
    url='https://github.com/limyunkai19/pytorch-cnn-visualization',
    author='Lim Yun Kai',
    author_email='yunkai96@hotmail.com'
)
