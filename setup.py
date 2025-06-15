from setuptools import setup, find_packages

# Read the content of the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the content of the requirements.txt file
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="jola",
    version="0.0.1",
    description="JoLA: Joint Localization and Activation Editing for Low-Resource Fine-Tuning",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/wenlai-lavine/jola",
    author="Wen Lai",
    author_email="wen.lai@tum.de",
    license="Apache License 2.0",
    packages=find_packages(include=['jola', 'jola.*']),
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)