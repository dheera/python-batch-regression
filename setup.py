from setuptools import setup, find_packages

setup(
    name="batch_regression",
    version="0.1.0",
    description="A PyTorch-based library for batch linear regression on GPUs.",
    author="Dheera Venkatraman",
    author_email="dheera@dheera.net",
    packages=find_packages(),
    install_requires=[
        "torch",  # Ensure that users have PyTorch installed.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
