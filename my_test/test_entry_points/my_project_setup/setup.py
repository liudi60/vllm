# setup.py

from setuptools import setup, find_packages

setup(
    name="myproject",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mycmd = mypackage.cli:main",   # ← 关键！
        ],
    },
    python_requires=">=3.6",
)