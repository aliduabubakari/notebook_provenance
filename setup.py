"""
Setup script for notebook_provenance package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read version
version = "0.2.0"

setup(
    name="notebook-provenance",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated provenance extraction and visualization for computational notebooks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/notebook-provenance",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.6",
        "matplotlib>=3.4",
        "numpy>=1.20",
    ],
    extras_require={
        "llm": [
            "openai>=1.0",
        ],
        "notebook": [
            "nbformat>=5.0",
        ],
        "neo4j": [
            "neo4j>=5.0",
        ],
        "syntax": [
            "pygments>=2.10",
        ],
        "all": [
            "openai>=1.0",
            "nbformat>=5.0",
            "neo4j>=5.0",
            "pygments>=2.10",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.10",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "notebook-provenance=notebook_provenance.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)