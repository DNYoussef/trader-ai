"""
Setup configuration for ML Intelligence System
Production-ready installation with all dependencies
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove version pinning comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                requirements.append(line)
    return requirements

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), '..', '..', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "ML Intelligence System for Trading AI"

setup(
    name="trader-ai-intelligence",
    version="1.0.0",
    author="AI Trading Intelligence Team",
    author_email="intelligence@trader-ai.com",
    description="Production-ready ML intelligence system for financial trading",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/trader-ai/intelligence",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "gpu": [
            "torch[cuda]>=2.1.0",
            "tensorflow[gpu]>=2.15.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "ipykernel>=6.26.0",
            "nbconvert>=7.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trader-ai-train=intelligence.training.cli:main",
            "trader-ai-predict=intelligence.prediction.cli:main",
            "trader-ai-evaluate=intelligence.evaluation.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "intelligence": [
            "config/*.yaml",
            "data/*.csv",
            "models/*.pkl",
            "templates/*.json",
        ],
    },
    zip_safe=False,
)