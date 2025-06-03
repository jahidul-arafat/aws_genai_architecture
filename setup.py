from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aws-genai-architecture-viz",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Interactive visualization of AWS GenAI LLM training architecture with detailed communication monitoring",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aws-genai-architecture-viz",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "flask>=2.0.0",
        "flask-socketio>=5.0.0",
        "python-socketio>=5.0.0",
        "eventlet>=0.33.0",
        "rich>=12.0.0",
        "colorama>=0.4.4",
        "pydantic>=1.8.0",
        "typing-extensions>=4.0.0",
        "numpy>=1.21.0",
        "datetime",
        "json",
        "threading",
        "queue",
        "time",
        "uuid",
        "dataclasses",
        "enum",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aws-genai-viz=aws_genai_architecture.cli:main",
        ],
    },
    package_data={
        "aws_genai_architecture": [
            "static/*.html",
            "static/*.css",
            "static/*.js",
            "templates/*.html",
        ],
    },
    include_package_data=True,
)