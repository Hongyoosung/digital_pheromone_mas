from setuptools import setup, find_packages

setup(
    name="digital_pheromone_mas",
    version="1.0.0",
    author="Research Team",
    description="4D Digital Pheromone Multi-Agent System with Distributed Attention Networks",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)