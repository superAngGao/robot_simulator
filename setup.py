from setuptools import find_packages, setup

setup(
    name="robot_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
    extras_require={
        "mesh": ["trimesh>=3.21"],
    },
    python_requires=">=3.10",
)
