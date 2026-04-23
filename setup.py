from setuptools import find_packages, setup

mesh_deps = ["trimesh>=3.21"]
rl_deps = ["gymnasium>=0.29", "torch"]

setup(
    name="robot_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
    extras_require={
        "mesh": mesh_deps,
        "rl": rl_deps,
        "rerun": ["rerun-sdk>=0.16"],
        "dev": [
            "pytest>=7.4",
            "pytest-cov",
            "scipy>=1.11",
            "pillow>=9.0",
            *mesh_deps,
            *rl_deps,
        ],
    },
    python_requires=">=3.10",
)
