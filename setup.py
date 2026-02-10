from setuptools import setup, find_packages

setup(
    name="nova-winograd",
    version="1.0.0",
    description="NOVA Winograd F(6,3) — Numerically stable large-tile Winograd convolution for AMD GPUs",
    author="Jayant Lohia",
    license="Proprietary",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
    ],
    extras_require={
        "bench": ["diffusers", "transformers", "torchvision"],
    },
)
