from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Heart problems classifier",
    author="ngc436",
    entry_points={
        "console_scripts": [
            "ml_example_train = classification.full_pipeline:run_full_pipeline"
        ]
    },
    install_requires=required,
    license="MIT",
)
