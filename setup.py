from setuptools import setup

setup(
    name="predict_alg",
    version="0.1.0",
    description="A simple Python package",
    author="Your Name",
    author_email="yevtefeevah@gmail.com",
    packages=["a.py", "alg_eda", "eda"],
    install_requires=["numpy", "pandas", "scikit-learn", "tensorflow", "keras", "matplotlib", "seaborn"],
)
