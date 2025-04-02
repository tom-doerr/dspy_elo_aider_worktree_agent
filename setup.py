from setuptools import setup, find_packages

with open("dspy_elo/__init__.py", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
            break

setup(
    name="dspy_elo",
    version=version,
    packages=find_packages(),
    install_requires=[
        'dspy',
        'pandas',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="ELO rating system for LLM output comparison",
    license="MIT",
    url="https://github.com/yourusername/dspy-elo",
    python_requires=">=3.8",
)
