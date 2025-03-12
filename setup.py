from setuptools import setup, find_packages

setup(
    name="pokerdata",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "sqlalchemy>=1.4.0",
        "pytest>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "process-poker-data=process_data:main",
            "visualize-poker-data=visualize_data:main",
            "analyze-poker-range=analyze_range:main",
        ],
    },
    author="PokerData Team",
    author_email="info@pokerdata.com",
    description="A package for processing poker hand history files",
    keywords="poker, data, analysis",
    python_requires=">=3.8",
)
