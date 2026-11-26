from setuptools import find_packages, setup


setup(
    packages=(
        find_packages(where="src")
        + find_packages(where=".", include=["otk_api", "otk_api.*"])
    ),
    package_dir={"": "src", "otk_api": "otk_api"},
    package_data={
        "otk_preprocess.annotation": ["*.csv"],
        "otk.data": ["*.csv", "*.csv.gz", "*.json"],
        "otk_api": [
            "README.md",
            "config.yml",
            "static/*.png",
            "static/*.md",
            "models/*/*.pkl",
            "models/*/*.pth",
            "models/*/*.yml",
            "models/*/*.csv",
        ],
    },
    exclude_package_data={
        "otk_api": ["models/tabpfn/best_model.pkl", "*.pdf", "models/*.png", "models/*.md"]
    },
)
