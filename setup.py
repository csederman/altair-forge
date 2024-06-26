import codecs

from setuptools import find_packages
from setuptools import setup


requirements = [
    "altair >= 5.0.1",
    "numpy >= 1.21",
    "pandas >= 2.0.3",
    "scikit-learn",
    "scipy",
]

extras_require = {"demos": []}

with open("./test-requirements.txt") as test_reqs_txt:
    test_requirements = [line for line in test_reqs_txt]


long_description = ""
with codecs.open("./README.md", encoding="utf-8") as readme_md:
    long_description = readme_md.read()

setup(
    name="altair_forge",
    use_scm_version={"write_to": "altair_forge/_version.py"},
    description="Plotting utilities and extended functionality for the Altair data visualization library..",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csederman/altair-forge",
    packages=find_packages(exclude=["tests.*", "tests"]),
    setup_requires=["setuptools_scm"],
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require=extras_require,
    python_requires=">=3.8",
    zip_safe=False,
    test_suite="tests",
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    maintainer="Casey Sederman",
    maintainer_email="crs7240@gmail.com",
)
