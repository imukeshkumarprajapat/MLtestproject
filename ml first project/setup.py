from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """This function will return the list of requirements"""
    requirements = []
    with open(file_path ) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', "") for req in requirements]
    return requirements

setup(
    name="ml_first_project",
    version="0.1.0",
    author="mukesh kumar",
    author_email="imukeshkumarprajapt@gamil.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description="A first machine learning project setup",
    license="MIT",
    )