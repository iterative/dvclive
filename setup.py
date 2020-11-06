from setuptools import find_packages, setup

install_requires = [
    "pylint==2.5.3",
]
setup(
    name="dvclive", packages=find_packages(), install_requires=install_requires
)
