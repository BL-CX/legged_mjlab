"""Install the Legged MJLab HIM and HIM-AMP training package."""

from setuptools import find_packages, setup


setup(
    name="legged_mjlab",
    version="0.1.0",
    description="Legged robot learning with MJLab",
    python_requires=">=3.10,<3.14",
    packages=find_packages(include=("src", "src.*")),
    include_package_data=False,
    package_data={
        "src.assets": ["motions/mocap_motions_go2/*.txt"],
        "src.assets.robots.unitree_go2": ["xmls/*.xml", "xmls/assets/*.obj"],
    },
    install_requires=[
        "mjlab==1.2.0",
        "mujoco==3.5.0",
        "mujoco-warp==3.5.0",
        "warp-lang==1.12.1",
        "numpy",
        # MJLab 1.2 imports scipy.interpolate but omits SciPy from its metadata.
        "scipy",
        "torch>=2.7.0",
        "PyYAML",
        "prettytable",
        "tyro>=1.0.1",
        "tensorboard>=2.20.0",
    ],
)
