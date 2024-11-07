from setuptools import setup

setup(
    name="gym_maze",
    version="0.0",
    url="https://github.com/JingtianYan/MAPF_visualize.git",
    author="Matthew T.K. Chan",
    license="MIT",
    packages=["gym_maze", "gym_maze.envs"],
    #   package_data = {
    #       "gym_mpf.envs": ["mapf_samples/*.npy"]
    #   },
    install_requires=[
        "gym==0.26.2",
        "pygame==2.4.0",
        "numpy==1.24.3",
        "fire==0.5.0",
        "matplotlib",
        "logdir",
        "opencv-python==4.8.0.76",
        "imageio==2.31.3",
        "moviepy==1.0.3",
        "Pillow==10.0.1",
        "matplotlib==3.7.3",
    ])
