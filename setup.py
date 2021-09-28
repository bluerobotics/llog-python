#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='llog',
    version='0.0.1',
    description='little log',
    author='Blue Robotics',
    url='https://github.com/bluerobotics/llog-python',
    packages=['llog'],
    package_data={'llog': ['br.png']},
    install_requires=[
        'jinja2',
        'matplotlib',
        'numpy',
        'pandas',
        'pypdf2',
    ],
)
