from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup()
d['packages'] = ['promp_ros']
d['package_dir'] = {'': 'src'}
setup(**d)