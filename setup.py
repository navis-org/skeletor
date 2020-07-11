try:
    from setuptools import setup
except BaseException:
    print('Setuptools not found - falling back to distutils')
    from distutils.core import setup

import re

VERSIONFILE = "skeletonizer/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setup(
    name='skeletor',
    version=verstr,
    packages=['skeletor', ],
    license='GNU GPL V3',
    description='Python 3 implementation of Skeleton Extraction by Mesh contraction algorithm',
    long_description=open('README.md').read(),
    url='https://github.com/schlegelp/skeletonizer',
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='mesh skeletonizer mesh contraction skeleton extraction',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=requirements,
    python_requires='>=3.3',
    zip_safe=False
)
