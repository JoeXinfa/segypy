DESCRIPTION = """\
A python module to read and write SEG-Y formatted files.
"""

from setuptools import setup, find_packages

setup(
    name = 'segypy',
    packages = find_packages(),
    include_package_data = True,
    version = '0.5.0',
    description = 'Segy read and write',
    author = 'Joseph Zhu',
    author_email = 'zhuu@chevron.com',
    url = 'http://136.171.178.114/zhuu/segypy.git',
    license = 'LGPL',
    long_description = DESCRIPTION,
)
