from io import open

from setuptools import find_packages, setup

with open('okgraph/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

INSTALL_REQUIRES = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    # TODO fix this accordingly to what stated at:
    #  <https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/>
    #  `install_requires` and `requirements.txt` are different and have
    #  different purpose
    for line in f.readlines():
        if not line.startswith('#') and not line.startswith('-'):
            INSTALL_REQUIRES.append(line)

setup(
    name='okgraph',
    version=version,
    description='OKgraph is a python library for unstructured natural language'
                ' understanding',
    long_description=readme,
    author='Maurizio Atzori',
    author_email='atzori@unica.it',
    maintainer='Maurizio Atzori',
    maintainer_email='atzori@unica.it',
    url='https://bitbucket.org/semanticweb/okgraph/src',
    license='gpl-3.0',

    keywords=[
        '',
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    python_requires='>=3.7,<3.10',
    install_requires=INSTALL_REQUIRES,
    tests_require=['coverage', 'pytest'],
    packages=find_packages(),
    include_package_data=True,
)
