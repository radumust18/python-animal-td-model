from setuptools import setup

setup(
    name='am_tdm',
    version='0.1.0',
    description='Python wrappers for the Animal Model and the Test-Day Model using the BLUPF90 suite of programs. '
                'Supports pedigree renumbering and variance components estimation using REML and Gibbs sampling. '
                'Information such as EBVs, permanent environmental effects, reliabilities, heritabilities of traits or '
                'deregressed proofs are computed',
    url='https://github.com/radumust18/python-animal-td-model',
    author='Radu-Ioan Mustatea',
    author_email='radu.mustatea18@gmail.com',
    license='MIT License',
    packages=['am_tdm'],
    install_requires=['numpy', 'scipy', 'pandas'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
)
