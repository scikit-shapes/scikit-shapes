from setuptools import setup


dependencies = [
    'numpy',
    'torch',
    'pykeops',
    'geomloss',
    'jaxtyping',
    'beartype',
    'pyvista',
    'vedo',
    'fast-simplification'
]

submodules = [
    'applications',
    'convolutions',
    'data',
    'decimation',
    'features',
    'loss',
    'morphing',
    'optimization',
    'tasks',
]

setup(name='Scikit-Shapes',
      version='1.0',
      description='Shape Analysis in Python',
      author='Scikit-Shapes Developers',
      author_email='skshapes@gmail.com',
      url='',
      install_requires=dependencies,
      packages=['skshapes'] + ['skshapes.' + submodule for submodule in submodules],
          
     )