Scikit-Shapes: Shape analysis in Python
=======================================

Welcome to the documentation of Scikit-Shapes, an open source library for shape analysis in Python.
Our source code is available on `GitHub <https://github.com/scikit-shapes/scikit-shapes>`__.

To get started, check out the installation instructions and have a look at the examples.

.. warning::

   This library is still in very active development.
   We expect to release a first usable version in September 2025.


Licensing and citations
-----------------------

This library is licensed under the permissive `MIT license <https://en.wikipedia.org/wiki/MIT_License>`__,
which is fully compatible with both **academic** and **commercial** applications.

Scikit-Shapes provides a user-friendly interface for a collection of research papers,
whose authors must be credited.
If you use this library in an academic context, **please run** the following piece of
code to print a **list of the references that have been used in your experiment**:

.. code-block:: python

   import skshapes as sks

   # Run your code here
   ...

   # At the end of your script, print the relevant references.
   # This list is updated dynamically by calls to the library functions
   # and may be displayed in both bibtex and plain text (APA) formats:
   print(sks.bibliography(style="bibtex"))
   print(sks.bibliography(style="APA"))


Acknowledgements
-----------------

Authors:

- `Jean Feydy <https://www.jeanfeydy.com/>`__ (2023-), project leader.
- `Louis Pujol <https://www.linkedin.com/in/louis-pujol-722997225/>`__ (2023-24), core developer.

This library is maintained in the
`HeKA team <https://team.inria.fr/heka/fr/>`__,
a joint research group between
`INRIA <https://www.inria.fr/en>`__,
`INSERM <https://www.inserm.fr/en>`__
and the
`Université Paris Cité <https://u-paris.fr/en/>`__.
It is funded by
`INRIA <https://inria.fr/en>`__, the
`PRAIRIE-PSAI <https://prairie-institute.fr/>`__
institute,
and is distributed under the permissive
`MIT license <https://github.com/scikit-shapes/scikit-shapes/blob/main/LICENSE>`__.


Content
-------

.. toctree::
   :maxdepth: 1

   motivation
   installation
   user_guide/index
   auto_examples/index
   stubs/skshapes
   explanation/index
   contributing
