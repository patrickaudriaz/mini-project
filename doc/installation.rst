.. vim: set fileencoding=utf-8 :

.. _activities_installation:


==============
 Installation
==============

Using Github
------------

This package depends on multiple packages to run properly. Please install the
version of these packages indicated in requirements.txt before trying to run 
the code examples. We recommend the miniconda_ package manager for this work.

To download a copy of this package, clone it from its repository in GitHub:

.. code:: sh

   $ git clone https://github.com/patrickaudriaz/mini-project.git activities
   $ cd activities
   $ conda create --name activities --file requirements.txt
   $ conda activate activities
   (activities) $ #you are ready to run baselines!

Using PyPI
----------

If you want to install the already packaged code, please follow this procedure:

.. code:: sh

   $ pip install rrgp
   $ #you are ready to run baselines!

.. include:: links.rst
