.. vim: set fileencoding=utf-8 :

.. _activities:

============================================================
 Reproducible Activities Recognition From Smartphone Sensors
============================================================

This package contains code to run random forest or svm classification
on a dataset_ created for the study of activities recognition from smartphone
sensors.

This package performs multiple tasks sequentially:

1. Download the dataset (if not already available)
2. Train a model (random forest or svm) to recognize activities
3. Predict the activities of the testing set
4. Evaluate the performances and save the analysis in the output folder

To run the script using the default configuration just follow the 
:ref:`activities_installation` and use the following command::

   python run.py

It will train an svm model using predefined parameters and save the analysis
in a *results* folder.

Documentation
-------------

.. toctree::

   installation
   guide
   troubleshooting
   api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
