CryoPARES Documentation
=======================

**CryoPARES** is a software package for assigning poses to 2D cryo-electron microscopy (cryo-EM) particle images using supervised deep learning.

The key idea is to train a neural network on a high-quality reference reconstruction, and then reuse this trained model to rapidly estimate particle poses in other, similar datasets.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   training_guide
   configuration_guide
   troubleshooting
   cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   GitHub Repository <https://github.com/rsanchezgarc/cryoPARES>
   Paper <https://www.biorxiv.org/content/10.1101/2025.03.04.641536v2>

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   conda create -n cryopares python=3.12
   conda activate cryopares
   pip install git+https://github.com/rsanchezgarc/cryoPARES.git

Training
~~~~~~~~

.. code-block:: bash

   ulimit -n 65536  # Increase file descriptor limit

   cryopares_train \
       --symmetry C1 \
       --particles_star_fname /path/to/aligned_particles.star \
       --train_save_dir /path/to/output \
       --n_epochs 20

Inference
~~~~~~~~~

.. code-block:: bash

   cryopares_infer \
       --particles_star_fname /path/to/new_particles.star \
       --checkpoint_dir /path/to/output/version_0 \
       --results_dir /path/to/results

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
