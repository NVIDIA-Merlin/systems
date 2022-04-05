Merlin Systems Example Notebooks
================================

Refer to the following page for an example notebook that
demonstrates how Merlin Systems simplifies building an
end-to-end recommender system.

.. toctree::
   :maxdepth: 1

   Serving Ranking Models With Merlin Systems <Getting_Started/Serving-Ranking-Models-With-Merlin-Systems.ipynb>


Running the Example Notebooks
-----------------------------

Docker containers are available from the NVIDIA GPU Cloud.
Access the catalog of containers at http://ngc.nvidia.com/catalog/containers.

All the training containers include NVTabular for feature engineering.
Choose the container that provides the modeling library that you prefer:

- Merlin Training (modeling with HugeCTR)
- Merlin TensorFlow Training
- Merlin PyTorch Training

All the inference containers include NVTabular so that workflows can be read and
they all include Triton Inference Server for deploying models to production and
serving recommendations.
Choose the container that can read models from the modeling library that you used:

- Merlin Inference (HugeCTR models)
- Merlin TensorFlow Inference
- Merlin PyTorch Inference

To run the example notebooks using Docker containers, perform the following steps:

1. Pull and start the container by running the following command:

   .. code-block:: shell

      docker run --gpus all --rm -it \
        -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
        <docker container> /bin/bash

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   .. code-block:: shell

      root@2efa5b50b909:

2. Install JupyterLab with ``pip`` by running the following command:

   .. code-block:: shell

      pip install jupyterlab

   For more information, see the JupyterLab `Installation Guide <https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html>`_.

3. Start the JupyterLab server by running the following command:

   .. code-block:: shell

      jupyter-lab --allow-root --ip='0.0.0.0'

   View the messages in your terminal to identify the URL for JupyterLab.
   The messages in your terminal show similar lines to the following example:

   .. code-block:: shell

      Or copy and paste one of these URLs:
         http://2efa5b50b909:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
      or http://127.0.0.1:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d

4. Open a browser and use the ``127.0.0.1`` URL provided in the messages by JupyterLab.

5. After you log in to JupyterLab, navigate to the ``/merlin/examples`` directory to try out the example notebooks.
