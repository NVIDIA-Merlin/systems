# Merlin Systems Example Notebook

These Jupyter notebooks demonstrate how to use Merlin Systems to deploy models to Triton Inference Server.

## Running the Example Notebooks

Docker containers are available from the NVIDIA GPU Cloud.
We use the latest stable version of the [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow/tags) container to run the example notebooks. To run the example notebooks using Docker containers, perform the following steps:

1. If you haven't already created a Docker volume to share models and datasets
   between containers, create the volume by running the following command:

   ```shell
   docker volume create merlin-examples
   ```

   For the ranking models example, note that the saved `dlrm` model that was created with Merlin Models, the NVTabular workflow, and processed synthetic parquet files should be stored in the `merlin-examples` folder so that they can be mounted to the container for performing inference with Merlin Systems.

1. Pull and start the container by running the following command:

   ```shell
   docker run --gpus all --rm -it \
     -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
     -v merlin-examples:/workspace/data \
     nvcr.io/nvidia/merlin/merlin-tensorflow:nightly /bin/bash
   ```

   > In production, instead of using the `nightly` tag, specify a release tag.
   > You can find the release tags and more information on the [merlin-tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow) container page.

   The container opens a shell when the run command execution is completed.
   Your shell prompt should look similar to the following example:

   ```shell
   root@2efa5b50b909:
   ```

1. Start the JupyterLab server by running the following command:

   ```shell
   jupyter-lab --allow-root --ip='0.0.0.0'
   ```

   View the messages in your terminal to identify the URL for JupyterLab.
   The messages in your terminal show similar lines to the following example:

   ```shell
   Or copy and paste one of these URLs:
   http://2efa5b50b909:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   or http://127.0.0.1:8888/lab?token=9b537d1fda9e4e9cadc673ba2a472e247deee69a6229ff8d
   ```

1. Open a browser and use the `127.0.0.1` URL provided in the messages by JupyterLab.

1. After you log in to JupyterLab, navigate to the `/systems/examples` directory to try out the example notebooks.
