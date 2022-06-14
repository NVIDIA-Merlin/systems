# Merlin Systems Example Notebook

This Jupyter notebook example demonstrates how to deploy a ranking model to Triton Inference Server.
As a prerequisite, the model must be trained and saved with Merlin Models.
See the [Exporting Ranking Models](https://github.com/NVIDIA-Merlin/models/blob/main/examples/04-Exporting-ranking-models.ipynb)
file or browse the [examples](https://github.com/NVIDIA-Merlin/models/tree/main/examples) directory of the Merlin Models repository.

- [Serving Ranking Models With Merlin Systems](Serving-Ranking-Models-With-Merlin-Systems.ipynb)

## Running the Example Notebook

Docker containers are available from the NVIDIA GPU Cloud.
Access the catalog of containers at <http://ngc.nvidia.com/catalog/containers>.

Use the following container to run the example notebook:

- Merlin TensorFlow Inference

To run the example notebooks using Docker containers, perform the following steps:

1. If you haven't already created a Docker volume to share models and datasets
   between containers, create the volume by running the following command:

   ```shell
   docker volume create merlin-examples
   ```
Note that the saved `dlrm` model, NVT `workflow` and processed synthetic parquet files should be stored in the `merlin-examples` folder so that they can be mounted to the inference container.


1. Pull and start the container by running the following command:

   ```shell
   docker run --gpus all --rm -it \
     -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host \
     -v merlin-examples:/workspace/data \
     <docker container> /bin/bash
   ```

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
