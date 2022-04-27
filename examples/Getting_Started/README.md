# NVIDIA Merlin Systems Example Notebooks
This jupyter notebook example is prepared to demonstrate how to deploy a ranking model trained and saved with Merlin Models to Triton Inference Server (TIS) using Merlin Systems library.
## Structure
The example notebook is structured as follows:
- load the saved DLRM model
- load the saved NVTabular workflow
- perform Inference with the TIS using Merlin Systems `Ensemble` class.
## Running the Example Notebooks
To run Merlin Systems examples, we have docker containers available on http://ngc.nvidia.com/catalog/containers/ with pre-installed versions.
- merlin-tensorflow-inference contains NVTabular with TensorFlow and Triton Inference support
To run the example notebooks using Docker containers, do the following:
1. Pull the container by running the following command:
   ```
   docker run -it --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8888:8888 -v <path to your data and files>:/workspace/data/ --ipc=host <docker container> /bin/bash
   ```
**Note:** To execute the `Serving-Ranking-Models-With-Merlin-Systems.ipynb` notebook, you need to mount the saved DLRM model and NVTabular workflow to the inference container with `-v` flag as demonstrated in the command above. 
The container will open a shell when the run command execution is completed. 
   
2. Install Tensorflow
    ```
    pip install tensorflow-gpu
    ```
3. Install jupyter-lab by running the following command:
   ```
   pip install jupyterlab
   ```
   
   For more information, see [Installation Guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
4. Start the jupyter-lab server by running the following command:
   ```
   jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
   ```
5. Open any browser to access the jupyter-lab server using <MachineIP>:8888.
6. Once in the server, navigate to the ```/systems/examples``` directory and execute the example notebooks.
