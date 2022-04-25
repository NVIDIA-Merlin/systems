import contextlib
import os
import signal
import subprocess
import time
from distutils.spawn import find_executable

import tritonclient
import tritonclient.grpc as grpcclient

import merlin.systems.triton as triton

TRITON_SERVER_PATH = find_executable("tritonserver")


@contextlib.contextmanager
def run_triton_server(modelpath):
    """This function starts up a Triton server instance and returns a client to it.

    Parameters
    ----------
    modelpath : string
        The path to the model to load.

    Yields
    ------
    client: tritonclient.InferenceServerClient
        The client connected to the Triton server.

    """
    cmdline = [
        TRITON_SERVER_PATH,
        "--model-repository",
        modelpath,
        "--backend-config=tensorflow,version=2",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with subprocess.Popen(cmdline, env=env) as process:
        try:
            with grpcclient.InferenceServerClient("localhost:8001") as client:
                # wait until server is ready
                for _ in range(60):
                    if process.poll() is not None:
                        retcode = process.returncode
                        raise RuntimeError(f"Tritonserver failed to start (ret={retcode})")

                    try:
                        ready = client.is_server_ready()
                    except tritonclient.utils.InferenceServerException:
                        ready = False

                    if ready:
                        yield client
                        return

                    time.sleep(1)

                raise RuntimeError("Timed out waiting for tritonserver to become ready")
        finally:
            # signal triton to shutdown
            process.send_signal(signal.SIGINT)


def run_ensemble_on_tritonserver(
    tmpdir,
    output_columns,
    df,
    model_name,
):
    """Starts up a Triton server instance, loads up the ensemble model,
    prepares the inference request and returns the unparsed inference
    response.

    Parameters
    ----------
    tmpdir : string
        Directory from which to load ensemble model.
    output_columns : [string]
        List of columns that will be predicted.
    df : dataframe-like
        A dataframe type object that contains rows of inputs to predict on.
    model_name : string
        The name of the ensemble model to use.

    Returns
    -------
    results : dict
        the results of the prediction, parsed by output column name.
    """
    response = None
    with run_triton_server(tmpdir) as client:
        response = send_triton_request(df, output_columns, client=client, triton_model=model_name)

    return response


def send_triton_request(
    df,
    outputs_list,
    client=None,
    endpoint="localhost:8001",
    request_id="1",
    triton_model="ensemble_model",
):
    """This function checks if the triton server is available and sends a request to the Triton
    server that has already been started.

    Parameters
    ----------
    df : dataframe
        The dataframe with the inputs to predict on.
    outputs_list : [string]
        A list of the output columns from the prediction.
    endpoint : str, optional
        The connection endpoint of the tritonserver instance, by default "localhost:8001"
    request_id : str, optional
        The id of the inference request, by default "1"
    triton_model : str, optional
        Name of the model to run inputs through, by default "ensemble_model"

    Returns
    -------
    results : dict
        A dictionary of parsed output column results from the prediction.

    """
    if not client:
        try:
            client = grpcclient.InferenceServerClient(url=endpoint)
        except Exception as e:
            raise e

    if not client.is_server_live():
        raise ValueError("Client could not establish commuincation with Triton Inference Server.")

    inputs = triton.convert_df_to_triton_input(df.columns, df, grpcclient.InferInput)

    outputs = [grpcclient.InferRequestedOutput(col) for col in outputs_list]
    with client:
        response = client.infer(triton_model, inputs, request_id=request_id, outputs=outputs)

    results = {}
    for col in outputs_list:
        results[col] = response.as_numpy(col)

    return results
