import contextlib
import logging
import os
import signal
import socket
import subprocess
import time
from distutils.spawn import find_executable

import tritonclient
import tritonclient.grpc as grpcclient

from merlin.systems import triton

LOG = logging.getLogger("merlin-systems")

TRITON_SERVER_PATH = find_executable("tritonserver")


@contextlib.contextmanager
def run_triton_server(
    model_repository: str,
    *,
    grpc_host: str = "localhost",
    grpc_port: int = 8001,
    backend_config: str = "tensorflow,version=2",
):
    """This function starts up a Triton server instance and returns a client to it.

    Parameters
    ----------
    model_repository : string
        The path to the model repository directory.
    grpc_host : string
        The host address for the triton gRPC server to bind to.
        Default is localhost.
    grpc_port : int
        The port for the triton gRPC server to listen on for requests.
        Default is 8001.
    backend_config : string
        A backend-specific configuration.
        Following the pattern <backend_name>,<setting>=<value>.
        Where <backend_name> is the name of the backend, such as 'tensorflow'

    Yields
    ------
    client: tritonclient.InferenceServerClient
        The client connected to the Triton server.

    """
    if grpc_port == 0 or grpc_port is None:
        grpc_port = _get_random_free_port()
    grpc_url = f"{grpc_host}:{grpc_port}"

    try:
        with grpcclient.InferenceServerClient(grpc_url) as client:
            if client.is_server_ready():
                raise RuntimeError(f"Another tritonserver is already running on {grpc_url}")
    except tritonclient.utils.InferenceServerException:
        pass

    cmdline = [
        TRITON_SERVER_PATH,
        "--model-repository",
        model_repository,
        f"--backend-config={backend_config}",
        f"--grpc-port={grpc_port}",
        f"--grpc-address={grpc_host}",
        "--cuda-memory-pool-byte-size=0:536870912",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with subprocess.Popen(cmdline, env=env) as process:
        try:
            with grpcclient.InferenceServerClient(grpc_url) as client:
                # wait until server is ready
                time_ranges = [60, 120, 300]
                for seconds in time_ranges:
                    for _ in range(seconds):
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
                    LOG.error("Failed to start tritonserver in %s seconds", seconds)

                raise RuntimeError("Timed out waiting for tritonserver to become ready")
        finally:
            # signal triton to shutdown
            process.send_signal(signal.SIGINT)


def _get_random_free_port():
    """Return a random free port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_ensemble_on_tritonserver(
    tmpdir,
    schema,
    df,
    output_columns,
    model_name,
):
    """Starts up a Triton server instance, loads up the ensemble model,
    prepares the inference request and returns the unparsed inference
    response.

    Parameters
    ----------
    tmpdir : string
        Directory from which to load ensemble model.
    schema : Schema
        Schema of the inputs in the dataframe
    df : dataframe-like
        A dataframe type object that contains rows of inputs to predict on.
    output_columns : [string]
        List of columns that will be predicted.
    model_name : string
        The name of the ensemble model to use.

    Returns
    -------
    results : dict
        the results of the prediction, parsed by output column name.
    """
    response = None
    with run_triton_server(tmpdir) as client:
        response = send_triton_request(
            schema, df, output_columns, client=client, triton_model=model_name
        )

    return response


def send_triton_request(
    schema,
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
    schema : Schema
        The schema of the inputs in the dataframe
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

    inputs = triton.convert_df_to_triton_input(schema, df, grpcclient.InferInput)

    outputs = [grpcclient.InferRequestedOutput(col) for col in outputs_list]
    with client:
        response = client.infer(triton_model, inputs, request_id=request_id, outputs=outputs)

    results = {}
    for col in outputs_list:
        results[col] = response.as_numpy(col)

    return results
