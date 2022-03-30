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
    inputs = triton.convert_df_to_triton_input(df.columns, df)
    outputs = [grpcclient.InferRequestedOutput(col) for col in output_columns]
    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs, outputs=outputs)

    return response
