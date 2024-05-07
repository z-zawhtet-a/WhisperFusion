import argparse
import ctypes
import multiprocessing
from multiprocessing import Manager, Queue, Value

from whisper_live.trt_server import TranscriptionServer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--whisper_tensorrt_path",
        type=str,
        default="/root/TensorRT-LLM/examples/whisper/whisper_small_en",
        help="Whisper TensorRT model path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if not args.whisper_tensorrt_path:
        raise ValueError("Please provide whisper_tensorrt_path to run the pipeline.")

    multiprocessing.set_start_method("spawn")

    lock = multiprocessing.Lock()

    manager = Manager()
    shared_output = manager.list()
    should_send_server_ready = Value(ctypes.c_bool, False)
    transcription_queue = Queue()

    whisper_server = TranscriptionServer()
    whisper_process = multiprocessing.Process(
        target=whisper_server.run,
        args=(
            "0.0.0.0",
            8080,
            transcription_queue,
            args.whisper_tensorrt_path,
            should_send_server_ready,
        ),
    )
    whisper_process.start()

    whisper_process.join()
