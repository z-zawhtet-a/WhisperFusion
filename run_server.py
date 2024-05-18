import argparse
from whisper_live.server import TranscriptionServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Websocket port to run the server on.",
    )
    parser.add_argument(
        "--trt_model_path",
        "-trt",
        type=str,
        default=None,
        help="Whisper TensorRT model path",
    )
    parser.add_argument(
        "--trt_multilingual",
        "-m",
        action="store_true",
        help="Boolean only for TensorRT model. True if multilingual.",
    )
    args = parser.parse_args()
    
    if args.trt_model_path is None:
        raise ValueError("Please Provide a valid tensorrt model path")

    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        port=args.port,
        whisper_tensorrt_path=args.trt_model_path,
        trt_multilingual=args.trt_multilingual,
    )
