import argparse
from whisper_live.server import TranscriptionServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8081,
        help="Websocket port to run the server on.",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default=None,
        help="Whisper model path",
    )
    args = parser.parse_args()
    
    if args.trt_model_path is None:
        raise ValueError("Please Provide a valid tensorrt model path")

    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        port=args.port,
        faster_whisper_custom_model_path=args.model_path,
    )
