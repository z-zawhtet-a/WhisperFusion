import base64
import json
import threading
import time

import soundfile as sf
import websocket


class WhisperClient:
    """
    A client for the Whisper Live server.
    """

    def __init__(
        self, host, port, api_key, wav_file, language="en", encoding="linear16"
    ):
        """
        Initialize the client.

        Args:
            host (str): The host of the server.
            port (str): The port of the server.
            api_key (str): The API key for the server.
            wav_file (str): The path to the WAV file to be transcribed.
            language (str): The language of the audio.
            encoding (str): The encoding of the audio.
        """
        self.socket_url = f"ws://{host}:{port}"
        self.api_key = api_key
        self.wav_file = wav_file
        self.encoding = encoding
        self.client_socket = None
        self.is_connected = False
        self.transcription_received = threading.Event()
        self.latest_transcription = None
        self.language = language

    def connect(self):
        self.client_socket = websocket.WebSocketApp(
            self.socket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.client_socket.run_forever()

    def on_open(self, ws):
        print("Connection opened")
        self.is_connected = True
        self.send_initial_message()
        threading.Thread(target=self.stream_audio).start()

    def on_message(self, ws, message):
        print(f"Received message: {message}")
        data = json.loads(message)

        if "status" in data:
            if data["status"] == "WAIT":
                print(f"Server is full. Estimated wait time: {data['message']} minutes")
                return
            elif data["status"] == "WARNING":
                print(f"Warning from server: {data['message']}")
                return

        if data.get("message") == "SERVER_READY":
            print("Server is ready to receive audio")
            return

        if data.get("message") == "DISCONNECT":
            print("Server has disconnected")
            self.is_connected = False
            return

        self.latest_transcription = data
        self.transcription_received.set()

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"Connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False

    def send_initial_message(self):
        initial_message = {
            "api_key": self.api_key,
            "encoding": self.encoding,
            "language": self.language,
            "task": "transcribe",
            "uid": "test_client",
            "use_vad": True,
        }
        self.client_socket.send(json.dumps(initial_message))

    def send_audio(self, audio_chunk):
        if self.is_connected:
            message = {"audio": base64.b64encode(audio_chunk).decode("utf-8")}
            self.client_socket.send(json.dumps(message))

    def stream_audio(self):
        chunk_duration = 0.1  # 100ms chunks
        with sf.SoundFile(self.wav_file) as audio_file:
            sample_rate = audio_file.samplerate
            channels = audio_file.channels
            chunk_size = int(sample_rate * chunk_duration)

            print(
                f"Audio details: {channels} channels, {sample_rate} Hz, {audio_file.format} format"
            )

            while audio_file.tell() < audio_file.frames and self.is_connected:
                chunk = audio_file.read(chunk_size, dtype="int16")
                if len(chunk) == 0:
                    break

                chunk_bytes = chunk.tobytes()
                self.send_audio(chunk_bytes)

                # Wait for transcription or timeout
                self.transcription_received.wait(timeout=0.5)
                self.transcription_received.clear()

                if self.latest_transcription:
                    segments = self.latest_transcription.get("segments", [])
                    eos = self.latest_transcription.get("eos", False)
                    for segment in segments:
                        print(f"Transcription: {segment['text']}")
                    if eos:
                        print("End of speech detected")

                # Simulate real-time streaming
                time.sleep(chunk_duration)

        print("Finished streaming audio")
        self.client_socket.close()


def main():
    client = WhisperClient(
        "localhost",
        "8080",
        "helloworld",
        "test_mulaw.wav",
        language="en",
        encoding="linear16",
    )
    client.connect()


if __name__ == "__main__":
    main()
