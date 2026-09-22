"""Persistent Vosk ASR server.

Loads the Vosk model once and serves transcriptions over HTTP on localhost,
so each request skips the multi-second model load of vosk-transcriber.

POST /transcribe  {"path": "/abs/path/to/audio.wav"}  ->  {"text": "..."}
GET  /health      -> {"ok": true}
"""

import json
import os
import subprocess
import sys
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

from vosk import KaldiRecognizer, Model, SetLogLevel

SAMPLE_RATE = 16000
HOST = "127.0.0.1"
PORT = int(os.environ.get("VOSK_SERVER_PORT", "8804"))


def read_pcm(path):
    """Return 16kHz mono 16-bit PCM bytes, converting with ffmpeg if needed."""
    try:
        with wave.open(path, "rb") as wf:
            if (
                wf.getframerate() == SAMPLE_RATE
                and wf.getnchannels() == 1
                and wf.getsampwidth() == 2
                # a recorder killed mid-write leaves a zero-length header
                and wf.getnframes() > 0
            ):
                return wf.readframes(wf.getnframes())
    except (wave.Error, EOFError):
        pass
    # non-matching or truncated file: let ffmpeg decode whatever it can
    return subprocess.run(
        ["ffmpeg", "-loglevel", "quiet", "-i", path,
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True,
    ).stdout


def transcribe(rec, path):
    # rec is reused across requests: FinalResult() resets it, and the first
    # decode on a fresh recognizer is several times slower than later ones
    pcm = read_pcm(path)
    parts = []
    chunk = 8000
    for i in range(0, len(pcm), chunk):
        if rec.AcceptWaveform(pcm[i:i + chunk]):
            parts.append(json.loads(rec.Result()).get("text", ""))
    parts.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(p for p in parts if p).strip()


def make_handler(rec):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/transcribe":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                path = json.loads(self.rfile.read(length))["path"]
                if not os.path.isfile(path):
                    self._send(400, {"error": f"file not found: {path}"})
                    return
                self._send(200, {"text": transcribe(rec, path)})
            except Exception as e:
                self._send(500, {"error": repr(e)})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    model_path = os.environ.get("VOSK_MODEL_PATH") or (
        sys.argv[1] if len(sys.argv) > 1 else ""
    )
    if not model_path:
        sys.exit("VOSK_MODEL_PATH is not set")
    SetLogLevel(-1)
    model = Model(model_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    # warm up on a short speech clip so the first real request is fast
    # (silence or noise do not exercise the decoder enough to help)
    warmup = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vosk_warmup.wav")
    if os.path.isfile(warmup):
        transcribe(rec, warmup)
    server = HTTPServer((HOST, PORT), make_handler(rec))
    print(f"vosk server listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
