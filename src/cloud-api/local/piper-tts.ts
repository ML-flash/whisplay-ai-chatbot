import { spawn, ChildProcessWithoutNullStreams } from "child_process";
import fs from "fs";
import dotenv from "dotenv";
import { ttsDir } from "../../utils/dir";
import { TTSResult } from "../../type";

dotenv.config();

const piperBinaryPath =
  process.env.PIPER_BINARY_PATH || "/home/pi/piper/piper";
const piperModelPath =
  process.env.PIPER_MODEL_PATH || "/home/pi/piper/voices/en_US-amy-medium.onnx";

// One long-running piper process: it loads the voice model once and turns
// each line written to stdin into a WAV file in ttsDir, printing its path.
// StreamResponser requests every sentence at once; a piper process per
// sentence reloads the model each time and runs a 512MB Pi out of memory.
// Here sentences are synthesized one at a time, in the order requested.
type PendingJob = { resolve: (result: TTSResult) => void };

let piper: ChildProcessWithoutNullStreams | null = null;
let stdoutBuffer = "";
const pending: PendingJob[] = [];

const wavResult = (filePath: string): TTSResult => {
  try {
    const header = Buffer.alloc(44);
    const fd = fs.openSync(filePath, "r");
    fs.readSync(fd, header, 0, 44, 0);
    fs.closeSync(fd);
    const channels = header.readUInt16LE(22);
    const sampleRate = header.readUInt32LE(24);
    const bitsPerSample = header.readUInt16LE(34);
    const dataBytes = fs.statSync(filePath).size - 44;
    const duration =
      (dataBytes / (sampleRate * channels * (bitsPerSample / 8))) * 1000;
    return { filePath, duration };
  } catch (error) {
    console.error("Failed to read Piper output:", filePath, error);
    return { duration: 0 };
  }
};

const failPending = (reason: string): void => {
  if (pending.length > 0) {
    console.error(`Piper ${reason}, dropping ${pending.length} sentence(s)`);
  }
  pending.splice(0).forEach((job) => job.resolve({ duration: 0 }));
};

const startPiper = (): ChildProcessWithoutNullStreams => {
  const proc = spawn(piperBinaryPath, [
    "--model",
    piperModelPath,
    "--output_dir",
    ttsDir,
    "--sentence_silence",
    "0.2",
    "--quiet",
  ]);
  stdoutBuffer = "";
  proc.stdout.setEncoding("utf8");
  proc.stdout.on("data", (chunk: string) => {
    stdoutBuffer += chunk;
    let newline: number;
    while ((newline = stdoutBuffer.indexOf("\n")) >= 0) {
      const filePath = stdoutBuffer.slice(0, newline).trim();
      stdoutBuffer = stdoutBuffer.slice(newline + 1);
      const job = pending.shift();
      job?.resolve(filePath ? wavResult(filePath) : { duration: 0 });
    }
  });
  proc.stderr.on("data", (data) => {
    console.error("Piper:", data.toString().trim());
  });
  proc.on("error", (error) => {
    console.error("Piper process error:", error);
    piper = null;
    failPending("failed to start");
  });
  proc.on("exit", (code, signal) => {
    piper = null;
    failPending(`exited (code ${code}, signal ${signal})`);
  });
  return proc;
};

const piperTTS = (text: string): Promise<TTSResult> => {
  // piper reads one utterance per line
  const line = text.replace(/\s+/g, " ").trim();
  if (!line) {
    return Promise.resolve({ duration: 0 });
  }
  return new Promise<TTSResult>((resolve) => {
    if (!piper) {
      piper = startPiper();
    }
    pending.push({ resolve });
    piper.stdin.write(line + "\n");
  });
};

// load the voice model at startup (~5s on a Pi Zero 2 W) rather than when
// the first answer is already waiting to be spoken
if ((process.env.TTS_SERVER || "").toLowerCase() === "piper") {
  piper = startPiper();
}

export default piperTTS;
