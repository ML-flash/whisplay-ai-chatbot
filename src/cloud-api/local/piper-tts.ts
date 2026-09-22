import { spawn } from "child_process";
import fs from "fs";
import dotenv from "dotenv";

dotenv.config();

// Get paths from .env
const piperBinary = process.env.PIPER_BINARY_PATH;
const piperModel = process.env.PIPER_MODEL_PATH;

const streamAudio = (text: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    // 1. Validation
    if (!piperBinary || !piperModel) {
      console.error("Error: PIPER_BINARY_PATH or PIPER_MODEL_PATH not set in .env");
      resolve();
      return;
    }

    if (!fs.existsSync(piperBinary)) {
      console.error(`Error: Piper binary not found at ${piperBinary}`);
      resolve();
      return;
    }

    if (!fs.existsSync(piperModel)) {
      console.error(`Error: Piper model not found at ${piperModel}`);
      resolve();
      return;
    }

    try {
      // 2. Spawn Processes
      const piperProcess = spawn(piperBinary, [
        "--model", piperModel,
        "--output-raw"
      ]);

      const aplayProcess = spawn("aplay", [
        "-r", "22050",
        "-f", "S16_LE",
        "-t", "raw",
        "-"
      ]);

      // 3. Pipe Audio: Piper -> Aplay
      piperProcess.stdout.pipe(aplayProcess.stdin);

      // 4. Send Text to Piper
      piperProcess.stdin.write(text);
      piperProcess.stdin.end();

      // 5. Error Logging
      piperProcess.stderr.on("data", (data) => {});
      aplayProcess.stderr.on("data", (data) => {});

      // 6. Resolve when audio finishes
      aplayProcess.on("close", (code) => {
        resolve();
      });

      aplayProcess.on("error", (err) => {
        console.error("APlay Error:", err);
        resolve();
      });

      piperProcess.on("error", (err) => {
        console.error("Piper Error:", err);
        resolve();
      });

    } catch (error) {
      console.error("Stream Error:", error);
      resolve();
    }
  });
};

// EXPORT FIX: Export the function directly
export default streamAudio;
