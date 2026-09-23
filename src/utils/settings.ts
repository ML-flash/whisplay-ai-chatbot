import fs from "fs";
import path from "path";

// Device settings changed by voice command, kept across restarts.
// Stored in the project root rather than data/, which CLEAN_DATA_FOLDER_ON_START
// may wipe.
export type DeviceSettings = {
  volume?: number; // 0-100
  brightness?: number; // 0-100
  volumeBeforeMute?: number;
  brightnessBeforeOff?: number;
};

const settingsPath = path.join(__dirname, "../..", "settings.json");

export const loadSettings = (): DeviceSettings => {
  try {
    return JSON.parse(fs.readFileSync(settingsPath, "utf8"));
  } catch {
    return {};
  }
};

export const saveSettings = (changes: DeviceSettings): void => {
  const settings = { ...loadSettings(), ...changes };
  try {
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
  } catch (error) {
    console.error("Failed to save settings:", error);
  }
};
