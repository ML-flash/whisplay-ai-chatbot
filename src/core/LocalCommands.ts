import { display, getCurrentStatus } from "../device/display";
import { getCurrentLogPercent, setVolumeByAmixer } from "../utils/volume";
import { loadSettings, saveSettings } from "../utils/settings";

// Device commands handled entirely on the Pi, without the LLM, so they are
// instant and work when the LLM server is unreachable. A command only
// matches when it is the whole utterance ("volume fifty"); anything else,
// e.g. "what's the volume of the sun", goes to the LLM as usual.

export type LocalCommand =
  | { target: "volume" | "brightness"; action: "set"; value: number }
  | { target: "volume" | "brightness"; action: "up" | "down" | "query" }
  | { target: "volume"; action: "mute" | "unmute" }
  | { target: "screen"; action: "on" | "off" };

const VOLUME_STEP = 10;
const BRIGHTNESS_STEP = 20;
// "set brightness to 0" would leave a black screen; "screen off" does that
const MIN_BRIGHTNESS = 5;

const SMALL_NUMBERS: Record<string, number> = {
  zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7,
  eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13,
  fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18,
  nineteen: 19,
};
const TENS: Record<string, number> = {
  twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70,
  eighty: 80, ninety: 90,
};

// "50", "fifty", "twenty five", "a hundred", "half", "max" -> 0..100
export const parsePercent = (phrase: string): number | null => {
  const words = phrase.trim().split(" ").filter(Boolean);
  if (words.length === 0) return null;
  if (words.length === 1 && /^\d{1,3}$/.test(words[0])) {
    const value = parseInt(words[0], 10);
    return value <= 100 ? value : null;
  }
  const joined = words.join(" ");
  if (joined === "half") return 50;
  if (["max", "maximum", "full", "hundred", "a hundred", "one hundred"].includes(joined)) {
    return 100;
  }
  if (words.length === 1 && words[0] in SMALL_NUMBERS) return SMALL_NUMBERS[words[0]];
  if (words[0] in TENS) {
    if (words.length === 1) return TENS[words[0]];
    const unit = SMALL_NUMBERS[words[1]];
    if (words.length === 2 && unit !== undefined && unit >= 1 && unit <= 9) {
      return TENS[words[0]] + unit;
    }
  }
  return null;
};

const FILLER_PREFIXES = [
  "please", "hey", "ok", "okay", "can you", "could you", "would you",
  "will you", "i want you to", "id like you to", "i would like you to",
];
const FILLER_WORDS = new Set(["the", "your", "my", "please", "percent", "a", "bit", "little", "some", "level"]);

export const normalizeCommandText = (text: string): string => {
  let normalized = text
    .toLowerCase()
    .replace(/%/g, " percent")
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9 ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  let stripped = true;
  while (stripped) {
    stripped = false;
    for (const prefix of FILLER_PREFIXES) {
      if (normalized === prefix || normalized.startsWith(prefix + " ")) {
        normalized = normalized.slice(prefix.length).trim();
        stripped = true;
      }
    }
  }
  // number words like "a hundred" keep their "a"
  normalized = normalized.replace(/\ba hundred\b/g, "hundred");
  return normalized
    .split(" ")
    .filter((word) => !FILLER_WORDS.has(word))
    .join(" ")
    .replace(/\bfor me$/, "")
    .trim();
};

const VOLUME_WORDS = "(?:volume|sound)";
const BRIGHTNESS_WORDS = "(?:(?:screen |display )?brightness)";
const SCREEN_WORDS = "(?:screen|display)";

const phrases = (...patterns: string[]) =>
  new RegExp(`^(?:${patterns.join("|")})$`);

const VOLUME_UP = phrases(
  `(?:turn |crank )?${VOLUME_WORDS} up`, "turn it up", "louder",
  "(?:speak|talk|be) louder", `(?:increase|raise) ${VOLUME_WORDS}`,
);
const VOLUME_DOWN = phrases(
  `(?:turn )?${VOLUME_WORDS} down`, "turn it down", "quieter", "softer",
  "(?:speak|talk|be) (?:quieter|softer)", `(?:decrease|lower|reduce) ${VOLUME_WORDS}`,
);
const VOLUME_MUTE = phrases("mute", `mute (?:${VOLUME_WORDS}|speaker|yourself|audio)`);
const VOLUME_UNMUTE = phrases("unmute", `unmute (?:${VOLUME_WORDS}|speaker|yourself|audio)`);
const VOLUME_MAX = phrases(`(?:max|maximum|full) ${VOLUME_WORDS}`, `${VOLUME_WORDS} (?:max|maximum|full)`, "loudest");
const VOLUME_QUERY = phrases(`(?:whats|what is) ${VOLUME_WORDS}`, "how loud are you");
const VOLUME_SET = new RegExp(`^(?:set |change |turn |put )?${VOLUME_WORDS} (?:to |at )?(.+)$`);

const BRIGHTNESS_UP = phrases(
  "brighter", `(?:turn )?${BRIGHTNESS_WORDS} up`, `(?:increase|raise) ${BRIGHTNESS_WORDS}`,
  `(?:make )?${SCREEN_WORDS} brighter`,
);
const BRIGHTNESS_DOWN = phrases(
  "dimmer", "dim", `dim ${SCREEN_WORDS}`, `(?:turn )?${BRIGHTNESS_WORDS} down`,
  `(?:decrease|lower|reduce) ${BRIGHTNESS_WORDS}`, `(?:make )?${SCREEN_WORDS} dimmer`,
);
const BRIGHTNESS_MAX = phrases(`(?:max|maximum|full) ${BRIGHTNESS_WORDS}`, `${BRIGHTNESS_WORDS} (?:max|maximum|full)`);
const BRIGHTNESS_QUERY = phrases(`(?:whats|what is) ${BRIGHTNESS_WORDS}`);
const BRIGHTNESS_SET = new RegExp(`^(?:set |change |turn |put )?${BRIGHTNESS_WORDS} (?:to |at )?(.+)$`);

const SCREEN_OFF = phrases(`(?:turn )?${SCREEN_WORDS} off`, `turn off ${SCREEN_WORDS}`);
const SCREEN_ON = phrases(`(?:turn )?${SCREEN_WORDS} on`, `turn on ${SCREEN_WORDS}`);

export const matchLocalCommand = (text: string): LocalCommand | null => {
  const t = normalizeCommandText(text);
  if (!t) return null;

  if (VOLUME_UP.test(t)) return { target: "volume", action: "up" };
  if (VOLUME_DOWN.test(t)) return { target: "volume", action: "down" };
  if (VOLUME_MUTE.test(t)) return { target: "volume", action: "mute" };
  if (VOLUME_UNMUTE.test(t)) return { target: "volume", action: "unmute" };
  if (VOLUME_MAX.test(t)) return { target: "volume", action: "set", value: 100 };
  if (VOLUME_QUERY.test(t)) return { target: "volume", action: "query" };
  const volumeSet = t.match(VOLUME_SET);
  if (volumeSet) {
    const value = parsePercent(volumeSet[1]);
    if (value !== null) return { target: "volume", action: "set", value };
  }

  if (BRIGHTNESS_UP.test(t)) return { target: "brightness", action: "up" };
  if (BRIGHTNESS_DOWN.test(t)) return { target: "brightness", action: "down" };
  if (BRIGHTNESS_MAX.test(t)) return { target: "brightness", action: "set", value: 100 };
  if (BRIGHTNESS_QUERY.test(t)) return { target: "brightness", action: "query" };
  const brightnessSet = t.match(BRIGHTNESS_SET);
  if (brightnessSet) {
    const value = parsePercent(brightnessSet[1]);
    if (value !== null) return { target: "brightness", action: "set", value };
  }

  if (SCREEN_OFF.test(t)) return { target: "screen", action: "off" };
  if (SCREEN_ON.test(t)) return { target: "screen", action: "on" };
  return null;
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, Math.round(value)));

const setVolume = (value: number): void => {
  setVolumeByAmixer(value);
  saveSettings({ volume: value });
};

const setBrightness = (value: number): void => {
  display({ brightness: value });
  saveSettings({ brightness: value });
};

// Apply the command and return the confirmation to show and speak.
export const runLocalCommand = (command: LocalCommand): string => {
  if (command.target === "screen") {
    const current = getCurrentStatus().brightness;
    if (command.action === "off") {
      if (current > 0) saveSettings({ brightnessBeforeOff: current });
      setBrightness(0);
      return "Screen off. Say screen on to turn it back on.";
    }
    if (current > 0) return "The screen is already on.";
    setBrightness(loadSettings().brightnessBeforeOff || 100);
    return "Screen on.";
  }

  if (command.target === "volume") {
    const current = Math.round(getCurrentLogPercent());
    switch (command.action) {
      case "query":
        return `Volume is at ${current} percent.`;
      case "mute":
        saveSettings({ volumeBeforeMute: current });
        setVolume(0);
        return "Muted.";
      case "unmute": {
        const restored = loadSettings().volumeBeforeMute || 50;
        setVolume(restored);
        return `Volume restored to ${restored} percent.`;
      }
      case "up":
      case "down": {
        const next = clamp(current + (command.action === "up" ? VOLUME_STEP : -VOLUME_STEP), 0, 100);
        if (next === current) {
          return `Volume is already at ${command.action === "up" ? "maximum" : "minimum"}.`;
        }
        setVolume(next);
        return `Volume ${next} percent.`;
      }
      case "set": {
        const value = clamp(command.value, 0, 100);
        setVolume(value);
        return `Volume set to ${value} percent.`;
      }
    }
  }

  const current = getCurrentStatus().brightness;
  switch (command.action) {
    case "query":
      return `Brightness is at ${current} percent.`;
    case "up":
    case "down": {
      const next = clamp(current + (command.action === "up" ? BRIGHTNESS_STEP : -BRIGHTNESS_STEP), MIN_BRIGHTNESS, 100);
      if (next === current) {
        return `Brightness is already at ${command.action === "up" ? "maximum" : "minimum"}.`;
      }
      setBrightness(next);
      return `Brightness ${next} percent.`;
    }
    case "set": {
      const value = clamp(command.value, MIN_BRIGHTNESS, 100);
      setBrightness(value);
      return `Brightness set to ${value} percent.`;
    }
  }
  return "";
};

// Restore volume and brightness saved by earlier voice commands.
export const applySavedSettings = (): void => {
  const { volume, brightness } = loadSettings();
  if (typeof volume === "number") {
    try {
      setVolumeByAmixer(volume);
    } catch (error) {
      console.error("Failed to restore volume:", error);
    }
  }
  if (typeof brightness === "number") {
    display({ brightness });
  }
};
