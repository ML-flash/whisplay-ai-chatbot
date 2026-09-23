# Rebuilding the device from a blank SD card

Target: Raspberry Pi Zero 2 W + PiSugar Whisplay HAT (WM8960 audio), talking
to LM Studio on the home PC.

## 1. Flash the card (Raspberry Pi Imager, on the PC)

Use a good card: SanDisk or Samsung, A1/A2 rated, 32 GB or more.

- **Device:** Raspberry Pi Zero 2 W
- **OS:** Raspberry Pi OS (other) → **Raspberry Pi OS Lite (64-bit)** (Debian 13 "trixie")
- **Edit settings** (OS customisation):
  - Hostname: `ChatPi`
  - Username / password: `avilanch2000` / your password
  - Wireless LAN: your SSID, password, country
  - Locale: your time zone and keyboard
  - **Services:** enable SSH → **Allow public-key authentication only**, and
    paste the PC's key from `C:\Users\Matt\.ssh\id_ed25519_pi.pub`

## 2. First boot

Insert the card, power the Pi with a supply rated **5 V / 2.5 A** (the Pi
Zero 2 W plus the HAT's speaker amplifier peaks well above 1 A), and give it
a couple of minutes to join Wi-Fi. From the PC: `ssh pi` (see `~/.ssh/config`;
update `HostName` if the Pi's address changed).

## 3. Run the bootstrap

```bash
curl -fsSL https://raw.githubusercontent.com/ML-flash/whisplay-ai-chatbot/master/setup/pi-bootstrap.sh -o pi-bootstrap.sh
LLM_HOST=172.16.101.249 bash pi-bootstrap.sh
sudo reboot
```

It installs the Whisplay WM8960 driver (legacy `support/wm8960` branch),
Node 20 + the chatbot build, Vosk (speech-to-text), Piper (voice), writes
`.env`, trims the system for a headless Pi Zero (no KMS driver, 16 MB GPU
memory, unneeded services off), keeps the system journal in RAM, rotates
the chatbot log, and installs `chatbot.service`. Roughly 30-60 minutes; safe
to re-run.

## 4. Check

```bash
systemctl status chatbot wm8960-soundcard --no-pager
vcgencmd get_throttled     # 0x0 = no undervoltage or throttling since boot
tail -f ~/whisplay-ai-chatbot/chatbot.log
```
