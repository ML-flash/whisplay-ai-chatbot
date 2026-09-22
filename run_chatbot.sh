#!/bin/bash
# Set working directory
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Find the sound card index for wm8960soundcard
card_index=$(awk '/wm8960soundcard/ {print $1}' /proc/asound/cards | head -n1)
# Default to 1 if not found
if [ -z "$card_index" ]; then
  card_index=1
fi
echo "Using sound card index: $card_index"

# Output current environment information (for debugging)
echo "===== Start time: $(date) =====" 
echo "Current user: $(whoami)" 
echo "Working directory: $(pwd)" 
working_dir=$(pwd)
echo "PATH: $PATH" 
echo "Python version: $(python3 --version)" 
echo "Node version: $(node --version)"
sleep 5
# Adjust volume
amixer -c $card_index set Speaker 114
# Start the service
echo "Starting Node.js application..."
cd $working_dir

# load .env variables, exclude comments and empty lines
# check if .env file exists
serve_ollama=false
if [ -f ".env" ]; then
  # Load only SERVE_OLLAMA from .env (ignore comments/other vars)
  if grep -Eq '^[[:space:]]*SERVE_OLLAMA[[:space:]]*=' .env; then
    val=$(grep -E '^[[:space:]]*SERVE_OLLAMA[[:space:]]*=' .env | tail -n1 | cut -d'=' -f2-)
    # trim whitespace and surrounding quotes
    SERVE_OLLAMA=$(echo "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
    export SERVE_OLLAMA
  fi
  # CUSTOM_FONT_PATH for displaying text on screen
  if grep -Eq '^[[:space:]]*CUSTOM_FONT_PATH[[:space:]]*=' .env; then
    val=$(grep -E '^[[:space:]]*CUSTOM_FONT_PATH[[:space:]]*=' .env | tail -n1 | cut -d'=' -f2-)
    # trim whitespace and surrounding quotes
    CUSTOM_FONT_PATH=$(echo "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
    export CUSTOM_FONT_PATH
  fi
  echo ".env variables loaded."
  # check if SERVE_OLLAMA is set to true
  if [ "$SERVE_OLLAMA" = "true" ]; then
    serve_ollama=true
  fi
else
  echo ".env file not found, please create one based on .env.template."
  exit 1
fi

if [ "$serve_ollama" = true ]; then
  echo "Starting Ollama server..."
  ollama serve &
fi

# start the persistent vosk server so the model is loaded only once
asr_server=$(grep -E '^[[:space:]]*ASR_SERVER[[:space:]]*=' .env | tail -n1 | cut -d'=' -f2- | tr -d '[:space:]"'"'")
vosk_pid=""
if [ "$asr_server" = "vosk" ]; then
  VOSK_MODEL_PATH=$(grep -E '^[[:space:]]*VOSK_MODEL_PATH[[:space:]]*=' .env | tail -n1 | cut -d'=' -f2- | tr -d '[:space:]"'"'")
  export VOSK_MODEL_PATH
  echo "Starting Vosk server..."
  python3 python/vosk_server.py &
  vosk_pid=$!
  # wait for the model to load before accepting requests
  for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:8804/health > /dev/null && break
    sleep 1
  done
fi

# if file use_npm exists and is true, use npm
if [ -f "use_npm" ]; then
  use_npm=true
else
  use_npm=false
fi

if [ "$use_npm" = true ]; then
  echo "Using npm to start the application..."
  SOUND_CARD_INDEX=$card_index npm start
else
  # run the compiled output directly: skipping the yarn wrapper and ts-node
  # saves ~80MB of RAM, which matters on a 512MB Pi Zero 2 W
  echo "Starting the application with node..."
  SOUND_CARD_INDEX=$card_index node dist/index.js
fi

# After the service ends, perform cleanup
echo "Cleaning up after service..."

if [ "$serve_ollama" = true ]; then
  echo "Stopping Ollama server..."
  pkill ollama
fi

if [ -n "$vosk_pid" ]; then
  echo "Stopping Vosk server..."
  kill $vosk_pid
fi

# Record end status
echo "===== Service ended: $(date) ====="
