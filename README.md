# WhisperLive (Lightweight Client)

A fork of WhisperLive designed for global keyboard dictation. This tool allows you to use your own private, GPU-accelerated server to dictate text directly into any application on your computer.

## Features

- **Global Hotkey:** Toggle dictation from anywhere (Default: Right Option).
- **Audio & Visual Feedback:** Gentle chirps and a minimal overlay confirm when you are recording.
- **Safety:** Automatic timeout ensures your microphone doesn't stay open by accident.

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements/client.txt
   ```

2. **Run Server** (on your GPU machine):

   ```bash
   python run_server.py --port 9095
   ```

3. **Run Client**:
   ```bash
   python run_client_keyboard.py --server YOUR_SERVER_IP --port 9095 --ui --audio
   ```

## Usage

- **Toggle Recording:** Press **Right Option (Alt)**.
- **Dictate:** Speak clearly; text will type out automatically.

---

_Based on the [Collabora WhisperLive Repository](https://github.com/collabora/WhisperLive)._
