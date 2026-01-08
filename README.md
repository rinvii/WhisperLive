# WhisperLive (Lightweight Client)

A fork of Collabora's WhisperLive optimized for easier installation and global keyboard dictation.

## Key Changes
- Swapped `PyAudio` for `sounddevice` for cross-platform compatibility.
- Made `av` (FFmpeg) optional to support legacy systems (macOS 10.15+).
- Added `run_client_keyboard.py` for OS-level dictation via hotkey toggle.
- Implemented "Burst-Typing" logic to eliminate real-time word stuttering.

## Alternative to Commercial Tools
This provides a private, self-hosted alternative to proprietary dictation software. It uses a remote GPU for inference while acting as a local keyboard input on your workstation.

## Setup
1. **Install Dependencies**:
   ```bash
   uv pip install sounddevice numpy websocket-client pynput scipy
   ```

2. **Run Client**:
   ```bash
   python run_client_keyboard.py --server YOUR_SERVER_IP:9095
   ```

3. **Usage**:
   - Tap **RIGHT SHIFT** to start/stop recording.
   - Text is typed automatically into the active window when finished.

## Original Project
For the full server setup and TensorRT acceleration, see the [Collabora WhisperLive Repository](https://github.com/collabora/WhisperLive).