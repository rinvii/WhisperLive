import sys
import argparse
import time
import threading
import math
import tkinter as tk
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from pynput.keyboard import Controller, Key
from whisper_live.client import TranscriptionClient


class AudioFeedback:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.sample_rate = 44100

    def play_activation(self, active=True):
        """Plays a pleasant modern beep using sounddevice."""
        if not self.enabled:
            return
        try:
            duration = 0.12
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)

            if active:
                # "Pop" On: Slight upward chirp
                freq_start, freq_end = 600, 800
                freq = np.linspace(freq_start, freq_end, len(t))
                wave = np.sin(2 * np.pi * freq * t)
            else:
                # "Pop" Off: Slight downward chirp
                freq_start, freq_end = 500, 300
                freq = np.linspace(freq_start, freq_end, len(t))
                wave = np.sin(2 * np.pi * freq * t)

            # Soft envelope
            attack = int(self.sample_rate * 0.01)
            decay = int(self.sample_rate * 0.05)
            envelope = np.ones_like(t)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-decay:] = np.linspace(1, 0, decay)

            audio = wave * envelope * 0.15
            sd.play(audio.astype(np.float32), self.sample_rate)
        except Exception:
            pass

    def play_success(self, word_count):
        """Plays a fixed sequence of cute pentatonic chirps on success."""
        if not self.enabled:
            return
        try:
            if word_count <= 0:
                return

            # Pentatonic scale (High/Cute): C6, D6, E6, G6, A6
            scale = [1046.50, 1174.66, 1318.51, 1567.98, 1760.00]
            note_dur = 0.05  # 50ms per chirp (slightly faster)
            gap_dur = 0.015  # 15ms gap

            # Fixed 3 chirps for a snappy "success" sound
            play_count = 3

            full_audio = []

            for _ in range(play_count):
                # Pick 3 random notes for a unique "bloop" each time
                freq = scale[np.random.randint(0, len(scale))]

                t = np.linspace(0, note_dur, int(self.sample_rate * note_dur), False)

                # Bubble sound: Sine wave with rapid pitch bend + Gaussian envelope
                freq_sweep = np.linspace(freq, freq * 0.85, len(t))
                wave = np.sin(2 * np.pi * freq_sweep * t)

                # Gaussian envelope for soft "bloop"
                envelope = np.exp(-0.5 * ((t - note_dur / 2) / (note_dur / 6)) ** 2)

                chunk = wave * envelope
                full_audio.append(chunk)
                full_audio.append(np.zeros(int(self.sample_rate * gap_dur)))  # Gap

            audio_data = np.concatenate(full_audio) * 0.12  # Slightly lower volume
            sd.play(audio_data.astype(np.float32), self.sample_rate)

        except Exception:
            pass


class TextInjector:
    def __init__(self):
        self.controller = Controller()

    def inject(self, text):
        """Inserts text using key-by-key typing."""
        self.controller.type(text + " ")


class KeyboardTranscriptionClient:
    def __init__(self, host, port, model, lang, use_ui=False, use_audio=False):
        self.host, self.port, self.model, self.lang = host, port, model, lang
        self.use_ui = use_ui

        # Initialize components
        self.audio = AudioFeedback(enabled=use_audio)
        self.injector = TextInjector()

        self.is_listening = False
        self.current_segments = []
        self.level = 0.0
        self.max_level = 0.01
        self.base_segment_count = 0
        self.stop_timer = None
        self.BURST_LIMIT = 60.0  # 1 minute limit

        if self.use_ui:
            self.root = tk.Tk()
            self._setup_ui()

        self.client = None
        self._connect()

    def _connect(self):
        print("[INFO]: Establishing persistent connection...")
        self.client = TranscriptionClient(
            self.host,
            self.port,
            lang=self.lang,
            model=self.model,
            log_transcription=False,
            send_last_n_segments=100,
            no_speech_thresh=0.1,
            transcription_callback=self.on_transcription,
        )

        # Monkey-patching multicast_packet to gate audio sending
        orig = self.client.multicast_packet

        def gated(packet, unconditional=False):
            if self.is_listening:
                try:
                    d = np.frombuffer(packet, dtype=np.float32)
                    lvl = float(np.abs(d).mean())
                    self.level = 0.5 * self.level + 0.5 * lvl
                    if lvl > self.max_level:
                        self.max_level = lvl
                    self.max_level *= 0.99
                except Exception:
                    pass
            if self.is_listening or unconditional:
                try:
                    orig(packet, unconditional)
                except Exception:
                    pass

        self.client.multicast_packet = gated  # type: ignore

        if self.client.client:
            self.client.client.on_error = lambda ws, err: None  # type: ignore
        threading.Thread(target=self.client, daemon=True).start()

    def _setup_ui(self):
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)
        self.root.configure(bg="#000001")
        self.root.wm_attributes("-transparent", True)
        w, h = 80, 24
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{sh - 100}")
        self.canvas = tk.Canvas(
            self.root, width=w, height=h, bg="#000001", highlightthickness=0
        )
        self.canvas.pack()
        self.bars = [
            self.canvas.create_rectangle(
                22 + i * 3, 12, 23 + i * 3, 12, fill="#00ffcc", outline=""
            )
            for i in range(12)
        ]
        self.root.update()

    def _update_ui(self):
        if self.use_ui:
            if self.is_listening:
                norm = self.level / max(0.001, self.max_level)
                now = time.time()
                for i, bar in enumerate(self.bars):
                    target_h = max(
                        2, min(2 + (norm * 4) + (math.sin(now * 12 + i * 0.5) * 2), 6)
                    )
                    coords = self.canvas.coords(bar)
                    self.canvas.coords(
                        bar, coords[0], 12 - target_h / 2, coords[2], 12 + target_h / 2
                    )
            self.root.after(50, self._update_ui)

    def on_transcription(self, full_text, segments):
        self.current_segments = segments

    def force_stop_burst(self):
        if self.is_listening:
            print("\n[INFO]: Burst limit reached (60s). Stopping...")
            # Call toggle_listening via main thread if UI is used
            if self.use_ui:
                self.root.after(0, self.toggle_listening)
            else:
                self.toggle_listening()

    def toggle_listening(self):
        if not self.is_listening:
            # Start Listening
            # Ensure connection is ready
            if not self.client or not getattr(self.client.client, 'recording', False):
                print('[INFO]: Re-establishing connection...')
                self._connect()
                
                # Wait for server to be ready (max 5 seconds)
                for _ in range(50):
                    if self.client and getattr(self.client.client, 'recording', False):
                        break
                    time.sleep(0.1)
                else:
                    print('[ERROR]: Connection timeout. Server not ready.')
                    return

            self.audio.play_activation(active=True)
            self.base_segment_count = len(self.current_segments)
            self.is_listening = True
            print('\n[MIC]: ON')
            if self.use_ui: self.root.attributes('-alpha', 1.0)
            
            # Start burst timer
            self.stop_timer = threading.Timer(self.BURST_LIMIT, self.force_stop_burst)
            self.stop_timer.start()

        else:
            # Stop Listening
            if self.stop_timer:
                self.stop_timer.cancel()
                self.stop_timer = None

            self.is_listening = False
            self.audio.play_activation(active=False)
            print("[MIC]: OFF - Finalizing...")
            if self.use_ui:
                self.root.attributes("-alpha", 0.0)

            try:
                # Flush buffer
                silence = np.zeros(2048, dtype=np.float32).tobytes()
                for _ in range(3):
                    if self.client:
                        self.client.multicast_packet(silence, unconditional=True)
            except Exception:
                pass

            time.sleep(0.7)
            new_segments = self.current_segments[self.base_segment_count :]
            if new_segments:
                text = " ".join(
                    [s["text"].strip() for s in new_segments if s["text"].strip()]
                )
                if text:
                    self.injector.inject(text)
                    print(f"[SUCCESS]: {text}")
                    # Play word chirps in background
                    threading.Thread(
                        target=self.audio.play_success, args=(len(text.split()),)
                    ).start()
            else:
                print("[INFO]: No new speech detected.")

    def run(self):
        def on_press(key):
            if key == keyboard.Key.alt_r:
                if self.use_ui:
                    self.root.after(0, self.toggle_listening)
                else:
                    self.toggle_listening()

        keyboard.Listener(on_press=on_press).start()

        info_str = "[INFO]: Ready - RIGHT OPTION"
        if self.use_ui:
            info_str += " | UI Active"
        if self.audio.enabled:
            info_str += " | Audio Active"

        if self.use_ui:
            self._update_ui()
            print(info_str)
            self.root.mainloop()
        else:
            print(info_str)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[INFO]: Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", "-s", type=str, default="localhost")
    parser.add_argument("--port", "-p", type=int, default=9095)
    parser.add_argument("--model", "-m", type=str, default="medium")
    parser.add_argument("--lang", "-l", type=str, default="en")
    parser.add_argument("--ui", action="store_true", help="Enable visual overlay")
    parser.add_argument(
        "--audio", action="store_true", help="Enable audio feedback chirps"
    )
    args = parser.parse_args()
    if ":" in args.server:
        host, port = args.server.split(":")
        args.server, args.port = host, int(port)
    app = KeyboardTranscriptionClient(
        args.server,
        args.port,
        args.model,
        args.lang,
        use_ui=args.ui,
        use_audio=args.audio,
    )
    app.run()
