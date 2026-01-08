import sys
import argparse
import time
import threading
import math
import tkinter as tk
from pynput import keyboard
from pynput.keyboard import Controller, Key
from whisper_live.client import TranscriptionClient

class KeyboardTranscriptionClient:
    def __init__(self, host, port, model, lang, use_ui=False):
        self.keyboard_controller = Controller()
        self.is_listening = False
        self.base_segment_count = 0
        self.current_segments = []
        self.use_ui = use_ui
        
        # Audio performance metrics
        self.level = 0.0
        self.max_level = 0.01
        
        if self.use_ui:
            self.root = tk.Tk()
            self._setup_ui()
        
        self.client = TranscriptionClient(
            host, port, lang=lang, model=model,
            log_transcription=False, send_last_n_segments=100,
            no_speech_thresh=0.1, transcription_callback=self.on_transcription
        )
        
        # Patch client for background audio analysis
        original_multicast = self.client.multicast_packet
        def gated_multicast(packet, unconditional=False):
            if self.is_listening and self.use_ui:
                try:
                    import numpy as np
                    data = np.frombuffer(packet, dtype=np.float32)
                    lvl = float(np.abs(data).mean())
                    self.level = 0.5 * self.level + 0.5 * lvl
                    if lvl > self.max_level: self.max_level = lvl
                    self.max_level *= 0.99
                except: pass
            
            if self.is_listening or unconditional:
                original_multicast(packet, unconditional)
        self.client.multicast_packet = gated_multicast

    def _setup_ui(self):
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.0)
        self.root.configure(bg='#000001')
        self.root.wm_attributes('-transparent', True)
        w, h = 60, 24
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f'{w}x{h}+{(sw-w)//2}+{sh-80}')
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg='#000001', highlightthickness=0)
        self.canvas.pack()
        self.bars = [self.canvas.create_rectangle(12+i*3, 12, 13+i*3, 12, fill='#00ffcc', outline='') for i in range(12)]
        self.root.update()

    def _update_ui(self):
        if self.use_ui:
            if self.is_listening:
                norm = self.level / max(0.001, self.max_level)
                now = time.time()
                for i, bar in enumerate(self.bars):
                    target_h = max(2, min(2 + (norm * 4) + (math.sin(now * 12 + i * 0.5) * 2), 6))
                    coords = self.canvas.coords(bar)
                    self.canvas.coords(bar, coords[0], 12 - target_h/2, coords[2], 12 + target_h/2)
            self.root.after(50, self._update_ui)

    def on_transcription(self, full_text, segments):
        self.current_segments = segments

    def toggle_listening(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.base_segment_count = len(self.current_segments)
            print('\n[MIC]: ON - Speak now...')
            if self.use_ui: self.root.attributes('-alpha', 1.0)
        else:
            print('\n[MIC]: OFF - Processing...')
            if self.use_ui: 
                self.root.attributes('-alpha', 0.0)
                self.root.after(600, self._finish_and_type)
            else:
                # No UI: Finish and type immediately (with small sync delay)
                time.sleep(0.6)
                self._finish_and_type()

    def _finish_and_type(self):
        new_segments = self.current_segments[self.base_segment_count:]
        if new_segments:
            text = " ".join([s['text'].strip() for s in new_segments if s['text'].strip()])
            if text:
                self.keyboard_controller.type(text + ' ')
                print(f'[TYPED]: {text}')

    def run(self):
        def on_press(key):
            if key == keyboard.Key.shift_r:
                if self.use_ui:
                    self.root.after(0, self.toggle_listening)
                else:
                    self.toggle_listening()
        
        keyboard.Listener(on_press=on_press).start()
        threading.Thread(target=self.client, daemon=True).start()
        
        if self.use_ui:
            self._update_ui()
            print('[INFO]: High-Accuracy Ready (UI Enabled) - RIGHT SHIFT to toggle')
            self.root.mainloop()
        else:
            print('[INFO]: High-Accuracy Ready (Headless Mode) - RIGHT SHIFT to toggle')
            while True: time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', '-s', type=str, default='localhost')
    parser.add_argument('--port', '-p', type=int, default=9095)
    parser.add_argument('--model', '-m', type=str, default='small')
    parser.add_argument('--lang', '-l', type=str, default='en')
    parser.add_argument('--ui', action='store_true', help='Enable the floating visualizer UI')
    args = parser.parse_args()
    
    if ':' in args.server:
        host, port = args.server.split(':')
        args.server, args.port = host, int(port)
        
    app = KeyboardTranscriptionClient(args.server, args.port, args.model, args.lang, use_ui=args.ui)
    app.run()
