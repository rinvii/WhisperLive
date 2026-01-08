import sys
import argparse
from pynput import keyboard
from pynput.keyboard import Controller, Key
from whisper_live.client import TranscriptionClient

class KeyboardTranscriptionClient:
    def __init__(self, host, port, model, lang):
        self.keyboard_controller = Controller()
        self.is_listening = False
        self.accumulated_text = ''
        self.last_full_text = ''
        self.start_offset = 0
        
        print(f'[INFO]: Initializing Keyboard Client...')
        print(f'[INFO]: Press RIGHT SHIFT to Start/Stop Listening')
        
        self.client = TranscriptionClient(
            host,
            port,
            lang=lang,
            model=model,
            log_transcription=False,
            transcription_callback=self.on_transcription
        )
        
        original_multicast = self.client.multicast_packet
        def gated_multicast(packet, unconditional=False):
            if self.is_listening or unconditional:
                original_multicast(packet, unconditional)
        self.client.multicast_packet = gated_multicast

    def on_transcription(self, full_text, segments):
        self.last_full_text = full_text
        if self.is_listening:
            # Only take the text that has appeared since we started this burst
            self.accumulated_text = full_text[self.start_offset:].strip()

    def toggle_listening(self):
        self.is_listening = not self.is_listening
        
        if self.is_listening:
            # Mark the current end of the global transcript as our starting point
            self.start_offset = len(self.last_full_text)
            self.accumulated_text = ''
            print('\n[MIC]: ON - Speak now...')
        else:
            print('\n[MIC]: OFF - Processing...')
            if self.accumulated_text:
                self.keyboard_controller.type(self.accumulated_text + ' ')
                print(f'[TYPED]: {self.accumulated_text}')
                # Update offset so we dont double-type if we start again immediately
                self.start_offset = len(self.last_full_text)
            else:
                print('[INFO]: No speech detected.')

    def run(self):
        def on_press(key):
            if key == keyboard.Key.shift_r:
                self.toggle_listening()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        
        try:
            self.client()
        except KeyboardInterrupt:
            print('\n[INFO]: Exiting...')
            sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', '-s', type=str, default='localhost:9095')
    parser.add_argument('--model', '-m', type=str, default='small')
    parser.add_argument('--lang', '-l', type=str, default='en')
    args = parser.parse_args()

    if ':' in args.server:
        host, port = args.server.split(':')
        port = int(port)
    else:
        host = args.server
        port = 9095

    app = KeyboardTranscriptionClient(host, port, args.model, args.lang)
    app.run()
