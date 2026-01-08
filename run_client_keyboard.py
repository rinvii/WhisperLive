import sys
import argparse
import time
from pynput import keyboard
from pynput.keyboard import Controller, Key
from whisper_live.client import TranscriptionClient

class KeyboardTranscriptionClient:
    def __init__(self, host, port, model, lang):
        self.keyboard_controller = Controller()
        self.is_listening = False
        self.base_segment_count = 0
        self.current_segments = []
        
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
        
        # Patch the client's multicast_packet to respect our toggle
        original_multicast = self.client.multicast_packet
        def gated_multicast(packet, unconditional=False):
            if self.is_listening or unconditional:
                original_multicast(packet, unconditional)
        self.client.multicast_packet = gated_multicast

    def on_transcription(self, full_text, segments):
        self.current_segments = segments

    def toggle_listening(self):
        self.is_listening = not self.is_listening
        
        if self.is_listening:
            # Capture how many segments existed before we started this burst
            self.base_segment_count = len(self.current_segments)
            print('\n[MIC]: ON - Speak now...')
        else:
            print('\n[MIC]: OFF - Processing...')
            # Small delay to allow the last chunk to reach on_transcription
            time.sleep(0.4)
            
            # Type only the NEW segments
            new_segments = self.current_segments[self.base_segment_count:]
            if new_segments:
                text_to_type = " ".join([s['text'].strip() for s in new_segments if s['text'].strip()])
                if text_to_type:
                    self.keyboard_controller.type(text_to_type + ' ')
                    print(f'[TYPED]: {text_to_type}')
                else:
                    print('[INFO]: No speech detected in segments.')
            else:
                print('[INFO]: No segments captured.')

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
    parser.add_argument('--server', '-s', type=str, default='localhost')
    parser.add_argument('--port', '-p', type=int, default=9095)
    parser.add_argument('--model', '-m', type=str, default='small')
    parser.add_argument('--lang', '-l', type=str, default='en')
    args = parser.parse_args()

    # Handle IP:PORT format in --server
    if ':' in args.server:
        host, port = args.server.split(':')
        args.server = host
        args.port = int(port)
        print(f'[INFO]: Using server {args.server} on port {args.port}')

    app = KeyboardTranscriptionClient(args.server, args.port, args.model, args.lang)
    app.run()
