import os
import shutil
import wave

import logging
import numpy as np
import sounddevice as sd
import threading
import json
import websocket
import uuid
import time
try:
    import av
except ImportError:
    av = None
import whisper_live.utils as utils


class Client:
    """
    Handles communication with a server using WebSocket.
    """
    INSTANCES = {}
    END_OF_AUDIO = "END_OF_AUDIO"

    def __init__(
        self,
        host=None,
        port=None,
        lang=None,
        translate=False,
        model="small",
        srt_file_path="output.srt",
        use_vad=True,
        use_wss=False,
        log_transcription=True,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        transcription_callback=None,
        enable_translation=False,
        target_language="fr",
        translation_callback=None,
        translation_srt_file_path="output_translated.srt",
    ):
        self.recording = False
        self.task = "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_received = None
        self.disconnect_if_no_response_for = 15
        self.language = lang
        self.model = model
        self.server_error = False
        self.srt_file_path = srt_file_path
        self.use_vad = use_vad
        self.use_wss = use_wss
        self.last_segment = None
        self.last_received_segment = None
        self.log_transcription = log_transcription
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        self.transcription_callback = transcription_callback

        self.enable_translation = enable_translation
        self.target_language = target_language
        self.translation_callback = translation_callback
        self.translation_srt_file_path = translation_srt_file_path
        self.last_translated_segment = None
        if translate:
            self.task = "translate"

        self.audio_bytes = None

        if host is not None and port is not None:
            socket_protocol = 'wss' if self.use_wss else "ws"
            socket_url = f"{socket_protocol}://{host}:{port}"
            self.client_socket = websocket.WebSocketApp(
                socket_url,
                on_open=lambda ws: self.on_open(ws),
                on_message=lambda ws, message: self.on_message(ws, message),
                on_error=lambda ws, error: self.on_error(ws, error),
                on_close=lambda ws, close_status_code, close_msg: self.on_close(
                    ws, close_status_code, close_msg
                ),
            )
        else:
            print("[ERROR]: No host or port specified.")
            return

        Client.INSTANCES[self.uid] = self

        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self.transcript = []
        self.translated_transcript = []
        print("[INFO]: * recording")

    def handle_status_messages(self, message_data):
        status = message_data["status"]
        if status == "WAIT":
            self.waiting = True
            print(f"[INFO]: Server is full. Estimated wait time {round(message_data['message'])} minutes.")
        elif status == "ERROR":
            print(f"Message from Server: {message_data['message']}")
            self.server_error = True
        elif status == "WARNING":
            print(f"Message from Server: {message_data['message']}")

    def process_segments(self, segments, translated=False):
        text = []
        for i, seg in enumerate(segments):
            if not text or text[-1] != seg["text"]:
                text.append(seg["text"].strip())
                if i == len(segments) - 1 and not seg.get("completed", False):
                    self.last_segment = seg
                elif self.server_backend == "faster_whisper" and seg.get("completed", False):
                    if translated:
                        if (not self.translated_transcript or float(seg['start']) >= float(self.translated_transcript[-1]['end'])):
                            self.translated_transcript.append(seg)
                    else:
                        if (not self.transcript or float(seg['start']) >= float(self.transcript[-1]['end'])):
                            self.transcript.append(seg)
        if not translated:
            if self.last_received_segment is None or self.last_received_segment != segments[-1]["text"]:
                self.last_response_received = time.time()
                self.last_received_segment = segments[-1]["text"]

        if translated:
            if self.translation_callback and callable(self.translation_callback):
                try:
                    self.translation_callback(" ".join(text), segments)
                except Exception as e:
                    print(f"[WARN] translation_callback raised: {e}")
                return
        else:
            if self.transcription_callback and callable(self.transcription_callback):
                try:
                    self.transcription_callback(" ".join(text), segments)
                except Exception as e:
                    print(f"[WARN] transcription_callback raised: {e}")
                return
        
        if self.log_transcription:
            original_text = [seg["text"] for seg in self.transcript[-4:]]
            if self.last_segment is not None and self.last_segment["text"] not in original_text:
                original_text.append(self.last_segment["text"])
            
            utils.clear_screen()
            utils.print_transcript(original_text)
            if self.enable_translation:
                print(f"\n\nTRANSLATION to {self.target_language}:")
                utils.print_transcript([seg["text"] for seg in self.translated_transcript[-4:]], translated=True)

    def on_message(self, ws, message):
        message = json.loads(message)

        if self.uid != message.get("uid"):
            print("[ERROR]: invalid client uid")
            return

        if "status" in message.keys():
            self.handle_status_messages(message)
            return

        if "message" in message.keys() and message["message"] == "DISCONNECT":
            print("[INFO]: Server disconnected due to overtime.")
            self.recording = False

        if "message" in message.keys() and message["message"] == "SERVER_READY":
            self.last_response_received = time.time()
            self.recording = True
            self.server_backend = message["backend"]
            print(f"[INFO]: Server Running with backend {self.server_backend}")
            return

        if "language" in message.keys():
            self.language = message.get("language")
            lang_prob = message.get("language_prob")
            print(
                f"[INFO]: Server detected language {self.language} with probability {lang_prob}"
            )
            return

        if "segments" in message.keys():
            self.process_segments(message["segments"])
        
        if "translated_segments" in message.keys():
            self.process_segments(message["translated_segments"], translated=True)

    def on_error(self, ws, error):
        print(f"[ERROR] WebSocket Error: {error}")
        self.server_error = True
        self.error_message = error

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[INFO]: Websocket connection closed: {close_status_code}: {close_msg}")
        self.recording = False
        self.waiting = False

    def on_open(self, ws):
        print("[INFO]: Opened connection")
        ws.send(
            json.dumps(
                {
                    "uid": self.uid,
                    "language": self.language,
                    "task": self.task,
                    "model": self.model,
                    "use_vad": self.use_vad,
                    "send_last_n_segments": self.send_last_n_segments,
                    "no_speech_thresh": self.no_speech_thresh,
                    "clip_audio": self.clip_audio,
                    "same_output_threshold": self.same_output_threshold,
                    "enable_translation": self.enable_translation,
                    "target_language": self.target_language,
                }
            )
        )

    def send_packet_to_server(self, message):
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(e)

    def close_websocket(self):
        try:
            self.client_socket.close()
        except Exception as e:
            print("[ERROR]: Error closing WebSocket:", e)

        try:
            self.ws_thread.join()
        except Exception as e:
            print("[ERROR:] Error joining WebSocket thread:", e)

    def get_client_socket(self):
        return self.client_socket

    def write_srt_file(self, output_path="output.srt"):
        if self.server_backend == "faster_whisper":
            if not self.transcript and self.last_segment is not None:
                self.transcript.append(self.last_segment)
            elif self.last_segment and self.transcript[-1]["text"] != self.last_segment["text"]:
                self.transcript.append(self.last_segment)
            utils.create_srt_file(self.transcript, output_path)

        if self.enable_translation:
            utils.create_srt_file(self.translated_transcript, self.translation_srt_file_path)

    def wait_before_disconnect(self):
        assert self.last_response_received
        while time.time() - self.last_response_received < self.disconnect_if_no_response_for:
            continue


class TranscriptionTeeClient:
    def __init__(self, clients, save_output_recording=False, output_recording_filename="./output_recording.wav", mute_audio_playback=False):
        self.clients = clients
        if not self.clients:
            raise Exception("At least one client is required.")
        self.chunk = 4096
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 60000
        self.save_output_recording = save_output_recording
        self.output_recording_filename = output_recording_filename
        self.mute_audio_playback = mute_audio_playback
        self.frames = b""
        self.stream = None

    def __call__(self, audio=None, rtsp_url=None, hls_url=None, save_file=None):
        assert sum(
            source is not None for source in [audio, rtsp_url, hls_url]
        ) <= 1, 'You must provide only one selected source'

        print("[INFO]: Waiting for server ready ...")
        for client in self.clients:
            while not client.recording:
                if client.waiting or client.server_error:
                    self.close_all_clients()
                    return

        print("[INFO]: Server Ready!")
        if hls_url is not None:
            self.process_hls_stream(hls_url, save_file)
        elif audio is not None:
            resampled_file = utils.resample(audio)
            self.play_file(resampled_file)
        elif rtsp_url is not None:
            self.process_rtsp_stream(rtsp_url)
        else:
            self.record()

    def close_all_clients(self):
        for client in self.clients:
            client.close_websocket()

    def write_all_clients_srt(self):
        for client in self.clients:
            client.write_srt_file(client.srt_file_path)

    def multicast_packet(self, packet, unconditional=False):
        for client in self.clients:
            if (unconditional or client.recording):
                client.send_packet_to_server(packet)

    def play_file(self, filename):
        with wave.open(filename, "rb") as wavfile:
            chunk_duration = self.chunk / float(wavfile.getframerate())
            
            def callback(outdata, frames, time, status):
                data = wavfile.readframes(frames)
                if data == b"":
                    raise sd.CallbackStop
                audio_array = self.bytes_to_float_array(data)
                self.multicast_packet(audio_array.tobytes())
                if not self.mute_audio_playback:
                    out_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    outdata[:] = out_data.reshape(-1, 1)
                else:
                    outdata.fill(0)

            try:
                stream = sd.OutputStream(
                    samplerate=wavfile.getframerate(),
                    channels=wavfile.getnchannels(),
                    dtype='float32',
                    callback=callback,
                    blocksize=self.chunk
                )
                with stream:
                    while any(client.recording for client in self.clients) and stream.active:
                        time.sleep(0.1)

                for client in self.clients:
                    client.wait_before_disconnect()
                self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
                self.write_all_clients_srt()
                self.close_all_clients()

            except KeyboardInterrupt:
                self.close_all_clients()
                self.write_all_clients_srt()
                print("[INFO]: Keyboard interrupt.")

    def process_rtsp_stream(self, rtsp_url):
        print("[INFO]: Connecting to RTSP stream...")
        try:
            container = av.open(rtsp_url, format="rtsp", options={"rtsp_transport": "tcp"})
            self.process_av_stream(container, stream_type="RTSP")
        except Exception as e:
            print(f"[ERROR]: Failed to process RTSP stream: {e}")
        finally:
            for client in self.clients:
                client.wait_before_disconnect()
            self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
            self.close_all_clients()
            self.write_all_clients_srt()
        print("[INFO]: RTSP stream processing finished.")

    def process_hls_stream(self, hls_url, save_file=None):
        print("[INFO]: Connecting to HLS stream...")
        try:
            container = av.open(hls_url, format="hls")
            self.process_av_stream(container, stream_type="HLS", save_file=save_file)
        except Exception as e:
            print(f"[ERROR]: Failed to process HLS stream: {e}")
        finally:
            for client in self.clients:
                client.wait_before_disconnect()
            self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
            self.close_all_clients()
            self.write_all_clients_srt()
        print("[INFO]: HLS stream processing finished.")

    def process_av_stream(self, container, stream_type, save_file=None):
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if not audio_stream:
            print(f"[ERROR]: No audio stream found in {stream_type} source.")
            return

        output_container = None
        if save_file:
            output_container = av.open(save_file, mode="w")
            output_audio_stream = output_container.add_stream(codec_name="pcm_s16le", rate=self.rate)

        try:
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    audio_data = frame.to_ndarray().tobytes()
                    self.multicast_packet(audio_data)

                    if save_file:
                        output_container.mux(frame)
        except Exception as e:
            print(f"[ERROR]: Error during {stream_type} stream processing: {e}")
        finally:
            time.sleep(5)
            self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
            if output_container:
                output_container.close()
            container.close()

    def save_chunk(self, n_audio_file):
        t = threading.Thread(
            target=self.write_audio_frames_to_file,
            args=(self.frames[:], f"chunks/{n_audio_file}.wav",),
        )
        t.start()

    def finalize_recording(self, n_audio_file):
        if self.save_output_recording and len(self.frames):
            self.write_audio_frames_to_file(
                self.frames[:], f"chunks/{n_audio_file}.wav"
            )
            n_audio_file += 1
        self.close_all_clients()
        if self.save_output_recording:
            self.write_output_recording(n_audio_file)
        self.write_all_clients_srt()

    def record(self):
        n_audio_file = 0
        if self.save_output_recording:
            if os.path.exists("chunks"):
                shutil.rmtree("chunks")
            os.makedirs("chunks")
        
        def callback(indata, frames, time, status):
            if not any(client.recording for client in self.clients):
                raise sd.CallbackStop
            
            data = (indata * 32768).astype(np.int16).tobytes()
            self.frames += data
            audio_array = self.bytes_to_float_array(data)
            self.multicast_packet(audio_array.tobytes())

        try:
            stream = sd.InputStream(
                samplerate=self.rate,
                channels=self.channels,
                dtype='float32',
                callback=callback,
                blocksize=self.chunk
            )
            with stream:
                while any(client.recording for client in self.clients):
                    if len(self.frames) > 60 * self.rate:
                        if self.save_output_recording:
                            self.save_chunk(n_audio_file)
                            n_audio_file += 1
                        self.frames = b""
                    time.sleep(0.1)

            self.write_all_clients_srt()

        except KeyboardInterrupt:
            self.finalize_recording(n_audio_file)

    def write_audio_frames_to_file(self, frames, file_name):
        with wave.open(file_name, "wb") as wavfile:
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            wavfile.writeframes(frames)

    def write_output_recording(self, n_audio_file):
        input_files = [
            f"chunks/{i}.wav"
            for i in range(n_audio_file)
            if os.path.exists(f"chunks/{i}.wav")
        ]
        with wave.open(self.output_recording_filename, "wb") as wavfile:
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            for in_file in input_files:
                with wave.open(in_file, "rb") as wav_in:
                    while True:
                        data = wav_in.readframes(self.chunk)
                        if data == b"":
                            break
                        wavfile.writeframes(data)
                os.remove(in_file)
        if os.path.exists("chunks"):
            shutil.rmtree("chunks")

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0


class TranscriptionClient(TranscriptionTeeClient):
    def __init__(
        self,
        host,
        port,
        lang=None,
        translate=False,
        model="small",
        use_vad=True,
        use_wss=False,
        save_output_recording=False,
        output_recording_filename="./output_recording.wav",
        output_transcription_path="./output.srt",
        log_transcription=True,
        mute_audio_playback=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        transcription_callback=None,
        enable_translation=False,
        target_language="fr",
        translation_callback=None,
        translation_srt_file_path="./output_translated.srt",
    ):
        self.client = Client(
            host,
            port,
            lang,
            translate,
            model,
            srt_file_path=output_transcription_path,
            use_vad=use_vad,
            use_wss=use_wss,
            log_transcription=log_transcription,
            send_last_n_segments=send_last_n_segments,
            no_speech_thresh=no_speech_thresh,
            clip_audio=clip_audio,
            same_output_threshold=same_output_threshold,
            transcription_callback=transcription_callback,
            enable_translation=enable_translation,
            target_language=target_language,
            translation_callback=translation_callback,
            translation_srt_file_path=translation_srt_file_path,
        )

        if save_output_recording and not output_recording_filename.endswith(".wav"):
            raise ValueError(f"Please provide a valid `output_recording_filename`: {output_recording_filename}")
        if not output_transcription_path.endswith(".srt"):
            raise ValueError(f"Please provide a valid `output_transcription_path`: {output_transcription_path}. The file extension should be `.srt`.")
        if not translation_srt_file_path.endswith(".srt"):
            raise ValueError(f"Please provide a valid `translation_srt_file_path`: {translation_srt_file_path}. The file extension should be `.srt`.")
        TranscriptionTeeClient.__init__(
            self,
            [self.client],
            save_output_recording=save_output_recording,
            output_recording_filename=output_recording_filename,
            mute_audio_playback=mute_audio_playback
        )