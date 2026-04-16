from RealtimeSTT import AudioToTextRecorder
import pyaudio
import numpy as np

from .mill_commands import MillCommands

class AudioProcess:
    RATE=48000
    CHUNK = 480





    # Priming-Prompt wird aus MillCommands generiert (single source of truth)
    _INITIAL_PROMPT = MillCommands().build_initial_prompt()

    def __init__(self, cmd_q, model: str = "tiny"):
        self.cmd_q = cmd_q
        # compute_type muss zum device passen: CUDA-Builds unterstützen int8 nicht
        # zuverlaessig -> int8_float16. Fuer CPU bleibt int8 die richtige Wahl.
        import torch  # lokaler Import, damit Module-Import billig bleibt
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "int8_float16"
        else:
            device = "cpu"
            compute_type = "int8"
        self.recorder = AudioToTextRecorder(
            language="de",              # Sprache
            compute_type=compute_type,
            device=device,
            model = model,              # Großes Model, langsamer aber besser
            use_microphone=True,     # Disable built-in microphone usage
            spinner=False,           # eigene Statusmeldungen stattdessen
            enable_realtime_transcription=True, # für echtzeit
            use_main_model_for_realtime = True, # für echtzeit
            silero_deactivity_detection=True,   # robuster gegenüber störgeräuschen
            post_speech_silence_duration=0.2,
            initial_prompt=self._INITIAL_PROMPT,
            on_recording_start=lambda: print("\n[MIC] Aufnahme läuft...    ", flush=True),
            on_recording_stop=lambda: print("[MIC] Aufnahme gestoppt, verarbeite...", flush=True),
        )
        self.p = pyaudio.PyAudio()

    def setupAudioDevice(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            #input_device_index=1
        )

            #print("Speak now\n")
        try:
            while True:
                # Read audio data from the stream (in the expected format)
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self.recorder.feed_audio(data)
        except Exception as e:
            print(f"\nsetupAudio_thread encountered an error: {e}")

    def recorder_transcription_thread(self):
        """Thread function to handle transcription and process the text."""
        def process_text(full_sentence):
            """Callback function to process the transcribed text."""
            print(f"[STT] {full_sentence}", flush=True)
            self.cmd_q.put(full_sentence)
        try:
            while True:
                # Get transcribed text and process it using the callback
                self.recorder.text(process_text)
        except Exception as e:
            print(f"\ntranscription_thread encountered an error: {e}")
