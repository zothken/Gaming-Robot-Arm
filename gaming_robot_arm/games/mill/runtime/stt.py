from RealtimeSTT import AudioToTextRecorder
import pyaudio
import numpy as np

class AudioProcess:
    RATE=48000
    CHUNK = 480







    # Primed Whisper auf deutsche Mühlenvokabular (A/B/C + Zahlen als Brettsymbole)
    _INITIAL_PROMPT = (
        "Mühle Brettspiel. Positionen: A1, A2, A3, A4, A5, A6, A7, A8, "
        "B1, B2, B3, B4, B5, B6, B7, B8, C1, C2, C3, C4, C5, C6, C7, C8. "
        "Befehle: nach, von, setze, schlage."
    )

    def __init__(self, cmd_q, model: str = "tiny"):
        self.cmd_q = cmd_q
        self.recorder = AudioToTextRecorder(
            language="de",              # Sprache
            compute_type="int8",
            device = "cuda",        # cuda für Nutzung der Grafikkarte
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
