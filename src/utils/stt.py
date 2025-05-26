import torch
import numpy as np
import sounddevice as sd
import time
from scipy.io.wavfile import write as wav_write
import whisper
import tempfile
import wave
import pyaudio
import os

class SpeechToText:
    def __init__(self, whisper_model_size="small"):
        self.whisper_model = whisper.load_model(whisper_model_size)
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )

    def transcribe_audio(self, audio_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            result = self.whisper_model.transcribe(temp_file.name, language='en', fp16=False, verbose=False)
            os.unlink(temp_file.name)
            return result["text"].strip()

    def voice_input(self, text, filename='output.wav', max_record_seconds=10, silence_duration=1.0):
        SAMPLE_RATE = 16000
        FRAME_SAMPLES = 512
        SILERO_THRESHOLD = 0.5

        frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
        silence_frames = int(silence_duration * frames_per_second)

        print("Recording... Speak into the microphone.")

        audio_buffer = []
        silent_frames_count = 0
        start_time = time.time()
        is_recording = True

        def callback(indata, frames, time_info, status):
            nonlocal silent_frames_count, is_recording

            chunk = indata[:FRAME_SAMPLES, 0].copy()
            if len(chunk) < FRAME_SAMPLES:
                return

            chunk_int16 = (chunk * 32767).astype(np.int16)
            audio_buffer.append(chunk_int16.copy())

            input_tensor = torch.from_numpy(chunk_int16).unsqueeze(0)

            with torch.no_grad():
                speech_prob = self.vad_model(input_tensor, SAMPLE_RATE).item()

            if speech_prob < SILERO_THRESHOLD:
                silent_frames_count += 1
                if silent_frames_count >= silence_frames:
                    is_recording = False
            else:
                silent_frames_count = 0

            if (time.time() - start_time) >= max_record_seconds:
                is_recording = False

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=FRAME_SAMPLES,
            callback=callback
        ):
            while is_recording and (time.time() - start_time < max_record_seconds):
                time.sleep(0.01)  # Reduce CPU usage
        if audio_buffer:
            full_audio = np.concatenate(audio_buffer)
            transcribed_text = self.transcribe_audio(full_audio.tobytes())
            print(f"{text} {transcribed_text}")
            wav_write(filename, SAMPLE_RATE, full_audio)
            return transcribed_text
        else:
            print("No audio recorded")
            return ""
