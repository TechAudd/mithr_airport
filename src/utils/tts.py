from elevenlabs.client import ElevenLabs
from elevenlabs import play
import io
import os

elabs_key = os.getenv("ELEVENLABS_API_KEY")
if elabs_key is None:
    raise ValueError("Please set the ELEVENLABS_API_KEY environment variable.")
client = ElevenLabs(api_key=elabs_key)

def botspeak(text, save_file='default.mp3'):
    # audio_stream = client.text_to_speech.stream(
    #     text=text,
    #     voice_id="JBFqnCBsd6RMkjVDRZzb",
    #     model_id="eleven_multilingual_v2",
    #     output_format="mp3_44100_128"
    # )
    # audio_data = b''
    # for chunk in audio_stream:
    #     audio_data += chunk
    # if save_file:
    #     with open(save_file, 'wb') as f:
    #         f.write(audio_data)
    # print(f"Audio saved to {save_file}" if save_file else "Audio not saved.")
    print("Bot: ", text)
    return text
    # play(io.BytesIO(audio_data))