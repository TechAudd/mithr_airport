import shutil
import io
import os
import asyncio
import time
from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import FileResponse
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
import wave
import numpy as np
from scipy.io import wavfile

import a2f.a2f_3d.client.auth as a2f_3d_auth
import a2f.a2f_3d.client.service as a2f_3d_service

a2f_router = APIRouter(prefix='/a2f')

elabs_key = os.getenv("ELEVENLABS_API_KEY")
if elabs_key is None:
    raise ValueError("Please set the ELEVENLABS_API_KEY environment variable.")
client = ElevenLabs(api_key=elabs_key)

def cleanup_files(*paths):
    for path in paths:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
        except Exception as e:
            print(f"Error cleaning up {path}: {e}")

@a2f_router.post("/text2animation")
async def process_audio_to_animation(
    text: str,
    function_id: str = "0961a6da-fb9e-4f2e-8491-247e5fd7bf8d",
    uri: str = "grpc.nvcf.nvidia.com:443",
    config_file: str = "a2f/config/config_claire.yml",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    start_time = time.perf_counter()
    audio_stream = client.text_to_speech.stream(
        text=text,
        voice_id="cgSgspJ2msm6clMCkdW9",
        model_id="eleven_multilingual_v2",
        output_format="pcm_24000",
    )
    end_time = time.perf_counter()
    print(f"Audio generation took {end_time - start_time:.6f} seconds")
    audio_data = b''
    start_time = time.perf_counter()
    for chunk in audio_stream:
        audio_data += chunk
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)

    buffer.seek(0)
    rate, data = wavfile.read(buffer)
    end_time = time.perf_counter()
    print(f"Audio export took {end_time - start_time:.6f} seconds")
    apikey = os.getenv("NVIDIA_NIM_API_KEY")
    metadata_args = [
        ("function-id", function_id),
        ("authorization", "Bearer " + apikey)
    ]
    start_time = time.perf_counter()
    channel = a2f_3d_auth.create_channel(uri=uri, use_ssl=True, metadata=metadata_args)
    stub = A2FControllerServiceStub(channel)
    end_time = time.perf_counter()
    print(f"Channel creation took {end_time - start_time:.6f} seconds")
    start_time = time.perf_counter()
    stream = stub.ProcessAudioStream()
    write = asyncio.create_task(a2f_3d_service.write_to_stream(stream, config_file, data=data, samplerate=rate))
    read = asyncio.create_task(a2f_3d_service.read_from_stream(stream))

    await write
    await read
    end_time = time.perf_counter()
    print(f"Stream processing took {end_time - start_time:.6f} seconds")

    path = read.result()
    if path:
        zip_path = shutil.make_archive("animation", 'zip', path)
        background_tasks.add_task(cleanup_files, zip_path, path)
        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=os.path.basename(zip_path)
        )
