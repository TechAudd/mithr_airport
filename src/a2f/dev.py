import shutil
import pdb
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pydub import AudioSegment
import io
import os
import asyncio
import a2f_3d.client.auth
import a2f_3d.client.service
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/text2animation")
async def process_audio_to_animation(
    text: str,
    function_id: str = "0961a6da-fb9e-4f2e-8491-247e5fd7bf8d",
    uri: str = "grpc.nvcf.nvidia.com:443",
    config_file: str = "config/config_claire.yml",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    audio_stream = client.text_to_speech.stream(
        text=text,
        voice_id="cgSgspJ2msm6clMCkdW9",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    audio_data = b''
    for chunk in audio_stream:
        audio_data += chunk
    filename = uuid4().hex
    save_file = f"{filename}.wav"
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
    audio.export(save_file, format="wav")
    apikey = os.getenv("NVIDIA_NIM_API_KEY")
    metadata_args = [
        ("function-id", function_id),
        ("authorization", "Bearer " + apikey)
    ]
    channel = a2f_3d.client.auth.create_channel(uri=uri, use_ssl=True, metadata=metadata_args)
    stub = A2FControllerServiceStub(channel)

    stream = stub.ProcessAudioStream()
    write = asyncio.create_task(a2f_3d.client.service.write_to_stream(stream, config_file, save_file))
    read = asyncio.create_task(a2f_3d.client.service.read_from_stream(stream))

    await write
    await read

    path = read.result()
    if path:
        zip_path = shutil.make_archive("animation", 'zip', path)
        background_tasks.add_task(cleanup_files, zip_path, path, save_file)
        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=os.path.basename(zip_path)
        )

