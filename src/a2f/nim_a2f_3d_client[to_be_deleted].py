import argparse, asyncio
import a2f_3d.client.auth
import a2f_3d.client.service
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                        description="Sample python application to send audio and receive animation data and emotion data through the Audio2Face-3D API.",
                        epilog="NVIDIA CORPORATION.  All rights reserved.")
    parser.add_argument("file", help="PCM-16 bits single channel audio file in WAV ccontainer to be sent to the Audio2Face-3D service")
    parser.add_argument("config", help="Configuration file for inference models")
    parser.add_argument("--apikey", type=str, required=True, help="NGC API Key to invoke the API function")
    parser.add_argument("--function-id", type=str, required=True, default="", help="Function ID to invoke the API function")
    return parser.parse_args()

async def main():
    args = parse_args()

    metadata_args = [("function-id", args.function_id), ("authorization", "Bearer " + args.apikey)]
    # Open gRPC channel and get Audio2Face-3D stub
    channel = a2f_3d.client.auth.create_channel(uri="grpc.nvcf.nvidia.com:443", use_ssl=True, metadata=metadata_args)
            
    stub = A2FControllerServiceStub(channel)

    stream = stub.ProcessAudioStream()
    write = asyncio.create_task(a2f_3d.client.service.write_to_stream(stream, args.config, args.file))
    read = asyncio.create_task(a2f_3d.client.service.read_from_stream(stream))

    await write
    await read

if __name__ == "__main__":
    asyncio.run(main())
