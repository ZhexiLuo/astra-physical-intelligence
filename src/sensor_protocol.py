import struct
from typing import BinaryIO


def send_packet(stream: BinaryIO, payload: bytes) -> None:
    stream.write(struct.pack(">Q", len(payload)))
    stream.write(payload)
    stream.flush()


def receive_packet(stream: BinaryIO) -> bytes | None:
    header = stream.read(8)
    if not header:
        return None
    return stream.read(struct.unpack(">Q", header)[0])
