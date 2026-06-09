"""BLE chunk framing for payloads larger than the 512-byte GATT limit."""

from __future__ import annotations

# Payload bytes per chunk (header "@i/n@" adds a few bytes; stay under 512 total)
CHUNK_PAYLOAD_SIZE = 480
CHUNK_HEADER_PREFIX = "@"
CHUNK_HEADER_SUFFIX = "@"


def split_payload(payload: str) -> list[bytes]:
    """Split UTF-8 payload into framed BLE chunks."""
    encoded = payload.encode("utf-8")
    if len(encoded) <= 512:
        return [encoded]

    chunks: list[bytes] = []
    total = (len(encoded) + CHUNK_PAYLOAD_SIZE - 1) // CHUNK_PAYLOAD_SIZE
    for index in range(total):
        start = index * CHUNK_PAYLOAD_SIZE
        piece = encoded[start : start + CHUNK_PAYLOAD_SIZE]
        header = f"{CHUNK_HEADER_PREFIX}{index}/{total}{CHUNK_HEADER_SUFFIX}".encode(
            "utf-8"
        )
        chunks.append(header + piece)
    return chunks


def merge_chunks(chunks: dict[int, bytes]) -> bytes:
    """Merge indexed chunk payloads (without headers) in order."""
    if not chunks:
        return b""
    total = len(chunks)
    parts = [chunks[i] for i in range(total)]
    return b"".join(parts)
