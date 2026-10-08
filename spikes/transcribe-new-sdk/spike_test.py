"""Spike: verify aws-sdk-transcribe-streaming works end-to-end against live Transcribe.

Not production code -- just proving the new Smithy-based SDK actually works
before considering it for a Pipecat STT service implementation.
"""

import asyncio
import time
import wave
from pathlib import Path

from smithy_http.aio.crt import AWSCRTHTTPClient

from aws_sdk_transcribe_streaming.client import AsyncTranscribeStreamingClient
from aws_sdk_transcribe_streaming.config import AsyncTranscribeStreamingConfig
from aws_sdk_transcribe_streaming.models import (
    AudioEvent,
    AudioStreamAudioEvent,
    LanguageCode,
    MediaEncoding,
    StartStreamTranscriptionInput,
    TranscriptResultStreamTranscriptEvent,
)

AUDIO_FILE = Path("/tmp/pipecat/scripts/provider-watch/assets/speech-16k.wav")
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
CHANNELS = 1
CHUNK_FRAMES = 1_600  # 100 ms of audio

results = []


async def send_audio(stream) -> None:
    loop = asyncio.get_running_loop()
    started = loop.time()
    audio_seconds = 0.0
    chunks_sent = 0

    try:
        with wave.open(str(AUDIO_FILE), "rb") as source:
            audio_format = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            )
            print(f"[spike] source wav format: {audio_format}")
            while chunk := source.readframes(CHUNK_FRAMES):
                await stream.input_stream.send(AudioStreamAudioEvent(value=AudioEvent(audio_chunk=chunk)))
                chunks_sent += 1
                audio_seconds += len(chunk) / (BYTES_PER_SAMPLE * SAMPLE_RATE * CHANNELS)
                delay = started + audio_seconds - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
        print(f"[spike] sent {chunks_sent} audio chunks ({audio_seconds:.2f}s of audio)")
    finally:
        await stream.input_stream.close()


async def print_transcripts(stream) -> None:
    _, output_stream = await stream.await_output()
    if output_stream is None:
        raise RuntimeError("The service returned no output stream")

    async for event in output_stream:
        if not isinstance(event, TranscriptResultStreamTranscriptEvent):
            print(f"[spike] non-transcript event: {type(event).__name__}")
            continue

        transcript = event.value.transcript
        if transcript is None:
            continue
        for result in transcript.results or []:
            alternatives = result.alternatives or []
            if alternatives and alternatives[0].transcript:
                kind = "FINAL" if not result.is_partial else "partial"
                text = alternatives[0].transcript
                print(f"[spike] [{kind}] {text}")
                results.append((kind, text))


async def main() -> None:
    t0 = time.time()
    config = await AsyncTranscribeStreamingConfig.resolve(
        region="us-east-1",
        transport=AWSCRTHTTPClient(),
    )
    print("[spike] config resolved")
    async with AsyncTranscribeStreamingClient(config=config) as client:
        print("[spike] client created, starting stream transcription")
        stream = await client.start_stream_transcription(
            input=StartStreamTranscriptionInput(
                language_code=LanguageCode.EN_US,
                media_sample_rate_hertz=SAMPLE_RATE,
                media_encoding=MediaEncoding.PCM,
            )
        )
        print(f"[spike] stream started after {time.time() - t0:.2f}s")

        async with stream:
            await asyncio.gather(
                send_audio(stream),
                print_transcripts(stream),
            )

    print(f"[spike] total elapsed: {time.time() - t0:.2f}s")
    print(f"[spike] total transcript events captured: {len(results)}")
    finals = [t for k, t in results if k == "FINAL"]
    print(f"[spike] FINAL transcript(s): {finals}")


if __name__ == "__main__":
    asyncio.run(main())
