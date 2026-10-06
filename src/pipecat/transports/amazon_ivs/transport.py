#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Amazon IVS Real-Time audio transport for Pipecat."""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from fractions import Fraction
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import aiohttp
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    BotConnectedFrame,
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.shared import acquires, releases

try:
    from aiortc import (
        AudioStreamTrack,
        RTCBundlePolicy,
        RTCConfiguration,
        RTCPeerConnection,
        RTCRtpSender,
        RTCSessionDescription,
        VideoStreamTrack,
    )
    from aiortc.mediastreams import MediaStreamError
    from av import AudioFrame, AudioResampler, VideoFrame
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'To use Amazon IVS, run `uv add "pipecat-ai[amazon-ivs]"` '
        'or `pip install "pipecat-ai[amazon-ivs]"`.'
    )
    raise Exception(f"Missing module: {e}")


PCM16_SAMPLE_BYTES = 2
NANOSECONDS_PER_SECOND = 1_000_000_000
IVS_WHIP_URL = "https://global.whip.live-video.net"
MAX_SDP_BYTES = 256 * 1024
MAX_SDP_LINES = 4_096
MAX_SDP_MEDIA_SECTIONS = 16
MAX_SDP_CANDIDATES = 256
PARTICIPANT_ID_PATTERN = re.compile(r"[A-Za-z0-9-]{1,64}")


class AmazonIVSError(RuntimeError):
    """Error raised for an Amazon IVS signalling or media failure."""


def require_ivs_url(value: str) -> str:
    """Validate an Amazon IVS endpoint before forwarding a participant token.

    Args:
        value: URL to validate.

    Returns:
        The validated URL.

    Raises:
        AmazonIVSError: If the URL is not a trusted Amazon IVS HTTPS endpoint.
    """
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError as error:
        raise AmazonIVSError("Amazon IVS endpoint used an untrusted destination") from error

    if (
        parsed.scheme != "https"
        or not hostname.endswith(".whip.live-video.net")
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        raise AmazonIVSError("Amazon IVS endpoint used an untrusted destination")
    return value


def safe_ivs_redirect(current_url: str, location: str) -> str:
    """Resolve and validate an Amazon IVS redirect.

    Args:
        current_url: Current trusted endpoint.
        location: Redirect location returned by Amazon IVS.

    Returns:
        The validated redirect URL.
    """
    return require_ivs_url(urljoin(current_url, location))


def require_ivs_subscription_url(value: str, participant_id: str) -> str:
    """Validate a WHEP endpoint for the selected IVS participant.

    Args:
        value: Full WHEP URL supplied by a trusted backend.
        participant_id: IVS participant to subscribe to.

    Returns:
        The validated subscription URL.

    Raises:
        AmazonIVSError: If the endpoint is untrusted or targets another participant.
    """
    if not PARTICIPANT_ID_PATTERN.fullmatch(participant_id):
        raise AmazonIVSError("Amazon IVS participant ID has an invalid format")
    validated = require_ivs_url(value)
    expected_suffix = f"/subscribe/{quote(participant_id, safe='')}"
    if not urlparse(validated).path.rstrip("/").endswith(expected_suffix):
        raise AmazonIVSError("Amazon IVS subscription endpoint targets another participant")
    return validated


def fix_ivs_answer_sdp(sdp: str) -> str:
    """Copy bundled ICE candidates into media sections that omit them.

    Args:
        sdp: SDP answer returned by Amazon IVS.

    Returns:
        An SDP answer accepted by aiortc.
    """
    if len(sdp.encode("utf-8")) > MAX_SDP_BYTES:
        raise AmazonIVSError("Amazon IVS SDP answer exceeded the size limit")
    lines = sdp.splitlines()
    if len(lines) > MAX_SDP_LINES:
        raise AmazonIVSError("Amazon IVS SDP answer exceeded the line limit")
    if sum(line.startswith("m=") for line in lines) > MAX_SDP_MEDIA_SECTIONS:
        raise AmazonIVSError("Amazon IVS SDP answer had too many media sections")
    candidates = [line for line in lines if line.startswith("a=candidate:")]
    if len(candidates) > MAX_SDP_CANDIDATES:
        raise AmazonIVSError("Amazon IVS SDP answer had too many ICE candidates")
    if not candidates:
        return sdp

    fixed: list[str] = []
    section: list[str] = []

    def flush() -> None:
        if not section:
            return
        if section[0].startswith("m=") and not any(
            line.startswith("a=candidate:") for line in section
        ):
            section.extend(candidates)
            section.append("a=end-of-candidates")
        fixed.extend(section)
        section.clear()

    for line in lines:
        if line.startswith("m="):
            flush()
        section.append(line)
    flush()
    result = "\r\n".join(fixed) + "\r\n"
    if len(result.encode("utf-8")) > MAX_SDP_BYTES:
        raise AmazonIVSError("Amazon IVS SDP answer expanded beyond the size limit")
    return result


def _pcm16_frame_width(num_channels: int) -> int:
    if num_channels < 1:
        raise ValueError("num_channels must be positive")
    return PCM16_SAMPLE_BYTES * num_channels


def _validate_pcm16(audio: bytes, num_channels: int) -> None:
    frame_width = _pcm16_frame_width(num_channels)
    if len(audio) % frame_width:
        raise ValueError(
            f"PCM16 byte length {len(audio)} is not aligned to {num_channels} channel(s)"
        )


def _av_frame_pts_nanoseconds(frame: AudioFrame) -> int | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return int(frame.pts * frame.time_base * NANOSECONDS_PER_SECOND)


def _pcm16_bytes_to_av_frame(
    audio: bytes,
    *,
    sample_rate: int,
    num_channels: int,
    pts_samples: int,
) -> AudioFrame:
    _validate_pcm16(audio, num_channels)
    layout = "mono" if num_channels == 1 else "stereo"
    samples = np.frombuffer(audio, dtype="<i2")
    frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout=layout)
    frame.sample_rate = sample_rate
    frame.pts = pts_samples
    frame.time_base = Fraction(1, sample_rate)
    return frame


class AmazonIVSIngressConverter:
    """Convert aiortc audio frames into Pipecat input frames.

    Args:
        sample_rate: Target Pipecat sample rate.
        num_channels: Target channel count.
        source: Transport source identifier applied to emitted frames.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        num_channels: int = 1,
        source: str | None = "amazon-ivs",
    ) -> None:
        """Initialize the ingress audio converter."""
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if num_channels not in (1, 2):
            raise ValueError("Amazon IVS supports mono or stereo PCM")

        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._source = source
        layout = "mono" if num_channels == 1 else "stereo"
        self._resampler = AudioResampler(format="s16", layout=layout, rate=sample_rate)

    def convert(self, frame: AudioFrame) -> list[InputAudioRawFrame]:
        """Convert one PyAV frame into zero or more Pipecat frames.

        Args:
            frame: PyAV audio frame received from aiortc.

        Returns:
            Converted Pipecat input frames.
        """
        converted: list[InputAudioRawFrame] = []
        for resampled in self._resampler.resample(frame):
            pcm = np.asarray(resampled.to_ndarray(), dtype="<i2").tobytes(order="C")
            _validate_pcm16(pcm, self._num_channels)
            output = InputAudioRawFrame(
                audio=pcm,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
            )
            output.pts = _av_frame_pts_nanoseconds(resampled)
            output.transport_source = self._source
            converted.append(output)
        return converted


class PCM16PlayoutBuffer:
    """Bounded, interruptible PCM16 FIFO.

    Args:
        sample_rate: Audio sample rate.
        num_channels: Audio channel count.
        frame_duration_ms: Duration of each aiortc output frame.
        max_buffer_ms: Maximum queued audio duration before oldest samples are dropped.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        num_channels: int,
        frame_duration_ms: int = 20,
        max_buffer_ms: int = 2_000,
    ) -> None:
        """Initialize the bounded PCM playout buffer."""
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if frame_duration_ms < 1:
            raise ValueError("frame_duration_ms must be positive")
        if max_buffer_ms < frame_duration_ms:
            raise ValueError("max_buffer_ms must hold at least one output frame")

        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.frame_duration_ms = frame_duration_ms
        self.frame_width = _pcm16_frame_width(num_channels)
        samples = sample_rate * frame_duration_ms
        if samples % 1_000:
            raise ValueError("frame_duration_ms must produce a whole number of samples")
        self.samples_per_frame = samples // 1_000
        self.chunk_bytes = self.samples_per_frame * self.frame_width
        max_bytes = sample_rate * max_buffer_ms * self.frame_width // 1_000
        self.max_buffer_bytes = max(self.chunk_bytes, max_bytes - (max_bytes % self.frame_width))

        self._runs: deque[tuple[bytearray, bool]] = deque()
        self._buffered_bytes = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self.dropped_bytes = 0
        self.interruptions = 0

    @property
    def closed(self) -> bool:
        """Return whether the buffer rejects new audio."""
        return self._closed

    @property
    def buffered_bytes(self) -> int:
        """Return queued audio bytes."""
        return self._buffered_bytes

    async def append(self, audio: bytes, *, interruptible: bool = True) -> bool:
        """Append PCM and discard oldest samples when the latency cap is exceeded."""
        _validate_pcm16(audio, self.num_channels)
        if not audio:
            return not self._closed

        async with self._lock:
            if self._closed:
                return False
            if self._runs and self._runs[-1][1] == interruptible:
                self._runs[-1][0].extend(audio)
            else:
                self._runs.append((bytearray(audio), interruptible))
            self._buffered_bytes += len(audio)
            overflow = self._buffered_bytes - self.max_buffer_bytes
            if overflow > 0:
                aligned = overflow + (-overflow % self.frame_width)
                self._discard_front(aligned)
                self.dropped_bytes += aligned
            return True

    async def pop_chunk(self) -> bytes:
        """Return one fixed-duration chunk, padding underruns with silence."""
        async with self._lock:
            take = min(self.chunk_bytes, self._buffered_bytes)
            chunk = self._take_front(take)
        if take < self.chunk_bytes:
            chunk += bytes(self.chunk_bytes - take)
        return chunk

    async def interrupt(self) -> None:
        """Discard interruptible audio while retaining protected frames."""
        async with self._lock:
            kept: deque[tuple[bytearray, bool]] = deque()
            kept_bytes = 0
            for audio, interruptible in self._runs:
                if not interruptible:
                    kept.append((audio, False))
                    kept_bytes += len(audio)
            self._runs = kept
            self._buffered_bytes = kept_bytes
            self.interruptions += 1

    async def close(self) -> None:
        """Reject future writes and discard queued audio."""
        async with self._lock:
            self._closed = True
            self._runs.clear()
            self._buffered_bytes = 0

    def _discard_front(self, byte_count: int) -> None:
        remaining = min(byte_count, self._buffered_bytes)
        while remaining and self._runs:
            audio, _ = self._runs[0]
            take = min(remaining, len(audio))
            del audio[:take]
            self._buffered_bytes -= take
            remaining -= take
            if not audio:
                self._runs.popleft()

    def _take_front(self, byte_count: int) -> bytes:
        chunks: list[bytes] = []
        remaining = min(byte_count, self._buffered_bytes)
        while remaining and self._runs:
            audio, _ = self._runs[0]
            take = min(remaining, len(audio))
            chunks.append(bytes(audio[:take]))
            del audio[:take]
            self._buffered_bytes -= take
            remaining -= take
            if not audio:
                self._runs.popleft()
        return b"".join(chunks)


class BufferedPCM16AudioTrack(AudioStreamTrack):
    """aiortc track backed by a bounded Pipecat PCM buffer.

    Args:
        sample_rate: Audio sample rate.
        num_channels: Audio channel count.
        frame_duration_ms: Duration of each emitted frame.
        max_buffer_ms: Maximum queued audio duration.
        pace: Whether to pace frames in real time.
    """

    kind = "audio"

    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        num_channels: int = 1,
        frame_duration_ms: int = 20,
        max_buffer_ms: int = 2_000,
        pace: bool = True,
    ) -> None:
        """Initialize the buffered aiortc audio track."""
        super().__init__()
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.buffer = PCM16PlayoutBuffer(
            sample_rate=sample_rate,
            num_channels=num_channels,
            frame_duration_ms=frame_duration_ms,
            max_buffer_ms=max_buffer_ms,
        )
        self._pace = pace
        self._next_deadline: float | None = None
        self._pts_samples = 0

    @property
    def buffered_bytes(self) -> int:
        """Return audio bytes waiting for aiortc."""
        return self.buffer.buffered_bytes

    async def write_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Queue a normalized Pipecat output frame.

        Args:
            frame: Pipecat output audio frame.

        Returns:
            True when the buffer accepted the frame.
        """
        if frame.sample_rate != self.sample_rate:
            raise ValueError(f"expected {self.sample_rate} Hz PCM, received {frame.sample_rate} Hz")
        if frame.num_channels != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} channel(s), received {frame.num_channels}"
            )
        return await self.buffer.append(frame.audio, interruptible=frame.interruptible)

    async def interrupt(self) -> None:
        """Discard audio already queued for WebRTC playout."""
        await self.buffer.interrupt()

    async def close(self) -> None:
        """Stop the track and discard pending audio."""
        await self.buffer.close()
        self.stop()

    async def recv(self) -> AudioFrame:
        """Return one fixed-duration PyAV frame, using silence on underrun."""
        if self.readyState != "live" or self.buffer.closed:
            raise MediaStreamError

        if self._pace:
            now = time.monotonic()
            if self._next_deadline is None:
                self._next_deadline = now
            delay = self._next_deadline - now
            if delay > 0:
                await asyncio.sleep(delay)

        pcm = await self.buffer.pop_chunk()
        frame = _pcm16_bytes_to_av_frame(
            pcm,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            pts_samples=self._pts_samples,
        )
        self._pts_samples += self.buffer.samples_per_frame
        if self._pace:
            assert self._next_deadline is not None
            self._next_deadline += self.buffer.frame_duration_ms / 1_000
        return frame


class BlankVideoTrack(VideoStreamTrack):
    """Blank video track required by Amazon IVS WHIP publishing."""

    kind = "video"

    async def recv(self) -> VideoFrame:
        """Return one blank H.264-compatible video frame."""
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame(320, 180, "yuv420p")
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.pts = pts
        frame.time_base = time_base
        return frame


class AmazonIVSParams(TransportParams):
    """Configuration parameters for Amazon IVS Real-Time audio.

    Parameters:
        audio_in_enabled: Subscribe to participant audio. Defaults to ``True``.
        audio_in_sample_rate: Input PCM sample rate in Hz. Defaults to 16000.
        audio_in_channels: Input PCM channel count. Defaults to mono.
        audio_in_passthrough: Forward input audio downstream. Defaults to ``True``.
        audio_out_enabled: Publish pipeline audio. Defaults to ``True``.
        audio_out_sample_rate: Output PCM sample rate in Hz. Defaults to 24000.
        audio_out_channels: Output PCM channel count. Defaults to mono.
        audio_out_10ms_chunks: Pipecat output chunks per write. Defaults to two.
        audio_out_end_silence_secs: End-of-stream silence in seconds. Defaults to zero.
        audio_out_auto_silence: Let Pipecat generate silence. Defaults to ``False`` because
            the aiortc track pads underruns.
        input_recv_timeout_secs: Seconds between checks while waiting for input. Must be
            positive and defaults to 5.
        input_source: Transport source identifier on input frames. Defaults to
            ``"amazon-ivs"``.
        output_frame_duration_ms: WebRTC publication frame duration in milliseconds.
            Must be positive and defaults to 20.
        output_max_buffer_ms: Maximum queued publication audio in milliseconds. Oldest
            sample-aligned audio is dropped above the limit. Defaults to 2000.
        connection_timeout_secs: Seconds allowed for WebRTC connection establishment.
            Must be positive and defaults to 15.
        sdp_redirect_limit: Maximum HTTP 307 redirects for WHIP or WHEP negotiation.
            Accepts 0 through 10 and defaults to 5.
    """

    audio_in_enabled: bool = True
    audio_in_sample_rate: int | None = 16_000
    audio_in_channels: int = Field(default=1, ge=1, le=2)
    audio_in_passthrough: bool = True
    audio_out_enabled: bool = True
    audio_out_sample_rate: int | None = 24_000
    audio_out_channels: int = Field(default=1, ge=1, le=2)
    audio_out_10ms_chunks: int = Field(default=2, gt=0)
    audio_out_end_silence_secs: int = 0
    audio_out_auto_silence: bool = False
    input_recv_timeout_secs: float = Field(default=5.0, gt=0)
    input_source: str | None = "amazon-ivs"
    output_frame_duration_ms: int = Field(default=20, gt=0)
    output_max_buffer_ms: int = Field(default=2_000, gt=0)
    connection_timeout_secs: float = Field(default=15.0, gt=0)
    sdp_redirect_limit: int = Field(default=5, ge=0, le=10)


class AmazonIVSCallbacks(BaseModel):
    """Callback handlers for Amazon IVS events.

    Parameters:
        on_connected: Called after publisher and subscriber connections are ready.
        on_disconnected: Called after the participant disconnects.
        on_client_connected: Called when the target audio track is available.
        on_client_disconnected: Called when the target audio track disconnects.
        on_error: Called when connection setup or a peer connection fails.
    """

    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_client_connected: Callable[[str], Awaitable[None]]
    on_client_disconnected: Callable[[str], Awaitable[None]]
    on_error: Callable[[str], Awaitable[None]]


class AmazonIVSTransportClient:
    """Manage Amazon IVS WHIP, WHEP, and aiortc lifecycle.

    Args:
        participant_token: Short-lived IVS participant token with required capabilities.
        subscribe_participant_id: Participant whose audio should be subscribed.
        params: Amazon IVS transport configuration.
        callbacks: Transport event callbacks.
        transport_name: Name used in diagnostics.
        subscription_url: Full WHEP endpoint supplied by a trusted backend.
    """

    def __init__(
        self,
        participant_token: str,
        subscribe_participant_id: str | None,
        params: AmazonIVSParams,
        callbacks: AmazonIVSCallbacks,
        transport_name: str,
        *,
        subscription_url: str | None = None,
    ) -> None:
        """Initialize the shared Amazon IVS signalling and media client."""
        self._participant_token = participant_token
        self._subscribe_participant_id = subscribe_participant_id
        self._params = params
        self._callbacks = callbacks
        self._transport_name = transport_name
        self._subscription_url = (
            require_ivs_subscription_url(subscription_url, subscribe_participant_id)
            if subscription_url and subscribe_participant_id
            else None
        )
        if subscription_url and not subscribe_participant_id:
            raise AmazonIVSError(
                "subscribe_participant_id is required when subscription_url is provided"
            )
        self._publisher_pc: RTCPeerConnection | None = None
        self._subscriber_pc: RTCPeerConnection | None = None
        self._input_track: AudioStreamTrack | None = None
        self._output_track: BufferedPCM16AudioTrack | None = None
        self._output_sample_rate: int | None = None
        self._connected = False
        self._client_connected = False
        self._failure_lock = asyncio.Lock()
        self._setup_failure_event = asyncio.Event()
        self._setup_error: AmazonIVSError | None = None

    @property
    def input_track(self) -> AudioStreamTrack:
        """Return the subscribed audio track.

        Raises:
            AmazonIVSError: If the subscriber connection is not ready.
        """
        if self._input_track is None:
            raise AmazonIVSError("Amazon IVS input track is not connected")
        return self._input_track

    @property
    def output_track(self) -> BufferedPCM16AudioTrack:
        """Return the publication audio track.

        Raises:
            AmazonIVSError: If the client has not been set up.
        """
        if self._output_track is None:
            raise AmazonIVSError("Amazon IVS output track is not configured")
        return self._output_track

    @property
    def connected(self) -> bool:
        """Return whether the Amazon IVS participant is connected."""
        return self._connected

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Configure output audio from pipeline settings.

        Args:
            setup: Frame processor setup.
        """
        if not self._params.audio_out_enabled:
            return

        self._output_sample_rate = self._params.audio_out_sample_rate or setup.audio_out_sample_rate
        self._ensure_output_track()

    def _ensure_output_track(self) -> None:
        """Create the publisher audio track when output is enabled."""
        if self._output_track is not None or not self._params.audio_out_enabled:
            return
        if self._output_sample_rate is None:
            raise AmazonIVSError("Amazon IVS output is not configured; call setup() first")
        self._output_track = BufferedPCM16AudioTrack(
            sample_rate=self._output_sample_rate,
            num_channels=self._params.audio_out_channels,
            frame_duration_ms=self._params.output_frame_duration_ms,
            max_buffer_ms=self._params.output_max_buffer_ms,
        )

    @acquires("connection")
    async def connect(self) -> None:
        """Connect the configured Amazon IVS publisher and subscriber."""
        self._setup_error = None
        self._setup_failure_event.clear()
        try:
            participant_id = self._subscribe_participant_id
            subscription_url = self._subscription_url
            if self._params.audio_in_enabled and (not participant_id or not subscription_url):
                raise AmazonIVSError(
                    "subscribe_participant_id and subscription_url are required "
                    "when audio input is enabled"
                )

            timeout = aiohttp.ClientTimeout(total=self._params.connection_timeout_secs)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if self._params.audio_out_enabled:
                    self._ensure_output_track()
                    await self._run_setup_step(self._connect_publisher(session))
                if self._params.audio_in_enabled:
                    await self._run_setup_step(
                        self._connect_subscriber(
                            session,
                            subscription_url,
                            participant_id,
                        )
                    )
            self._connected = True
        except BaseException as error:
            await self._close_connections()
            await self._callbacks.on_error(str(error) or type(error).__name__)
            raise

        logger.info(f"{self._transport_name} connected to Amazon IVS")
        await self._callbacks.on_connected()

    @releases("connection")
    async def disconnect(self) -> None:
        """Release one transport reference and close the participant when unused."""
        was_connected = self._connected
        self._connected = False
        await self._close_connections()

        if was_connected:
            logger.info(f"{self._transport_name} disconnected from Amazon IVS")
            await self._callbacks.on_disconnected()

    async def cleanup(self) -> None:
        """Close all peer connections regardless of reference count."""
        was_connected = self._connected
        self._connected = False
        await self._close_connections()
        if was_connected:
            await self._callbacks.on_disconnected()

    async def interrupt_output(self) -> None:
        """Discard audio already queued for WebRTC publication."""
        if self._output_track is not None:
            await self._output_track.interrupt()

    async def input_ended(self) -> None:
        """Tear down the shared session when the subscribed audio track ends."""
        await self._handle_peer_failure("subscriber", "ended")

    async def _run_setup_step(self, awaitable: Awaitable[None]) -> None:
        """Run one signalling step while monitoring previously connected peers."""
        step_task = asyncio.create_task(awaitable)
        failure_task = asyncio.create_task(self._setup_failure_event.wait())
        try:
            done, _ = await asyncio.wait(
                {step_task, failure_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure_task in done and self._setup_failure_event.is_set():
                raise self._setup_error or AmazonIVSError(
                    "Amazon IVS peer connection failed during setup"
                )

            await step_task
            if self._setup_failure_event.is_set():
                raise self._setup_error or AmazonIVSError(
                    "Amazon IVS peer connection failed during setup"
                )
        finally:
            for task in (step_task, failure_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(step_task, failure_task, return_exceptions=True)

    async def _connect_publisher(
        self,
        session: aiohttp.ClientSession,
    ) -> None:
        pc = self._peer_connection("publisher")
        self._publisher_pc = pc
        pc.addTransceiver(self.output_track, direction="sendrecv")
        video_transceiver = pc.addTransceiver(BlankVideoTrack(), direction="sendrecv")
        h264_codecs = [
            codec
            for codec in RTCRtpSender.getCapabilities("video").codecs
            if codec.mimeType.lower() == "video/h264"
        ]
        if not h264_codecs:
            raise AmazonIVSError("aiortc has no H.264 video codec capability")
        video_transceiver.setCodecPreferences(h264_codecs)
        await pc.setLocalDescription(await pc.createOffer())
        answer = await self._post_sdp(
            session,
            IVS_WHIP_URL,
            pc.localDescription.sdp,
        )
        await pc.setRemoteDescription(RTCSessionDescription(fix_ivs_answer_sdp(answer), "answer"))
        await self._wait_connected(pc)

    async def _connect_subscriber(
        self,
        session: aiohttp.ClientSession,
        subscription_url: str,
        participant_id: str,
    ) -> None:
        pc = self._peer_connection("subscriber")
        self._subscriber_pc = pc
        pc.addTransceiver("audio", direction="recvonly")
        loop = asyncio.get_running_loop()
        track_future: asyncio.Future[AudioStreamTrack] = loop.create_future()

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio" and not track_future.done():
                track_future.set_result(track)

        await pc.setLocalDescription(await pc.createOffer())
        answer = await self._post_sdp(
            session,
            subscription_url,
            pc.localDescription.sdp,
        )
        await pc.setRemoteDescription(RTCSessionDescription(fix_ivs_answer_sdp(answer), "answer"))
        await self._wait_connected(pc)
        self._input_track = await asyncio.wait_for(
            track_future,
            self._params.connection_timeout_secs,
        )
        self._client_connected = True
        await self._callbacks.on_client_connected(participant_id)

    def _peer_connection(self, label: str) -> RTCPeerConnection:
        config = RTCConfiguration()
        config.bundlePolicy = RTCBundlePolicy.MAX_BUNDLE
        pc = RTCPeerConnection(config)

        @pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            if pc.connectionState == "failed":
                error = AmazonIVSError(f"Amazon IVS {label} peer connection failed")
                if not self._connected:
                    self._setup_error = error
                    self._setup_failure_event.set()
                else:
                    await self._handle_peer_failure(label, pc.connectionState)

        return pc

    async def _handle_peer_failure(self, label: str, state: str) -> None:
        """Tear down both peer connections after a runtime media failure."""
        async with self._failure_lock:
            if not self._connected:
                return
            self._connected = False
            await self._callbacks.on_error(f"Amazon IVS {label} peer connection entered {state}")
            await self._close_connections()
            await self._callbacks.on_disconnected()

    async def _post_sdp(
        self,
        session: aiohttp.ClientSession,
        url: str,
        offer: str,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self._participant_token}",
            "Content-Type": "application/sdp",
        }
        current_url = require_ivs_url(url)
        for _ in range(self._params.sdp_redirect_limit + 1):
            async with session.post(
                current_url,
                data=offer,
                headers=headers,
                allow_redirects=False,
            ) as response:
                if response.status == 201:
                    body = await response.content.read(MAX_SDP_BYTES + 1)
                    if len(body) > MAX_SDP_BYTES:
                        raise AmazonIVSError("Amazon IVS SDP answer exceeded the size limit")
                    try:
                        return body.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise AmazonIVSError("Amazon IVS SDP answer was not UTF-8") from error
                if response.status == 307:
                    location = response.headers.get("Location")
                    if not location:
                        raise AmazonIVSError("Amazon IVS SDP redirect omitted Location")
                    current_url = safe_ivs_redirect(current_url, location)
                    continue
                raise AmazonIVSError(f"Amazon IVS SDP exchange failed: HTTP {response.status}")
        raise AmazonIVSError("Amazon IVS SDP exchange exceeded the redirect limit")

    async def _wait_connected(self, pc: RTCPeerConnection) -> None:
        async def poll() -> None:
            while pc.connectionState not in {"connected", "completed"}:
                if pc.connectionState in {"closed", "failed"}:
                    raise AmazonIVSError(f"Amazon IVS peer connection entered {pc.connectionState}")
                await asyncio.sleep(0.05)

        await asyncio.wait_for(poll(), self._params.connection_timeout_secs)

    async def _close_connections(self) -> None:
        participant_id = self._subscribe_participant_id if self._client_connected else None
        self._client_connected = False
        connections = [self._subscriber_pc, self._publisher_pc]
        output_track = self._output_track
        self._subscriber_pc = None
        self._publisher_pc = None
        self._input_track = None
        self._output_track = None

        closers = [connection.close() for connection in connections if connection is not None]
        if output_track is not None:
            closers.append(output_track.close())
        results = await asyncio.gather(*closers, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]

        if participant_id:
            try:
                await self._callbacks.on_client_disconnected(participant_id)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise AmazonIVSError("Amazon IVS resource cleanup failed") from errors[0]


class AmazonIVSInputTransport(BaseInputTransport):
    """Receive Amazon IVS audio and emit Pipecat input frames."""

    def __init__(
        self,
        transport: BaseTransport,
        client: AmazonIVSTransportClient,
        params: AmazonIVSParams,
        **kwargs,
    ) -> None:
        """Initialize the Amazon IVS input transport.

        Args:
            transport: Parent transport.
            client: Shared Amazon IVS client.
            params: Transport configuration.
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._converter: AmazonIVSIngressConverter | None = None
        self._receive_task: asyncio.Task | None = None

    @property
    def receive_task_active(self) -> bool:
        """Return whether the aiortc receive loop is running."""
        return self._receive_task is not None and not self._receive_task.done()

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up audio conversion and connect to Amazon IVS.

        Args:
            setup: Frame processor setup.
        """
        await super().setup(setup)
        self._converter = AmazonIVSIngressConverter(
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_in_channels,
            source=self._params.input_source,
        )
        await self._client.setup(setup)
        await self._client.connect()

    async def start(self, frame: StartFrame) -> None:
        """Start receiving Amazon IVS audio.

        Args:
            frame: Pipeline start frame.
        """
        await super().start(frame)
        if self._params.audio_in_enabled and self._receive_task is None:
            if not self._client.connected:
                raise AmazonIVSError(
                    "Amazon IVS input is disconnected; create a new transport to reconnect"
                )
            self._receive_task = self.create_task(
                self._receive_audio(),
                name="amazon_ivs_audio_receive",
            )
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame) -> None:
        """Stop receiving Amazon IVS audio.

        Args:
            frame: Pipeline end frame.
        """
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel Amazon IVS input immediately.

        Args:
            frame: Pipeline cancel frame.
        """
        await super().cancel(frame)
        await self._teardown()

    async def cleanup(self) -> None:
        """Release Amazon IVS input resources."""
        await super().cleanup()
        await self._teardown()
        await self._transport.cleanup()

    async def _teardown(self) -> None:
        if self._receive_task is not None:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._client.disconnect()

    async def _receive_audio(self) -> None:
        current_task = asyncio.current_task()
        try:
            if self._converter is None:
                return
            track = self._client.input_track
            while True:
                try:
                    media_frame = await asyncio.wait_for(
                        track.recv(),
                        timeout=self._params.input_recv_timeout_secs,
                    )
                except TimeoutError:
                    continue
                except MediaStreamError:
                    await self._client.input_ended()
                    return
                if not isinstance(media_frame, AudioFrame):
                    continue
                for pipecat_frame in self._converter.convert(media_frame):
                    await self.push_audio_frame(pipecat_frame)
        finally:
            if self._receive_task is current_task:
                self._receive_task = None


class AmazonIVSOutputTransport(BaseOutputTransport):
    """Publish Pipecat audio through Amazon IVS."""

    def __init__(
        self,
        transport: BaseTransport,
        client: AmazonIVSTransportClient,
        params: AmazonIVSParams,
        **kwargs,
    ) -> None:
        """Initialize the Amazon IVS output transport.

        Args:
            transport: Parent transport.
            client: Shared Amazon IVS client.
            params: Transport configuration.
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Set up output audio and connect to Amazon IVS.

        Args:
            setup: Frame processor setup.
        """
        await super().setup(setup)
        await self._client.setup(setup)
        await self._client.connect()

    async def start(self, frame: StartFrame) -> None:
        """Start publishing Pipecat audio.

        Args:
            frame: Pipeline start frame.
        """
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame) -> None:
        """Stop publishing Pipecat audio.

        Args:
            frame: Pipeline end frame.
        """
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel Amazon IVS output immediately.

        Args:
            frame: Pipeline cancel frame.
        """
        await super().cancel(frame)
        await self._client.disconnect()

    async def cleanup(self) -> None:
        """Release Amazon IVS output resources."""
        await super().cleanup()
        await self._client.disconnect()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process output frames and clear queued audio on interruption.

        Args:
            frame: Frame to process.
            direction: Frame direction.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            await self._client.interrupt_output()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Queue one Pipecat audio frame for Amazon IVS publication.

        Args:
            frame: Output audio frame.

        Returns:
            True when the publication track accepted the frame.
        """
        return await self._client.output_track.write_frame(frame)


class AmazonIVSTransport(BaseTransport):
    """Amazon IVS Real-Time transport for one subscribed participant.

    A trusted backend creates the IVS stage and returns an opaque, short-lived
    participant token plus the full WHEP subscription URL. Do not parse, log, or
    expose either value. The subscription URL must target
    ``subscribe_participant_id``. This transport uses WHEP to subscribe to that
    participant and the global IVS WHIP endpoint to publish the agent response.

    Set ``audio_in_enabled=False`` to create an output-only transport without
    ``subscribe_participant_id`` or ``subscription_url``. The participant token
    is always required.

    Event handlers available:

    - on_connected(transport): Connected to Amazon IVS.
    - on_disconnected(transport): Disconnected from Amazon IVS.
    - on_client_connected(transport, client): Target participant audio is ready;
      ``client`` is ``{"id": participant_id}``.
    - on_client_disconnected(transport, client): Target participant audio disconnected;
      ``client`` is ``{"id": participant_id}``.
    - on_error(transport, message): Signalling or peer connection error as a string.

    Example::

        import os

        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.transports.amazon_ivs import AmazonIVSParams, AmazonIVSTransport

        transport = AmazonIVSTransport(
            participant_token=os.environ["IVS_PARTICIPANT_TOKEN"],
            subscribe_participant_id=os.environ["IVS_SOURCE_PARTICIPANT_ID"],
            subscription_url=os.environ["IVS_SUBSCRIPTION_URL"],
            params=AmazonIVSParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )
        pipeline = Pipeline([transport.input(), transport.output()])

    Args:
        participant_token: Short-lived IVS participant token with publish and subscribe access.
        subscribe_participant_id: Participant whose audio should enter the pipeline. May
            be ``None`` only when audio input is disabled.
        params: Amazon IVS transport configuration.
        subscription_url: Opaque full WHEP endpoint supplied by a trusted backend. May
            be ``None`` only when audio input is disabled.
        input_name: Optional name for the input processor.
        output_name: Optional name for the output processor.

    Raises:
        AmazonIVSError: If the subscription URL is untrusted or does not target the
            selected participant.
    """

    def __init__(
        self,
        participant_token: str,
        subscribe_participant_id: str | None = None,
        params: AmazonIVSParams | None = None,
        *,
        subscription_url: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> None:
        """Initialize the Amazon IVS transport."""
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params or AmazonIVSParams()
        callbacks = AmazonIVSCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_error=self._on_error,
        )
        self._client = AmazonIVSTransportClient(
            participant_token,
            subscribe_participant_id,
            self._params,
            callbacks,
            self.name,
            subscription_url=subscription_url,
        )
        self._input: AmazonIVSInputTransport | None = None
        self._output: AmazonIVSOutputTransport | None = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_error")

    def get_client_id(self, client: Any) -> str:
        """Return the IVS participant ID from a client event payload."""
        if isinstance(client, dict):
            return str(client.get("id", ""))
        return ""

    def input(self) -> AmazonIVSInputTransport:
        """Return the Amazon IVS input processor."""
        if self._input is None:
            self._input = AmazonIVSInputTransport(
                self,
                self._client,
                self._params,
                name=self._input_name,
            )
        return self._input

    def output(self) -> AmazonIVSOutputTransport:
        """Return the Amazon IVS output processor."""
        if self._output is None:
            self._output = AmazonIVSOutputTransport(
                self,
                self._client,
                self._params,
                name=self._output_name,
            )
        return self._output

    async def _on_connected(self) -> None:
        await self._call_event_handler("on_connected")
        if self._input:
            await self._input.push_frame(BotConnectedFrame())

    async def _on_disconnected(self) -> None:
        await self._call_event_handler("on_disconnected")

    async def _on_client_connected(self, participant_id: str) -> None:
        client = {"id": participant_id}
        await self._call_event_handler("on_client_connected", client)
        if self._input:
            await self._input.push_frame(ClientConnectedFrame())

    async def _on_client_disconnected(self, participant_id: str) -> None:
        await self._call_event_handler("on_client_disconnected", {"id": participant_id})

    async def _on_error(self, message: str) -> None:
        await self._call_event_handler("on_error", message)
