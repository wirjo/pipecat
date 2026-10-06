#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Amazon IVS Real-Time audio transport."""

import asyncio
from fractions import Fraction
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("aiortc")
pytest.importorskip("av")

from aiortc import AudioStreamTrack  # noqa: E402
from aiortc.mediastreams import MediaStreamError  # noqa: E402
from av import AudioFrame  # noqa: E402

from pipecat.frames.frames import (  # noqa: E402
    CancelFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402
from pipecat.transports.amazon_ivs.transport import (  # noqa: E402
    AmazonIVSCallbacks,
    AmazonIVSError,
    AmazonIVSInputTransport,
    AmazonIVSOutputTransport,
    AmazonIVSParams,
    AmazonIVSTransport,
    AmazonIVSTransportClient,
    BufferedPCM16AudioTrack,
    fix_ivs_answer_sdp,
    require_ivs_subscription_url,
    require_ivs_url,
    safe_ivs_redirect,
)
from pipecat.utils.asyncio.task_manager import TaskManager  # noqa: E402
from tests.frame_processor_helpers import frame_processor_setup  # noqa: E402


class FakeParentTransport:
    """Parent transport double used to verify processor cleanup."""

    def __init__(self):
        self.cleanup_calls = 0

    async def cleanup(self):
        self.cleanup_calls += 1


class FakeInputTrack(AudioStreamTrack):
    """Provider-free aiortc track returning predetermined audio frames."""

    kind = "audio"

    def __init__(self, frames=None):
        super().__init__()
        self._frames = iter(frames or [])

    async def recv(self):
        try:
            return next(self._frames)
        except StopIteration as error:
            raise MediaStreamError from error


class BlockingInputTrack(AudioStreamTrack):
    """aiortc track that records cancellation while blocked in ``recv``."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self.read_started = asyncio.Event()
        self.read_cancelled = asyncio.Event()

    async def recv(self):
        self.read_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.read_cancelled.set()
            raise


class FakeAmazonIVSClient:
    """Provider-free client double for input and output processor tests."""

    def __init__(self, input_track=None, output_track=None):
        self._input_track = input_track
        self._output_track = output_track
        self.setup_calls = 0
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.input_ended_calls = 0
        self.connected = False

    @property
    def input_track(self):
        if self._input_track is None:
            raise RuntimeError("input track is not configured")
        return self._input_track

    @property
    def output_track(self):
        if self._output_track is None:
            raise RuntimeError("output track is not configured")
        return self._output_track

    async def setup(self, _setup):
        self.setup_calls += 1

    async def connect(self):
        self.connect_calls += 1
        self.connected = True

    async def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    async def cleanup(self):
        await self.disconnect()

    async def interrupt_output(self):
        await self.output_track.interrupt()

    async def input_ended(self):
        self.input_ended_calls += 1
        self.connected = False


def make_av_audio_frame(
    samples: np.ndarray,
    *,
    sample_rate: int = 16_000,
    pts: int | None = 0,
) -> AudioFrame:
    """Build a packed mono PyAV audio frame."""
    frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = sample_rate
    frame.pts = pts
    frame.time_base = Fraction(1, sample_rate)
    return frame


def media_sections(sdp: str) -> dict[str, list[str]]:
    """Split SDP into media sections keyed by media type."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in sdp.splitlines():
        if line.startswith("m="):
            media_type = line.removeprefix("m=").split()[0]
            current = [line]
            sections[media_type] = current
        elif current is not None:
            current.append(line)
    return sections


async def wait_until(predicate, timeout: float = 1.0) -> None:
    """Wait for an asynchronous transport side effect."""

    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=timeout)


def test_amazon_ivs_params_default_to_audio_only_realtime_settings() -> None:
    params = AmazonIVSParams()

    assert params.audio_in_enabled is True
    assert params.audio_in_sample_rate == 16_000
    assert params.audio_in_channels == 1
    assert params.audio_out_enabled is True
    assert params.audio_out_sample_rate == 24_000
    assert params.audio_out_channels == 1
    assert params.audio_out_end_silence_secs == 0
    assert params.audio_out_auto_silence is False
    assert params.input_recv_timeout_secs == 5.0
    assert params.output_frame_duration_ms == 20
    assert params.output_max_buffer_ms == 2_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"audio_in_channels": 0},
        {"audio_in_channels": 3},
        {"audio_out_channels": 0},
        {"audio_out_channels": 3},
        {"audio_out_10ms_chunks": 0},
    ],
)
def test_amazon_ivs_params_reject_invalid_audio_geometry(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AmazonIVSParams(**kwargs)


@pytest.mark.parametrize(
    "url",
    [
        "https://global.whip.live-video.net",
        "https://iad.whip.live-video.net/session",
        "https://iad.whip.live-video.net:443/session?stage=test",
    ],
)
def test_require_ivs_url_accepts_only_https_service_subdomains(url: str) -> None:
    assert require_ivs_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://abc.live-video.net/session",
        "https://live-video.net/session",
        "https://abc.live-video.net/session",
        "https://example.com/session",
        "https://abc.live-video.net.example.com/session",
        "https://live-video.net@example.com/session",
        "https://user@example.live-video.net/session",
        "https://abc.live-video.net:8443/session",
        "https://abc.live-video.net:not-a-port/session",
        "https://abc.live-video.net/session#fragment",
        "https://127.0.0.1/session",
    ],
)
def test_require_ivs_url_rejects_untrusted_endpoints(url: str) -> None:
    with pytest.raises((RuntimeError, ValueError), match="untrusted"):
        require_ivs_url(url)


@pytest.mark.parametrize(
    ("current_url", "location", "expected"),
    [
        (
            "https://global.whip.live-video.net/session/start",
            "../regional/session",
            "https://global.whip.live-video.net/regional/session",
        ),
        (
            "https://global.whip.live-video.net",
            "https://iad.whip.live-video.net/session",
            "https://iad.whip.live-video.net/session",
        ),
    ],
)
def test_safe_ivs_redirect_resolves_trusted_absolute_and_relative_locations(
    current_url: str, location: str, expected: str
) -> None:
    assert safe_ivs_redirect(current_url, location) == expected


@pytest.mark.parametrize(
    "location",
    [
        "http://abc.live-video.net/session",
        "https://example.com/session",
        "//example.com/session",
        "https://user@abc.live-video.net/session",
        "https://abc.live-video.net:8443/session",
    ],
)
def test_safe_ivs_redirect_rejects_token_exfiltration_destinations(location: str) -> None:
    with pytest.raises((RuntimeError, ValueError), match="untrusted"):
        safe_ivs_redirect("https://global.whip.live-video.net/session", location)


def test_subscription_url_must_target_selected_participant() -> None:
    url = "https://iad.whip.live-video.net/session/subscribe/user-123"

    assert require_ivs_subscription_url(url, "user-123") == url

    with pytest.raises(RuntimeError, match="another participant"):
        require_ivs_subscription_url(url, "user-456")


def test_fix_ivs_answer_sdp_copies_candidates_into_each_missing_media_section() -> None:
    candidate = "a=candidate:1 1 UDP 1 192.0.2.1 5000 typ host"
    answer = (
        "v=0\r\n"
        "a=group:BUNDLE 0 1 2\r\n"
        "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        "a=mid:0\r\n"
        f"{candidate}\r\n"
        "a=end-of-candidates\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 102\r\n"
        "a=mid:1\r\n"
        "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n"
        "a=mid:2\r\n"
    )

    fixed = fix_ivs_answer_sdp(answer)
    sections = media_sections(fixed)

    assert fixed.count(candidate) == 3
    assert fixed.count("a=end-of-candidates") == 3
    assert candidate in sections["video"]
    assert candidate in sections["application"]
    assert fix_ivs_answer_sdp(fixed) == fixed


def test_fix_ivs_answer_sdp_preserves_answers_without_candidates() -> None:
    answer = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:0\r\n"

    assert fix_ivs_answer_sdp(answer) == answer


def test_fix_ivs_answer_sdp_rejects_oversized_or_pathological_answers() -> None:
    with pytest.raises(RuntimeError, match="size limit"):
        fix_ivs_answer_sdp("v=0\r\n" + ("a=x\r\n" * 60_000))

    answer = "v=0\r\n" + "".join(
        f"a=candidate:{index} 1 UDP 1 192.0.2.1 5000 typ host\r\n" for index in range(257)
    )
    with pytest.raises(RuntimeError, match="too many ICE candidates"):
        fix_ivs_answer_sdp(answer)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"num_channels": 0},
        {"frame_duration_ms": 0},
        {"frame_duration_ms": 20, "max_buffer_ms": 10},
        {"sample_rate": 44_100, "frame_duration_ms": 7},
    ],
)
def test_buffered_track_rejects_invalid_pcm_geometry(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BufferedPCM16AudioTrack(pace=False, **kwargs)


@pytest.mark.asyncio
async def test_buffered_track_drops_oldest_audio_when_latency_cap_is_exceeded() -> None:
    track = BufferedPCM16AudioTrack(
        sample_rate=1_000,
        num_channels=1,
        frame_duration_ms=20,
        max_buffer_ms=40,
        pace=False,
    )
    samples = np.arange(60, dtype="<i2")

    try:
        accepted = await track.write_frame(
            OutputAudioRawFrame(
                audio=samples.tobytes(),
                sample_rate=1_000,
                num_channels=1,
            )
        )
        first = await track.recv()
        second = await track.recv()
    finally:
        await track.close()

    assert accepted is True
    assert first.to_ndarray().tobytes() == samples[20:40].tobytes()
    assert second.to_ndarray().tobytes() == samples[40:60].tobytes()


@pytest.mark.asyncio
async def test_buffered_track_converts_pipecat_pcm_to_fixed_duration_pyav_frames() -> None:
    track = BufferedPCM16AudioTrack(
        sample_rate=24_000,
        num_channels=1,
        frame_duration_ms=20,
        max_buffer_ms=100,
        pace=False,
    )
    samples = np.arange(480, dtype="<i2")

    try:
        await track.write_frame(
            OutputAudioRawFrame(
                audio=samples.tobytes(),
                sample_rate=24_000,
                num_channels=1,
            )
        )
        first = await track.recv()
        second = await track.recv()
    finally:
        await track.close()

    assert first.sample_rate == 24_000
    assert first.samples == 480
    assert first.pts == 0
    assert first.time_base == Fraction(1, 24_000)
    assert first.to_ndarray().tobytes() == samples.tobytes()
    assert second.pts == 480
    assert second.samples == 480
    assert second.to_ndarray().tobytes() == bytes(480 * 2)


@pytest.mark.asyncio
async def test_buffered_track_rejects_mismatched_or_partial_pipecat_pcm() -> None:
    track = BufferedPCM16AudioTrack(sample_rate=24_000, num_channels=1, pace=False)

    try:
        with pytest.raises(ValueError, match="24"):
            await track.write_frame(
                OutputAudioRawFrame(audio=b"\0\0", sample_rate=16_000, num_channels=1)
            )
        with pytest.raises(ValueError, match="channel"):
            await track.write_frame(
                OutputAudioRawFrame(audio=b"\0\0\0\0", sample_rate=24_000, num_channels=2)
            )
        with pytest.raises(ValueError, match="align"):
            await track.write_frame(
                OutputAudioRawFrame(audio=b"\0", sample_rate=24_000, num_channels=1)
            )
    finally:
        await track.close()


@pytest.mark.asyncio
async def test_buffered_track_close_discards_audio_and_rejects_future_io() -> None:
    track = BufferedPCM16AudioTrack(sample_rate=24_000, pace=False)
    frame = OutputAudioRawFrame(audio=b"\1\0" * 480, sample_rate=24_000, num_channels=1)
    await track.write_frame(frame)

    await track.close()

    assert track.buffered_bytes == 0
    assert await track.write_frame(frame) is False
    with pytest.raises(MediaStreamError):
        await track.recv()


@pytest.mark.asyncio
async def test_buffered_track_interruption_retains_protected_audio() -> None:
    track = BufferedPCM16AudioTrack(
        sample_rate=1_000,
        num_channels=1,
        frame_duration_ms=20,
        max_buffer_ms=100,
        pace=False,
    )
    protected = OutputAudioRawFrame(
        audio=b"\1\0" * 20,
        sample_rate=1_000,
        num_channels=1,
    )
    protected.interruptible = False
    interruptible = OutputAudioRawFrame(
        audio=b"\2\0" * 20,
        sample_rate=1_000,
        num_channels=1,
    )

    try:
        await track.write_frame(protected)
        await track.write_frame(interruptible)
        await track.interrupt()
        frame = await track.recv()
    finally:
        await track.close()

    assert frame.to_ndarray().tobytes() == protected.audio


@pytest.mark.asyncio
async def test_input_transport_converts_pyav_audio_to_pipecat_frames() -> None:
    samples = np.arange(320, dtype="<i2")
    input_track = FakeInputTrack([make_av_audio_frame(samples, sample_rate=16_000, pts=160)])
    client = FakeAmazonIVSClient(input_track=input_track)
    parent = FakeParentTransport()
    transport = AmazonIVSInputTransport(
        transport=parent,
        client=client,
        params=AmazonIVSParams(),
    )
    transport.push_audio_frame = AsyncMock()
    await transport.setup(
        frame_processor_setup(
            TaskManager(),
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=24_000,
        )
    )

    try:
        await transport.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await wait_until(lambda: transport.push_audio_frame.await_count == 1)
        frame = transport.push_audio_frame.await_args.args[0]
    finally:
        await transport.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
        await transport.cleanup()

    assert isinstance(frame, InputAudioRawFrame)
    assert frame.audio == samples.tobytes()
    assert frame.sample_rate == 16_000
    assert frame.num_channels == 1
    assert frame.num_frames == 320
    assert frame.pts == 10_000_000
    assert frame.transport_source == "amazon-ivs"
    assert client.setup_calls == 1
    assert client.connect_calls == 1
    assert parent.cleanup_calls == 1


@pytest.mark.asyncio
async def test_input_track_end_clears_receive_task_and_notifies_client() -> None:
    input_track = FakeInputTrack([])
    client = FakeAmazonIVSClient(input_track=input_track)
    parent = FakeParentTransport()
    transport = AmazonIVSInputTransport(
        transport=parent,
        client=client,
        params=AmazonIVSParams(),
    )
    await transport.setup(frame_processor_setup(TaskManager()))

    try:
        await transport.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await wait_until(lambda: client.input_ended_calls == 1)

        assert transport._receive_task is None
        with pytest.raises(RuntimeError, match="disconnected"):
            await transport.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    finally:
        await transport.cleanup()


@pytest.mark.asyncio
async def test_setup_failure_from_an_already_connected_peer_aborts_remaining_setup(
    monkeypatch,
) -> None:
    callbacks = AmazonIVSCallbacks(
        on_connected=AsyncMock(),
        on_disconnected=AsyncMock(),
        on_client_connected=AsyncMock(),
        on_client_disconnected=AsyncMock(),
        on_error=AsyncMock(),
    )
    client = AmazonIVSTransportClient(
        participant_token="participant-token",
        subscribe_participant_id="source-participant",
        subscription_url=("https://iad.whip.live-video.net/session/subscribe/source-participant"),
        params=AmazonIVSParams(),
        callbacks=callbacks,
        transport_name="test-transport",
    )
    await client.setup(frame_processor_setup(TaskManager()))

    async def connect_publisher(_session):
        return None

    async def connect_subscriber(_session, _subscription_url, _participant_id):
        client._setup_error = AmazonIVSError("publisher failed during subscriber setup")
        client._setup_failure_event.set()
        await asyncio.Future()

    monkeypatch.setattr(client, "_connect_publisher", connect_publisher)
    monkeypatch.setattr(client, "_connect_subscriber", connect_subscriber)

    with pytest.raises(AmazonIVSError, match="publisher failed"):
        await client.connect()

    assert client.connected is False
    callbacks.on_connected.assert_not_awaited()
    callbacks.on_error.assert_awaited_once_with("publisher failed during subscriber setup")


@pytest.mark.asyncio
async def test_setup_step_cancellation_cancels_both_child_tasks() -> None:
    callbacks = AmazonIVSCallbacks(
        on_connected=AsyncMock(),
        on_disconnected=AsyncMock(),
        on_client_connected=AsyncMock(),
        on_client_disconnected=AsyncMock(),
        on_error=AsyncMock(),
    )
    client = AmazonIVSTransportClient(
        participant_token="participant-token",
        subscribe_participant_id=None,
        params=AmazonIVSParams(audio_in_enabled=False),
        callbacks=callbacks,
        transport_name="test-transport",
    )
    step_started = asyncio.Event()
    step_cancelled = asyncio.Event()
    failure_wait_started = asyncio.Event()
    failure_wait_cancelled = asyncio.Event()

    class TrackingFailureEvent:
        async def wait(self):
            failure_wait_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                failure_wait_cancelled.set()
                raise

        def is_set(self):
            return False

    client._setup_failure_event = TrackingFailureEvent()

    async def slow_step():
        step_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            step_cancelled.set()
            raise

    setup_task = asyncio.create_task(client._run_setup_step(slow_step()))
    await asyncio.gather(step_started.wait(), failure_wait_started.wait())
    setup_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await setup_task

    assert step_cancelled.is_set()
    assert failure_wait_cancelled.is_set()


@pytest.mark.asyncio
async def test_cleanup_attempts_every_resource_when_one_close_fails() -> None:
    callbacks = AmazonIVSCallbacks(
        on_connected=AsyncMock(),
        on_disconnected=AsyncMock(),
        on_client_connected=AsyncMock(),
        on_client_disconnected=AsyncMock(),
        on_error=AsyncMock(),
    )
    client = AmazonIVSTransportClient(
        participant_token="participant-token",
        subscribe_participant_id=None,
        params=AmazonIVSParams(audio_in_enabled=False),
        callbacks=callbacks,
        transport_name="test-transport",
    )

    class ClosingResource:
        def __init__(self, *, fail=False):
            self.closed = False
            self.fail = fail

        async def close(self):
            self.closed = True
            if self.fail:
                raise RuntimeError("close failed")

    subscriber = ClosingResource(fail=True)
    publisher = ClosingResource()
    output_track = ClosingResource()
    client._subscriber_pc = subscriber
    client._publisher_pc = publisher
    client._output_track = output_track

    with pytest.raises(AmazonIVSError, match="cleanup failed"):
        await client._close_connections()

    assert subscriber.closed is True
    assert publisher.closed is True
    assert output_track.closed is True


@pytest.mark.asyncio
async def test_input_cancel_stops_a_blocked_audio_read() -> None:
    input_track = BlockingInputTrack()
    client = FakeAmazonIVSClient(input_track=input_track)
    parent = FakeParentTransport()
    transport = AmazonIVSInputTransport(
        transport=parent,
        client=client,
        params=AmazonIVSParams(),
    )
    await transport.setup(frame_processor_setup(TaskManager()))

    try:
        await transport.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(input_track.read_started.wait(), timeout=1)

        await transport.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)

        assert input_track.read_cancelled.is_set()
        assert transport._receive_task is None
    finally:
        await transport.cleanup()


@pytest.mark.asyncio
async def test_output_interruption_clears_base_and_webrtc_playout_buffers() -> None:
    params = AmazonIVSParams(
        audio_out_sample_rate=24_000,
        audio_out_10ms_chunks=2,
        output_frame_duration_ms=20,
        output_max_buffer_ms=200,
    )
    track = BufferedPCM16AudioTrack(
        sample_rate=24_000,
        frame_duration_ms=20,
        max_buffer_ms=200,
        pace=False,
    )
    client = FakeAmazonIVSClient(output_track=track)
    parent = FakeParentTransport()
    transport = AmazonIVSOutputTransport(
        transport=parent,
        client=client,
        params=params,
    )
    await transport.setup(
        frame_processor_setup(
            TaskManager(),
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=24_000,
        )
    )

    try:
        await transport.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        sender = transport._media_senders[None]
        pcm = np.arange(sender.audio_chunk_size // 2, dtype="<i2").tobytes()
        await transport.process_frame(
            OutputAudioRawFrame(audio=pcm, sample_rate=24_000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await wait_until(lambda: track.buffered_bytes == sender.audio_chunk_size)

        await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

        assert track.buffered_bytes == 0
        assert sender._audio_queue.empty()
    finally:
        await transport.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
        await transport.cleanup()


def test_transport_composes_single_shared_client_and_stable_processors() -> None:
    params = AmazonIVSParams(output_max_buffer_ms=250)
    transport = AmazonIVSTransport(
        "participant-token",
        "source-participant-id",
        params=params,
        input_name="amazon-ivs-input",
        output_name="amazon-ivs-output",
    )

    input_transport = transport.input()
    output_transport = transport.output()

    assert isinstance(transport._client, AmazonIVSTransportClient)
    assert isinstance(input_transport, AmazonIVSInputTransport)
    assert isinstance(output_transport, AmazonIVSOutputTransport)
    assert input_transport is transport.input()
    assert output_transport is transport.output()
    assert input_transport._client is transport._client
    assert output_transport._client is transport._client
    assert input_transport._params is params
    assert output_transport._params is params
    assert input_transport.name == "amazon-ivs-input"
    assert output_transport.name == "amazon-ivs-output"


def test_transport_rejects_subscription_url_without_participant_id() -> None:
    with pytest.raises(RuntimeError, match="subscribe_participant_id"):
        AmazonIVSTransport(
            "participant-token",
            None,
            subscription_url=("https://iad.whip.live-video.net/session/subscribe/user-123"),
        )
