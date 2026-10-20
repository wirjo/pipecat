#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AWS Transcribe STT connection settings."""

import urllib.parse
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.aws.utils import get_presigned_url

CREDENTIALS = {"access_key": "AKIDEXAMPLE", "secret_key": "secret", "session_token": None}


def _query(url: str) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))


def _param_names(url: str) -> list[str]:
    query = urllib.parse.urlparse(url).query
    return [pair.split("=", 1)[0] for pair in query.split("&")]


def test_presigned_url_includes_vocabulary_and_speaker_params():
    """Vocabulary and speaker options are sent as query parameters."""
    url = get_presigned_url(
        region="us-east-1",
        credentials=CREDENTIALS,
        language_code="en-US",
        vocabulary_name="product-terms",
        vocabulary_filter_name="profanity",
        vocabulary_filter_method="remove",
        show_speaker_label=True,
    )
    query = _query(url)

    assert query["vocabulary-name"] == "product-terms"
    assert query["vocabulary-filter-name"] == "profanity"
    assert query["vocabulary-filter-method"] == "remove"
    assert query["show-speaker-label"] == "true"


def test_presigned_url_signed_params_are_sorted():
    """SigV4 requires the canonical query string in sorted order."""
    url = get_presigned_url(
        region="us-east-1",
        credentials=CREDENTIALS,
        language_code="en-US",
        vocabulary_name="product-terms",
        vocabulary_filter_name="profanity",
        vocabulary_filter_method="tag",
        show_speaker_label=True,
    )
    signed = [name for name in _param_names(url) if name != "X-Amz-Signature"]

    assert signed == sorted(signed)


def test_presigned_url_omits_unset_options():
    """Unset options leave the URL as it was before they existed."""
    query = _query(
        get_presigned_url(region="us-east-1", credentials=CREDENTIALS, language_code="en-US")
    )

    for name in (
        "vocabulary-name",
        "vocabulary-filter-name",
        "vocabulary-filter-method",
        "show-speaker-label",
        "enable-channel-identification",
    ):
        assert name not in query


def _service(**settings) -> AWSTranscribeSTTService:
    return AWSTranscribeSTTService(
        aws_access_key_id="AKIDEXAMPLE",
        api_key="secret",
        region="us-east-1",
        sample_rate=16000,
        settings=AWSTranscribeSTTService.Settings(**settings),
    )


async def _connect_and_capture(service: AWSTranscribeSTTService) -> dict:
    with (
        patch("pipecat.services.aws.stt.get_presigned_url", return_value="wss://x") as presign,
        patch.object(service, "_websocket_connect", AsyncMock()),
        patch.object(service, "_call_event_handler", AsyncMock()),
    ):
        await service._connect_websocket()
    return presign.call_args.kwargs


def test_defaults_leave_options_off():
    """By default no vocabulary is used and speakers are not labelled."""
    service = _service()

    assert service._settings.vocabulary_name is None
    assert service._settings.vocabulary_filter_name is None
    assert service._settings.vocabulary_filter_method is None
    assert service._settings.show_speaker_label is False


@pytest.mark.asyncio
async def test_settings_are_passed_to_presigned_url():
    """Settings reach the signed connection URL."""
    service = _service(
        vocabulary_name="product-terms",
        vocabulary_filter_name="profanity",
        vocabulary_filter_method="tag",
        show_speaker_label=True,
    )
    kwargs = await _connect_and_capture(service)

    assert kwargs["vocabulary_name"] == "product-terms"
    assert kwargs["vocabulary_filter_name"] == "profanity"
    assert kwargs["vocabulary_filter_method"] == "tag"
    assert kwargs["show_speaker_label"] is True
    assert kwargs["enable_channel_identification"] is False


@pytest.mark.asyncio
async def test_filter_method_without_filter_name_is_dropped():
    """AWS rejects a filter method with no filter, so it is not sent."""
    service = _service(vocabulary_filter_method="remove")
    kwargs = await _connect_and_capture(service)

    assert kwargs["vocabulary_filter_method"] is None


@pytest.mark.asyncio
async def test_changing_vocabulary_reconnects():
    """Vocabulary is bound to the connection, so an update reconnects."""
    service = _service()
    service._websocket = object()

    with (
        patch.object(service, "_disconnect", AsyncMock()) as disconnect,
        patch.object(service, "_connect", AsyncMock()) as connect,
    ):
        changed = await service._update_settings(
            AWSTranscribeSTTService.Settings(vocabulary_name="product-terms")
        )

    assert "vocabulary_name" in changed
    disconnect.assert_awaited_once()
    connect.assert_awaited_once()
    assert service._settings.vocabulary_name == "product-terms"
