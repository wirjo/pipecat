# Spike: new AWS SDK for Transcribe streaming (`aws-sdk-transcribe-streaming`)

Not a PR candidate yet. This is an evaluation-only spike proving the new
official SDK actually works, before anyone proposes it as a replacement for
Pipecat's current hand-rolled WebSocket + SigV4 Transcribe implementation.

## What this proves

`spike_test.py` ran successfully against **live AWS Transcribe** (account
575108946562, us-east-1, 2026-10-08) using `aws-sdk-transcribe-streaming`
0.11.0 + `awscrt` 0.32.2, streaming `scripts/provider-watch/assets/speech-16k.wav`
(16kHz/16-bit/mono PCM) in 100ms chunks, paced to real time.

Real output:

```
[spike] config resolved
[spike] client created, starting stream transcription
[spike] stream started after 0.02s
[spike] source wav format: (1, 2, 16000)
[spike] sent 51 audio chunks (5.06s of audio)
[spike] [FINAL] Quick brown fox jumps over the lazy dog. Pippica makes voice agents easy to build.
[spike] total elapsed: 5.15s
[spike] total transcript events captured: 11
```

Partial results streamed in correctly throughout; the final transcript
arrived promptly after audio ended (5.15s elapsed for 5.06s of real-time
audio — i.e. ~90ms of end-to-end overhead, not accounting for the fixed
~20ms stream-start latency separately logged above).

## What's gained vs. the current hand-rolled implementation

The new SDK's `StartStreamTranscriptionInput` model has a far richer, fully
typed settings surface than what Pipecat's `AWSTranscribeSTTSettings`
exposes today:

```
language_code, media_sample_rate_hertz, media_encoding, vocabulary_name,
session_id, vocabulary_filter_name, vocabulary_filter_method,
show_speaker_label, enable_channel_identification, number_of_channels,
enable_partial_results_stabilization, partial_results_stability,
content_identification_type, content_redaction_type, pii_entity_types,
language_model_name, identify_language, language_options,
preferred_language, identify_multiple_languages, vocabulary_names,
vocabulary_filter_names, session_resume_window, transcript_format
```

This covers everything proposed in `feat/transcribe-expose-settings`
(vocabulary, channel ID, speaker labels) *plus* PII redaction, automatic/
multi-language identification, custom language models, and even
`session_resume_window` (built-in session resumption) — all typed, no
hand-built query strings.

## What it costs

- **Developer Preview.** AWS's own docs: "intended for evaluation and
  testing only... Do not use it for production workloads. For production
  applications, use the AWS SDK for Python (Boto3)." The GitHub repo
  (`aws/aws-sdk-python`) says the client "is still under active
  development."
- **New native dependency.** Bidirectional streaming *requires* the AWS CRT
  HTTP client (`awscrt`, a compiled extension, installed via the
  `aws-sdk-transcribe-streaming[awscrt]` extra). Checked whether Nova
  Sonic's `aws_sdk_bedrock_runtime` (same Smithy SDK family, already a
  Pipecat dependency) already pulls this in — **it does not**:
  `aws_sdk_bedrock_runtime`'s default transport is `smithy-http[aiohttp]`;
  `awscrt` is only an optional extra it doesn't use by default, and
  `nova_sonic/llm.py` has zero references to `AWSCRTHTTPClient`. So this
  would be a genuinely new dependency class for Pipecat's `aws` extras, not
  something already riding along with Nova Sonic.
- **Same Python floor.** Requires Python 3.12+, matching the existing
  `aws-nova-sonic` extra's constraint — not a new constraint on its own,
  but another product tied to the same floor.
- Confirmed (not re-tested here) that AWS's own community SDK
  (`awslabs/amazon-transcribe-streaming-sdk`) is deprecated *in favor of
  this one* — "This SDK is deprecated. Please use the new official SDK
  instead: aws-sdk-transcribe-streaming."

## Recommendation

Real evidence says the SDK works and is a strict feature superset of the
current implementation. But the explicit "Developer Preview / do not use
in production" label plus the new native-dependency requirement are real
reasons not to propose swapping Pipecat's production Transcribe
implementation onto it yet. Suggested next step: watch for it to graduate
out of Developer Preview (or for AWS to publish a stability commitment),
and in the meantime ship the small, safe win on `feat/transcribe-expose-
settings` (exposing the vocabulary/channel/speaker-label settings that are
*already* wired into the current hand-rolled implementation) rather than
blocking that on this bigger, riskier migration.
