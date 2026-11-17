# TADA

HumeAI TADA is a voice-cloning TTS model built on Llama 3.2. It emits one 512-dimensional acoustic frame per text token through a flow-matching head, then a DAC codec decoder produces 24 kHz audio. `HumeAI/tada-3b-ml` is the multilingual 3B checkpoint; `HumeAI/tada-1b` shares the same code path.

## Requirements

- A reference audio clip and its transcript. The transcript is required because the encoder performs forced alignment between the reference text and audio; without it the prompt representation is wrong.
- A local Llama 3.2 tokenizer or Hugging Face access to `meta-llama/Llama-3.2-1B`. The tokenizer is not shipped with the TADA checkpoints.
- The `HumeAI/tada-codec` repository (encoder, aligners, decoder). Weights are downloaded on first use or can be pointed at with `codec_model_path`.

## Serve

```bash
sgl-omni serve \
  --model-path HumeAI/tada-3b-ml \
  --config examples/configs/tada.yaml \
  --allowed-local-media-path /path/to/references \
  --port 8000
```

```bash
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "HumeAI/tada-3b-ml",
    "input": "Please call Stella and ask her to bring these things with her from the store.",
    "ref_audio": "/path/to/references/ljspeech.wav",
    "ref_text": "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired.",
    "language": "English",
    "response_format": "wav"
  }' \
  -o /tmp/tada.wav
```

The output is 24 kHz mono.

## Languages

`language` accepts the canonical API names: English, Japanese, German, French, Spanish, Chinese, Arabic, Italian, Polish, Portuguese. It defaults to English; set `preprocessing.factory.default_language` to change the server default. English uses the default aligner; every other language loads `aligner-<code>` from `tada-codec`, cached with `aligner_cache_size` (each about 0.9 GB).

Quality tiers: English is primary; Chinese, French, Italian, Japanese, Portuguese, Polish, and German are in the paper's training mix (Mandarin is tier-2). Spanish and Arabic ship aligners but are outside the published mix; treat them as experimental. For best results the reference clip should be in the target language and the transcript must match the audio.

The reference stage runs an alignment health check modeled on the paper's data filter: it flags non-monotonic or out-of-range positions, four-token runs spanning three frames or fewer, gaps over 150 frames (3 s), and implausible frames-per-token. `alignment_check` selects `off`, `warn` (default), or `reject`.

## Continuation

Set `num_extra_steps` in the request (or `preprocessing.factory.num_extra_steps` as the server default) to keep generating text and speech past the transcript. TTS mode is teacher-forced and does not sample text for the input transcript.

## Streaming

Set `"stream": true` to receive audio chunks while the engine walks the text. The codec is non-causal, so the vocoder decodes overlapping windows: a 50-frame chunk with 128 frames of left context and 32 frames of right lookahead. Audio starts once the first `chunk + right` frames exist, and each later chunk is decoded with the context on both sides, then trimmed to its 50 new frames. Concurrent streams decode their ready windows together in one batch (grouped by window length); `stream_max_batch_size` caps that batch. The window knobs are `vocoder.factory.stream_chunk_frames`, `stream_left_context_frames`, `stream_right_context_frames`, and `stream_max_batch_size`.

Streaming removes the non-streaming decoder limit: each window is about 210 frames, so long requests no longer hit the 8192-frame buffer.

## Tuning

`tts_engine.factory` exposes `num_flow_matching_steps` (default 10), `acoustic_cfg_scale` (default 1.8, the paper value for zero-negative guidance; upstream code and UI use 1.6), `duration_cfg_scale` (default 1.0, matching the paper's decision to skip duration guidance), `cfg_schedule`, `time_schedule`, `noise_temperature`, and `negative_condition_source` (`zero` or `prompt`; the upstream dual-row `negative_step_output` mode is not implemented). The streaming knobs live under `vocoder.factory` (`stream_chunk_frames`, `stream_left_context_frames`, `stream_right_context_frames`).

`num_transition_steps` (default 5) is separate from `shift_acoustic`; the official demo uses 0.

## Known limitations

- Reference audio and transcript are required; zero-shot generation is not exposed.
- v1 uses head-local CFG with a zero negative condition. The paper notes this is a supported option, but long-form speaker drift is possible; `negative_condition_source: prompt` is the cheap alternative to A/B on GPU.
- KV retraction/re-prefill is not supported; the engine fails loudly. Size `mem_fraction_static` and `max_running_requests` for the expected concurrency. A TADA token costs about 112 KiB of KV.
- The non-streaming decoder caps expanded audio at 8192 frames (about 164 s at 50 Hz) and raises over that limit. Streaming lifts the cap by decoding bounded windows.
- Memory: roughly 6.4 GB backbone, 1.8 GB flow head, 0.6 GB encoder, 0.9 GB per cached aligner, and 0.65 GB decoder, comfortable on 24 GB.

## Small-GPU tuning (16 GB, e.g. RTX 5070 Ti)

The 3B checkpoint is tight on 16 GB: roughly 8.2 GB of engine weights (6.4 GB Llama plus 1.8 GB flow head) and about 2.15 GB of codec stages share one card. Start from `examples/configs/tada.yaml` with:

- `tts_engine.engine.mem_fraction_static: 0.6` (the engine weights alone need more than 0.5)
- `tts_engine.engine.max_running_requests: 2` and `tts_engine.factory.max_running_requests: 2`
- `reference_encode.factory.aligner_cache_size: 1` (one 0.9 GB aligner instead of two)
- `preprocessing.factory.max_target_tokens: 512`, and reference clips of 10-20 seconds
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation

If it still OOMs, drop `max_running_requests` to 1. `HumeAI/tada-1b` is much lighter (about 2.5 GB backbone plus a roughly 0.4 GB head) and is the recommended first end-to-end check.

The RTX 5070 Ti is Blackwell (compute capability 12.0, sm_120). The pinned stack uses CUDA 13 wheels, which should cover it, but verify after GPU passthrough:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))"
# expect (12, 0) and the 5070 Ti name
```

The 9000-series consumer cards are newer than most of the pinned SGLang kernels; if a FlashInfer or FlashAttention kernel rejects sm_120, try the eager fallbacks first (`disable_cuda_graph` is already on) and report the failing kernel. A VM must have the GPU passed through; without passthrough only the CPU stages run.
