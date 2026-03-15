"""PMI-based quality scoring and per-sample score selection.

Provides handlers for computing PMI quality scores and DiT alignment
scores on generated audio samples.
"""
import gc
import threading
import traceback

import torch
import gradio as gr

from acestep.ui.gradio.i18n import t

# Serialise all scoring calls.
# Gradio triggers auto-score for every sample near-simultaneously.
# Without a lock, concurrent DiT passes double the VRAM pressure.
_SCORING_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# VRAM management helpers for scoring
# ---------------------------------------------------------------------------

def _move_to_cpu_if_cuda(module):
    """Move an nn.Module to CPU if currently on CUDA. Returns original device or None."""
    if module is None or not hasattr(module, 'parameters'):
        return None
    try:
        dev = next(module.parameters()).device
        if dev.type == 'cuda':
            module.cpu()
            return dev
    except StopIteration:
        pass
    return None


def _offload_for_dit_scoring(dit_handler, llm_handler):
    """Offload models not needed during DiT alignment scoring to CPU.

    DiT alignment scoring (get_lyric_score) only needs dit_handler.model
    (the DiT decoder).  Everything else — text_encoder, VAE, and the entire
    LLM — can be temporarily moved to system RAM to free VRAM for the
    attention tensor burst that scoring produces.

    On a 5090 (31.5 GB VRAM) with SFT + 4B LM loaded (23.8 GB baseline):
      • text_encoder  ~0.6 GB
      • VAE           ~0.3 GB
      • LLM 4B        ~8.0 GB
      Total freed     ~9 GB  → scoring runs cleanly at ~22 GB, no overflow.

    Returns a restore-map consumed by _restore_after_dit_scoring.
    """
    restore = {}

    # dit_handler: text_encoder and VAE are not used by the decoder scoring pass
    if dit_handler is not None:
        for attr in ('text_encoder', 'vae'):
            m = getattr(dit_handler, attr, None)
            dev = _move_to_cpu_if_cuda(m)
            if dev is not None:
                restore[('dit', attr)] = dev

    # llm_handler: move every nn.Module instance to CPU
    # (LLM finished PMI scoring before this point; it is not needed for DiT alignment)
    if llm_handler is not None:
        for attr, val in vars(llm_handler).items():
            if isinstance(val, torch.nn.Module):
                dev = _move_to_cpu_if_cuda(val)
                if dev is not None:
                    restore[('llm', attr)] = dev

    if restore:
        gc.collect()
        torch.cuda.empty_cache()

    return restore


def _restore_after_dit_scoring(dit_handler, llm_handler, restore):
    """Restore models offloaded by _offload_for_dit_scoring back to GPU."""
    if not restore:
        return

    for (handler_type, attr), dev in restore.items():
        handler = dit_handler if handler_type == 'dit' else llm_handler
        if handler is not None:
            m = getattr(handler, attr, None)
            if m is not None:
                m.to(dev)

    gc.collect()
    torch.cuda.empty_cache()


def _offload_for_pmi_scoring(dit_handler):
    """Offload the DiT model to CPU before PMI (LLM) scoring.

    PMI scoring loads a separate HF copy of the 4B LM (~8 GB) onto the GPU.
    With the baseline load (23.8 GB) that pushes total VRAM to ~32 GB — over
    the 5090's 31.5 GB budget.  The DiT model (~15 GB) is not needed during
    PMI scoring, so we move it to system RAM first, score cleanly, then restore.

    Returns a restore-map consumed by _restore_after_pmi_scoring.
    """
    restore = {}
    if dit_handler is None:
        return restore

    # Offload the DiT decoder (largest single occupant) and any other
    # nn.Module attributes except text_encoder and vae (those are small
    # and may be needed for conditioning lookups inside PMI scoring).
    for attr, val in vars(dit_handler).items():
        if isinstance(val, torch.nn.Module) and attr not in ('text_encoder', 'vae'):
            dev = _move_to_cpu_if_cuda(val)
            if dev is not None:
                restore[('dit', attr)] = dev

    if restore:
        gc.collect()
        torch.cuda.empty_cache()

    return restore


def _restore_after_pmi_scoring(dit_handler, restore):
    """Restore models offloaded by _offload_for_pmi_scoring back to GPU."""
    if not restore:
        return
    for (handler_type, attr), dev in restore.items():
        if handler_type == 'dit' and dit_handler is not None:
            m = getattr(dit_handler, attr, None)
            if m is not None:
                m.to(dev)
    gc.collect()
    torch.cuda.empty_cache()


def calculate_score_handler(
    llm_handler,
    audio_codes_str,
    caption,
    lyrics,
    lm_metadata,
    bpm,
    key_scale,
    time_signature,
    audio_duration,
    vocal_language,
    score_scale,
    dit_handler,
    extra_tensor_data,
    inference_steps,
):
    """Calculate PMI-based quality score for generated audio.

    PMI (Pointwise Mutual Information) removes condition bias:
    ``score = log P(condition|codes) - log P(condition)``

    Falls back to DiT alignment scoring when audio codes are unavailable
    (e.g. Cover/Repaint modes).

    Args:
        llm_handler: LLM handler instance.
        audio_codes_str: Generated audio codes string.
        caption: Caption text used for generation.
        lyrics: Lyrics text used for generation.
        lm_metadata: LM-generated metadata dictionary.
        bpm: BPM value.
        key_scale: Key scale value.
        time_signature: Time signature value.
        audio_duration: Audio duration value.
        vocal_language: Vocal language value.
        score_scale: Sensitivity scale parameter.
        dit_handler: DiT handler instance (for alignment scoring).
        extra_tensor_data: Dictionary containing tensors for the specific sample.
        inference_steps: Number of inference steps used.

    Returns:
        Score display string.
    """
    from acestep.core.scoring.lm_score import calculate_pmi_score_per_condition

    has_audio_codes = audio_codes_str and audio_codes_str.strip()
    has_dit_alignment_data = dit_handler and extra_tensor_data and lyrics and lyrics.strip()

    if not has_audio_codes and not has_dit_alignment_data:
        return t("messages.no_codes")

    try:
        scores_per_condition = {}
        global_score = 0.0
        alignment_report = ""

        # PMI-based scoring (requires audio codes and LLM)
        # Before loading the HF scoring model (~8 GB) to GPU, offload the DiT
        # model (~15 GB) to CPU — it is not needed during PMI scoring.
        # This keeps peak VRAM within budget for the forward pass.
        if has_audio_codes:
            if not llm_handler.llm_initialized:
                if not has_dit_alignment_data:
                    return t("messages.lm_not_initialized")
            else:
                metadata = {}
                if lm_metadata and isinstance(lm_metadata, dict):
                    metadata.update(lm_metadata)
                if bpm is not None and 'bpm' not in metadata:
                    try:
                        metadata['bpm'] = int(bpm)
                    except Exception:
                        pass
                if caption and 'caption' not in metadata:
                    metadata['caption'] = caption
                if audio_duration is not None and audio_duration > 0 and 'duration' not in metadata:
                    try:
                        metadata['duration'] = int(audio_duration)
                    except Exception:
                        pass
                if key_scale and key_scale.strip() and 'keyscale' not in metadata:
                    metadata['keyscale'] = key_scale.strip()
                if vocal_language and vocal_language.strip() and 'language' not in metadata:
                    metadata['language'] = vocal_language.strip()
                if time_signature and time_signature.strip() and 'timesignature' not in metadata:
                    metadata['timesignature'] = time_signature.strip()

                _pmi_restore = _offload_for_pmi_scoring(dit_handler)
                try:
                    scores_per_condition, global_score, _status = calculate_pmi_score_per_condition(
                        llm_handler=llm_handler,
                        audio_codes=audio_codes_str,
                        caption=caption or "",
                        lyrics=lyrics or "",
                        metadata=metadata if metadata else None,
                        temperature=1.0,
                        topk=10,
                        score_scale=score_scale,
                    )
                finally:
                    _restore_after_pmi_scoring(dit_handler, _pmi_restore)

        # DiT alignment scoring
        # PMI scoring (LLM) is complete at this point.  Before running the DiT
        # decoder with output_attentions=True (which produces a large attention
        # tensor burst), offload the LLM + text_encoder + VAE to CPU so scoring
        # fits entirely in VRAM.  Models are restored immediately after.
        if has_dit_alignment_data:
            _restore_map = _offload_for_dit_scoring(dit_handler, llm_handler)
            try:
                align_result = dit_handler.get_lyric_score(
                    pred_latent=extra_tensor_data.get('pred_latent'),
                    encoder_hidden_states=extra_tensor_data.get('encoder_hidden_states'),
                    encoder_attention_mask=extra_tensor_data.get('encoder_attention_mask'),
                    context_latents=extra_tensor_data.get('context_latents'),
                    lyric_token_ids=extra_tensor_data.get('lyric_token_ids'),
                    vocal_language=vocal_language or "en",
                    inference_steps=int(inference_steps),
                    seed=42,
                )
                if align_result.get("success"):
                    lm_align = align_result.get("lm_score", 0.0)
                    dit_align = align_result.get("dit_score", 0.0)
                    alignment_report = (
                        f"  • llm lyrics alignment score: {lm_align:.4f}\n"
                        f"  • dit lyrics alignment score: {dit_align:.4f}\n"
                        "\n(Measures how well lyrics timestamps match audio energy using Cross-Attention)"
                    )
                else:
                    alignment_report = f"\n⚠️ Alignment Score Failed: {align_result.get('error', 'Unknown error')}"
            except Exception as e:
                alignment_report = f"\n⚠️ Alignment Score Error: {str(e)}"
            finally:
                # Always restore — even if scoring raised an exception
                _restore_after_dit_scoring(dit_handler, llm_handler, _restore_map)

        # Format display string
        if has_audio_codes and llm_handler.llm_initialized:
            if global_score == 0.0 and not scores_per_condition:
                if alignment_report and not alignment_report.startswith("\n⚠️"):
                    return "📊 DiT Alignment Scores (LM codes not available):\n" + alignment_report
                return t("messages.score_failed", error="PMI scoring returned no results")

            condition_lines = [
                f"  • {name}: {val:.4f}"
                for name, val in sorted(scores_per_condition.items())
            ]
            conditions_display = "\n".join(condition_lines) if condition_lines else "  (no conditions)"
            final_output = (
                f"✅ Global Quality Score: {global_score:.4f} (0-1, higher=better)\n\n"
                f"📊 Per-Condition Scores (0-1):\n{conditions_display}\n"
            )
            if alignment_report:
                final_output += alignment_report + "\n"
            final_output += "Note: Metadata uses Top-k Recall, Caption/Lyrics use PMI"
            return final_output
        else:
            if alignment_report and not alignment_report.startswith("\n⚠️"):
                return "📊 DiT Alignment Scores (LM codes not available for Cover/Repaint mode):\n" + alignment_report
            elif alignment_report:
                return alignment_report
            return "⚠️ No scoring data available"

    except Exception as e:
        return t("messages.score_error", error=str(e)) + f"\n{traceback.format_exc()}"


def calculate_score_handler_with_selection(
    dit_handler,
    llm_handler,
    sample_idx,
    score_scale,
    current_batch_index,
    batch_queue,
):
    """Calculate quality score for a specific sample from batch queue data.

    Reads all parameters from the historical batch rather than current UI
    values, ensuring scores reflect the actual generation settings.

    Args:
        dit_handler: DiT handler instance.
        llm_handler: LLM handler instance.
        sample_idx: 1-based sample index (1-8).
        score_scale: Sensitivity scale parameter.
        current_batch_index: Current batch index.
        batch_queue: Batch queue dict.

    Returns:
        Tuple of ``(score_display_update, accordion_update, batch_queue)``.
    """
    if current_batch_index not in batch_queue:
        return gr.skip(), gr.skip(), batch_queue

    batch_data = batch_queue[current_batch_index]
    params = batch_data.get("generation_params", {})

    caption = params.get("captions", "")
    lyrics = params.get("lyrics", "")
    bpm = params.get("bpm")
    key_scale = params.get("key_scale", "")
    time_signature = params.get("time_signature", "")
    audio_duration = params.get("audio_duration", -1)
    vocal_language = params.get("vocal_language", "")
    inference_steps = params.get("inference_steps", 8)
    lm_metadata = batch_data.get("lm_generated_metadata", None)

    stored_codes = batch_data.get("codes", "")
    stored_allow_lm_batch = batch_data.get("allow_lm_batch", False)

    audio_codes_str = ""
    if stored_allow_lm_batch and isinstance(stored_codes, list):
        if 0 <= sample_idx - 1 < len(stored_codes):
            code_item = stored_codes[sample_idx - 1]
            audio_codes_str = code_item if isinstance(code_item, str) else ""
    else:
        audio_codes_str = stored_codes if isinstance(stored_codes, str) else ""

    # Extract tensor data for alignment scoring
    extra_tensor_data = None
    extra_outputs = batch_data.get("extra_outputs", {})
    if extra_outputs and dit_handler:
        pred_latents = extra_outputs.get("pred_latents")
        if pred_latents is not None:
            idx0 = sample_idx - 1
            if 0 <= idx0 < pred_latents.shape[0]:
                try:
                    extra_tensor_data = {
                        "pred_latent": pred_latents[idx0:idx0 + 1],
                        "encoder_hidden_states": extra_outputs.get("encoder_hidden_states")[idx0:idx0 + 1],
                        "encoder_attention_mask": extra_outputs.get("encoder_attention_mask")[idx0:idx0 + 1],
                        "context_latents": extra_outputs.get("context_latents")[idx0:idx0 + 1],
                        "lyric_token_ids": extra_outputs.get("lyric_token_ids")[idx0:idx0 + 1],
                    }
                    if any(v is None for v in extra_tensor_data.values()):
                        extra_tensor_data = None
                except Exception as e:
                    print(f"Error slicing tensor data for score: {e}")
                    extra_tensor_data = None

    with _SCORING_LOCK:
        score_display = calculate_score_handler(
            llm_handler, audio_codes_str, caption, lyrics, lm_metadata,
            bpm, key_scale, time_signature, audio_duration, vocal_language,
            score_scale, dit_handler, extra_tensor_data, inference_steps,
        )

        # Free only the sliced tensor data for THIS sample.
        # Do NOT clear the batch-level extra_outputs — other samples in the
        # same batch still need their slices from it.  extra_outputs tensors
        # are CPU RAM (moved there during generation), not VRAM, so leaving
        # them until the batch is finished is fine.
        if extra_tensor_data is not None:
            for v in extra_tensor_data.values():
                if isinstance(v, torch.Tensor):
                    del v
            extra_tensor_data.clear()
        gc.collect()
        torch.cuda.empty_cache()

    if current_batch_index in batch_queue:
        if "scores" not in batch_queue[current_batch_index]:
            batch_queue[current_batch_index]["scores"] = [""] * 8
        batch_queue[current_batch_index]["scores"][sample_idx - 1] = score_display

    return (
        gr.update(value=score_display, visible=True),
        gr.skip(),
        batch_queue,
    )
