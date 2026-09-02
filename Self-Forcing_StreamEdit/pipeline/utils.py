import os
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
import torch


# -------------------- tokenizer helper --------------------
def find_phrase_token_indices(tokenizer, prompt: str | List[str], phrase: str | List[str],
                              add_special_tokens_prompt=True,
                              add_special_tokens_phrase=False) -> List[int]:
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(phrase, str):
        phrase = [phrase]
    ret_list = []
    for pr, ph in zip(prompt, phrase):
        ret_list.append(
            _find_phrase_token_indices_single(
                tokenizer, pr, ph, add_special_tokens_prompt, add_special_tokens_phrase
            )
        )
    return ret_list


def find_phrase_group_token_indices(
    tokenizer,
    prompt: str | List[str],
    phrases: str | List[str],
    add_special_tokens_prompt=True,
    add_special_tokens_phrase=False,
) -> List[List[int]]:
    """Find the union of several phrase spans for every prompt.

    ``find_phrase_token_indices`` pairs one phrase with one prompt.  Semantic
    role competition instead needs several edit or preserve phrases from the
    same target prompt.  This helper keeps the result batch-shaped while
    deduplicating overlapping phrase spans.  Missing phrases are rejected: a
    silently empty preserve group would disable the background veto while the
    run still appeared to use it.
    """
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    phrase_group = [phrases] if isinstance(phrases, str) else list(phrases)
    if not phrase_group or any(not phrase for phrase in phrase_group):
        raise ValueError("Semantic phrase groups must be non-empty")

    result = []
    for prompt_index, current_prompt in enumerate(prompts):
        indices = set()
        for current_phrase in phrase_group:
            current = _find_phrase_token_indices_single(
                tokenizer,
                current_prompt,
                current_phrase,
                add_special_tokens_prompt,
                add_special_tokens_phrase,
            )
            if not current:
                raise ValueError(
                    f"Phrase {current_phrase!r} was not found in target "
                    f"prompt {prompt_index}: {current_prompt!r}"
                )
            indices.update(current)
        result.append(sorted(indices))
    return result

def _find_phrase_token_indices_single(tokenizer, prompt: str, phrase: str,
                              add_special_tokens_prompt=True,
                              add_special_tokens_phrase=False) -> List[int]:
    if not phrase:
        return []
    enc = tokenizer(prompt, padding=False, truncation=False,
                    add_special_tokens=add_special_tokens_prompt,
                    return_attention_mask=False, return_tensors=None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    def _clean(ids): return [i for i in ids if i not in {pad_id, eos_id, None}]
    ids = _clean(enc["input_ids"])
    phrase_ids = _clean(tokenizer(phrase, add_special_tokens=add_special_tokens_phrase)["input_ids"])
    if len(phrase_ids) == 0 or len(ids) < len(phrase_ids):
        return []
    for i in range(0, len(ids) - len(phrase_ids) + 1):
        if ids[i:i+len(phrase_ids)] == phrase_ids:
            return list(range(i, i+len(phrase_ids)))
    return []
