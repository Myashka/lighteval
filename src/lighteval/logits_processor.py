from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from typing import List
from transformers import PreTrainedTokenizer
from loguru import logger


def get_new_line_tokens(tokenizer: PreTrainedTokenizer):
    new_line_tokens = [token for token in tokenizer.get_vocab().values() if tokenizer.decode(token).endswith("\n")]

    return set(new_line_tokens)


def text_to_token(tokenizer: PreTrainedTokenizer, text: str, last: bool):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if not last and len(tokens) > 2:
        raise Exception(f"Can't convert {text} to token. It has {len(tokens)} tokens.")

    return tokens[-1]


LogitsProcessor = Union[
    Callable[[List[int], torch.Tensor], torch.Tensor], Callable[[List[int], List[int], torch.Tensor], torch.Tensor]
]
"""LogitsProcessor is a function that takes a list
of previously generated tokens, the logits tensor
for the next token and, optionally, prompt tokens as a
first argument, and returns a modified tensor of logits
to sample from."""


class EarlyExitThinkLogitsProcessor:
    def __init__(
        self,
        target_token_text: str,
        tokenizer: PreTrainedTokenizer,
        threshold: float,
        complete_sentences: bool = False,
    ):
        self.tokenizer = tokenizer
        self.target_token_id = tokenizer.convert_tokens_to_ids(target_token_text)
        self.threshold = threshold
        self.counter = 0

        self.complete_sentences = complete_sentences

        self.stop_tokens = {
            text_to_token(tokenizer, ".", last=True),
            text_to_token(tokenizer, "!", last=True),
            text_to_token(tokenizer, "?", last=True),
            text_to_token(tokenizer, ";", last=True),
        }

        self.new_line_tokens = get_new_line_tokens(tokenizer)
        self.stop_tokens.update(self.new_line_tokens)

    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        if not past_token_ids:
            return logits
        if self.target_token_id in past_token_ids:
            return logits
        if self.complete_sentences and past_token_ids[-1] not in self.stop_tokens:
            return logits

        probs = F.softmax(logits, dim=-1)
        target_prob = probs[self.target_token_id].item()

        if target_prob > self.threshold:
            self.counter += 1
            mask = torch.full_like(logits, float('-inf'))
            mask[self.target_token_id] = 0
            return logits + mask

        return logits
