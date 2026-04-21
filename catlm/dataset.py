"""CatLM dataset loading."""

import json

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


ASSISTANT_MARKER = "\n<|im_start|>assistant\n"


class CatDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_len: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.samples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                enc = self.tokenizer.encode(text)
                ids = enc.ids
                offsets = enc.offsets
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    offsets = offsets[:max_len]
                if len(ids) >= 2:
                    masked = self._mask_prompt_targets(text, ids, offsets)
                    if masked is not None:
                        self.samples.append(masked)

    def _mask_prompt_targets(self, text, ids, offsets):
        marker_pos = text.find(ASSISTANT_MARKER)
        if marker_pos == -1:
            return None

        assistant_start = marker_pos + len(ASSISTANT_MARKER)
        y = ids[1:].copy()

        seen_assistant_token = False
        for target_idx in range(1, len(ids)):
            token_start, token_end = offsets[target_idx]
            include = token_start >= assistant_start
            if include:
                seen_assistant_token = True
            elif (
                seen_assistant_token
                and ids[target_idx] == 2  # <|im_end|>
                and token_start == 0
                and token_end == 0
            ):
                include = True

            if not include:
                y[target_idx - 1] = 0

        if all(token_id == 0 for token_id in y):
            return None

        return ids, y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, y = self.samples[idx]
        x = ids[:-1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
    return padded_x, padded_y


def get_dataloader(path, tokenizer_path, max_len=512, batch_size=32, shuffle=True):
    dataset = CatDataset(path, tokenizer_path, max_len)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
