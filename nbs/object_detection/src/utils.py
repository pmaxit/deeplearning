import torch

class CTCDecoder:
    def __init__(self, vocab, blank_idx=0):
        self.blank_idx = blank_idx
        self.vocab = vocab
        self.idx2char = {i + 1: c for i, c in enumerate(vocab)}  # start from 1
        self.idx2char[blank_idx] = ''  # blank = 0

    def __call__(self, log_probs):
        preds = torch.argmax(log_probs, dim=2)  # (T, B)
        return [self._decode_seq(preds[:, i]) for i in range(preds.size(1))]

    def _decode_seq(self, seq):
        seq = seq.cpu().tolist()
        return ''.join(
            self.idx2char.get(idx, '') for i, idx in enumerate(seq)
            if idx != self.blank_idx and (i == 0 or idx != seq[i - 1])
        )

    def target_to_str(self, targets):
        return [''.join(self.idx2char.get(idx.item(), '') for idx in seq if idx.item() > 0)
                for seq in targets.cpu()]
