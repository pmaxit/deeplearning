from fastai.vision.all import *

class CTCAccuracy(Metric):
    def __init__(self, decoder):  # Pass in CTCDecoder instance
        self.decoder = decoder

    def reset(self): self.correct, self.total = 0, 0

    def accumulate(self, learn):
        log_probs, targets = learn.pred, learn.y  # (T, B, C), (B, S)
        pred_strs = self.decoder(log_probs)
        true_strs = self.decoder.target_to_str(targets)
        self.correct += sum(p == t for p, t in zip(pred_strs, true_strs))
        self.total += len(pred_strs)

    @property
    def value(self):
        return self.correct / self.total if self.total > 0 else None

    @property
    def name(self): return "acc"
