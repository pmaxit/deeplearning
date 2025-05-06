# train_captcha_fastai.py
from fastai.vision.all import *
from captcha.image import ImageCaptcha
import string, random, torch.nn.functional as F

# Constants
CHARS = string.ascii_uppercase + string.digits
VOCAB = list(CHARS)
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 = blank for CTC
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
BLANK_TOKEN = 0
IMG_WIDTH, IMG_HEIGHT = 160, 60
MAX_LEN = 5

# 1. Generate synthetic captcha dataset
def create_sample():
    generator = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
    label = ''.join(random.choices(VOCAB, k=MAX_LEN))
    img = generator.generate_image(label).convert("RGB")
    return PILImage.create(img), label

class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, size): self.items = [create_sample() for _ in range(size)]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

# 2. Encoding/Decoding
def encode_label(lbl): return tensor([CHAR2IDX[c] for c in lbl], dtype=torch.long)
def decode_prediction(preds):
    preds = preds.argmax(-1)
    result = []
    for seq in preds:
        seq = seq.tolist()
        prev = -1
        s = [IDX2CHAR[c] for c in seq if c != BLANK_TOKEN and c != prev and (prev := c)]
        result.append(''.join(s))
    return result

# 3. Custom transforms
class EncodeTransform(Transform):
    def __init__(self): pass
    def encodes(self, x): return encode_label(x)

# 4. DataBlock
def create_sample():
    generator = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
    label = ''.join(random.choices(VOCAB, k=MAX_LEN))
    img = generator.generate_image(label).convert("RGB")
    return img, label

def get_dls(train_sz=1000, valid_sz=200, bs=64):
    # Step 1: Create image-label pairs
    raw_data = [create_sample() for _ in range(train_sz + valid_sz)]
    
    # Step 2: Convert images to fastai-readable PILImage format
    items = [[PILImage.create(img), lbl] for img, lbl in raw_data]

    # Step 3: Create DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, TransformBlock(type_tfms=EncodeTransform())),
        get_items=lambda _: items,  # items is a list of (image, label) tuples
        splitter=IndexSplitter(range(train_sz, train_sz + valid_sz)),
        get_x=lambda x: x[0],
        get_y=lambda x: x[1],
        item_tfms=Resize((IMG_HEIGHT, IMG_WIDTH)),
        batch_tfms=Normalize.from_stats(*imagenet_stats)
    )

    # Step 4: Return DataLoaders
    return dblock.dataloaders(source=None, bs=bs)


# 5. Model
class CRNN(nn.Module):
    def __init__(self, num_classes=len(VOCAB) + 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
        )
        self.rnn = nn.LSTM(128 * 7, 256, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# 6. Custom CTC Loss
class CTCLoss(Module):
    def __init__(self): self.loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, zero_infinity=True)
    def forward(self, preds, lbls):
        log_probs = preds.log_softmax(2)
        input_lengths = tensor([preds.size(1)] * preds.size(0), dtype=torch.long)
        target_lengths = tensor([len(l[l != 0]) for l in lbls], dtype=torch.long)
        return self.loss_fn(log_probs.permute(1, 0, 2), lbls, input_lengths, target_lengths)

# 7. Show predictions
def show_preds(dl, learn, n=6):
    xb, yb = first(dl)
    preds = learn.model(xb)
    decoded = decode_prediction(preds.cpu())
    true = [''.join([IDX2CHAR[int(i)] for i in y if i != 0]) for y in yb]
    for i in range(n):
        show_image(xb[i], title=f"Pred: {decoded[i]}\nTrue: {true[i]}")
    plt.show()

# 8. Train
if __name__ == "__main__":
    dls = get_dls()
    model = CRNN()
    learn = Learner(dls, model, loss_func=CTCLoss(), metrics=[]).to_fp32()
    learn.fit_one_cycle(10, 1e-3)
    show_preds(dls.valid, learn)
