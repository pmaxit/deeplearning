# Constants and setup
import torch, string, random, io
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from captcha.image import ImageCaptcha
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from transformers import PreTrainedModel, PretrainedConfig, TrainingArguments, Trainer
import os
import numpy as np
import evaluate
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

CHARS = string.ascii_uppercase + string.digits
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARS)}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}
NUM_CLASSES = len(CHAR_TO_ID) + 1
IMG_WIDTH, IMG_HEIGHT = 200, 60
MAX_LABEL_LEN = 5
PAD_TOKEN = -1
BLANK_TOKEN = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_captcha_dataset(size=100):
    gen = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
    features = Features({'image': HFImage(), 'label': Value('string')})
    data = []
    for _ in range(size):
        label = ''.join(random.choices(CHARS, k=MAX_LABEL_LEN))
        img = gen.generate_image(label).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        data.append({'image': {'bytes': buf.getvalue()}, 'label': label})
    return Dataset.from_list(data, features=features)

def encode_label(text):
    return [CHAR_TO_ID[c] for c in text]

# Torch transforms
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_for_hf(batch):
    pixel_values, labels, lengths = [], [], []
    for img, text in zip(batch["image"], batch["label"]):
        img = Image.open(io.BytesIO(img)) if isinstance(img, bytes) else img
        img_tensor = transform(img)
        label_tensor = torch.tensor(encode_label(text), dtype=torch.long)

        pixel_values.append(img_tensor)
        labels.append(label_tensor)
        lengths.append(len(label_tensor))
    
    return {
        "pixel_values": pixel_values,  # list of Tensors
        "labels": labels,
        "label_lengths": lengths
    }

class CTCModelConfig(PretrainedConfig):
    def __init__(self, num_classes=NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

class CTCModel(PreTrainedModel):
    config_class = CTCModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bilstm = nn.LSTM(input_size=256 * 3, hidden_size=256,
                              num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, config.num_classes)

    def forward(self, pixel_values, labels=None, label_lengths=None):
        x = self.backbone(pixel_values)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 2, 1).reshape(b, w, -1)  # [B, T, F]
        x, _ = self.bilstm(x)
        logits = self.classifier(x)  # [B, T, C]

        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.full(size=(b,), fill_value=logits.size(1), dtype=torch.long)
            loss = F.ctc_loss(
                log_probs.transpose(0, 1), labels,
                input_lengths, label_lengths,
                blank=BLANK_TOKEN, reduction="mean", zero_infinity=True
            )
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

def data_collator(batch):
    images = [b["pixel_values"] for b in batch]
    if isinstance(images[0], list):  # still list? convert all manually
        images = [torch.tensor(img) for img in images]
    images = torch.stack(images)

    labels = [b["labels"] for b in batch]
    if isinstance(labels[0], list):
        labels = [torch.tensor(lbl, dtype=torch.long) for lbl in labels]

    label_lengths = torch.tensor([b["label_lengths"] for b in batch], dtype=torch.long)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PAD_TOKEN)

    return {
        "pixel_values": images,
        "labels": labels_padded,
        "label_lengths": label_lengths
    }

def decode_batch_predictions(logits):
    pred_ids = torch.argmax(torch.tensor(logits), dim=-1)  # [B, T]
    results = []

    for seq in pred_ids:
        decoded = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != BLANK_TOKEN and idx != prev:
                decoded.append(ID_TO_CHAR.get(idx, ''))
            prev = idx
        results.append(''.join(decoded))
    return results

def decode_labels(label_tensor):
    if isinstance(label_tensor, np.ndarray):
        label_tensor = torch.tensor(label_tensor)

    results = []
    for seq in label_tensor:
        decoded = []
        for idx in seq:
            # ensure idx is a scalar, even if it's a 1-element tensor or array
            try:
                if isinstance(idx, torch.Tensor):
                    val = idx.item() if idx.numel() == 1 else int(idx[0].item())
                elif isinstance(idx, np.ndarray):
                    val = int(idx.item()) if idx.size == 1 else int(idx[0])
                else:
                    val = int(idx)
            except Exception as e:
                print(f"[decode_labels] Skipping invalid idx={idx}, error={e}")
                continue

            if val > 0:  # skip BLANK (0) and PAD (-1)
                decoded.append(ID_TO_CHAR.get(val, ''))
        results.append(''.join(decoded))
    return results



def compute_metrics(eval_preds):
    logits, labels = eval_preds

    preds = decode_batch_predictions(logits)
    refs = decode_labels(labels)

    return {
        "cer": cer_metric.compute(predictions=preds, references=refs),
        "wer": wer_metric.compute(predictions=preds, references=refs),
        "acc": sum(p == r for p, r in zip(preds, refs)) / len(refs)
    }

def train():
    dataset = DatasetDict({
        "train": create_captcha_dataset(2000),
        "validation": create_captcha_dataset(200)
    }).map(process_for_hf, batched=True, batch_size=32, remove_columns=["image", "label"])

    dataset.set_format("torch")
    config = CTCModelConfig()
    model = CTCModel(config).to(device)

    training_args = TrainingArguments(
        output_dir="./ctc_captcha_pytorch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        learning_rate=1e-3,
        logging_steps=50,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=None,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained("./models/ctc_model_pytorch")

def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    tensor = transform(img)  # same torchvision transform as training
    return tensor.unsqueeze(0)  # [1, 3, H, W]

def decode_prediction(logits):
    pred_ids = torch.argmax(logits, dim=-1)  # [B, T]
    results = []
    for seq in pred_ids:
        decoded = []
        prev = None
        for idx in seq:
            idx = idx.item()
            if idx != BLANK_TOKEN and idx != prev:
                decoded.append(ID_TO_CHAR.get(idx, ""))
            prev = idx
        results.append("".join(decoded))
    return results[0]  # assuming batch size 1

from pathlib import Path
import matplotlib.pyplot as plt

def predict_single_image(model, img: Image.Image, save_output=True):
    model.eval()
    with torch.no_grad():
        input_tensor = preprocess_image(img).to(device)
        out = model(pixel_values=input_tensor)
        pred = decode_prediction(out["logits"].cpu())
    return pred

def generate_captcha():
    generator = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
    label = ''.join(random.choices(CHARS, k=MAX_LABEL_LEN))
    img = generator.generate_image(label).convert("RGB")
    return img, label

def predict():
    model = CTCModel.from_pretrained("./models/ctc_model_pytorch").to(device)
    img, label = generate_captcha()
    pred = predict_single_image(model, img)

    print(f"Predicted: {pred}, Actual: {label}")

    # Visualize and save output
    os.makedirs("output", exist_ok=True)
    output_path = f"output/captcha_{label}_{pred}.png"
    img.save(output_path)

    # Optional: Display inline
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"GT: {label} | Pred: {pred}")
    plt.show()


if __name__ == "__main__":
    #train()
    predict()