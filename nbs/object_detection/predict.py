import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
from PIL import Image

# --- Constants
IMG_WIDTH, IMG_HEIGHT = 200, 60
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ID_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}  # CTC labels start from 1
ID_TO_CHAR[0] = ''  # blank

# --- Generate one CAPTCHA image
def generate_captcha():
    generator = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)
    label = ''.join(np.random.choice(CHARS, 5))
    img = generator.generate_image(label).convert("RGB")
    return img, label

# --- Preprocess image
def preprocess_image(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img).astype("float32") / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return tf.convert_to_tensor(img)

# --- Decode CTC prediction
def decode_prediction(logits):
    pred_ids = tf.argmax(logits, axis=-1).numpy()
    results = []
    for seq in pred_ids:
        text = []
        prev = -1
        for idx in seq:
            if idx != prev and idx != 0:
                text.append(ID_TO_CHAR.get(idx, ''))
            prev = idx
        results.append(''.join(text))
    return results[0]

# --- Run inference
def predict_single_image(model, img):
    x = preprocess_image(img)
    x = tf.expand_dims(x, 0)  # batch dimension
    logits = model(x, training=False)
    pred = decode_prediction(logits)
    return pred

# --- Main usage
if __name__ == "__main__":
    # Load trained model (base_model only)
    model = tf.keras.models.load_model("models/ctc_model_full.keras", compile=False)

    # Generate and predict
    img, true_label = generate_captcha()
    pred = predict_single_image(model, img)

    # Display
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"GT: {true_label}  |  Pred: {pred}")
    plt.show()
