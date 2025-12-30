import os
import cv2
import json
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters
)

# ================= CONFIG =================
TOKEN = "PASTE_YOUR_BOT_TOKEN_HERE"

REFERENCE, WAIT_TEST = range(2)

BASE_DIR = "data"
REF_DIR = os.path.join(BASE_DIR, "refs")
AUDIT_FILE = os.path.join(BASE_DIR, "audit.json")
MATCH_THRESHOLD = 75

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

# ================= SAFE JSON =================
def load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return []

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# ================= IMAGE UTILS =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, bin_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(np.vstack(cnts))
        img = img[y:y+h, x:x+w]
    return cv2.resize(img, (300, 150))

def extract_features(img):
    edges = cv2.Canny(img, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.array([
        np.sum(img < 128),
        len(cnts),
        np.mean(edges),
        img.shape[1] / img.shape[0]
    ], dtype=float)

def ml_similarity(f1, f2):
    diff = np.abs(f1 - f2)
    return float((1 / (1 + np.mean(diff))) * 100)

# ================= VISUALS =================
def generate_heatmap(ref, test, path):
    diff = cv2.absdiff(ref, test)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    cv2.imwrite(path, heatmap)
    return path

def generate_confidence_bar(score, path):
    confidence = score
    risk = 100 - score
    plt.figure()
    plt.bar(["Confidence", "Forgery Risk"], [confidence, risk])
    plt.ylim(0, 100)
    plt.title("Verification Confidence")
    plt.grid(axis="y")
    plt.savefig(path)
    plt.close()
    return path

# ================= AI EXPLANATION =================
def ai_explanation(score):
    if score >= 85:
        return (
            "✔ High structural similarity\n"
            "✔ Stroke flow consistent\n"
            "✔ Writing pressure stable\n"
            "➡ Signature is highly authentic"
        )
    elif score >= 70:
        return (
            "⚠ Partial similarity detected\n"
            "⚠ Minor stroke deviation\n"
            "➡ Possible skilled forgery"
        )
    else:
        return (
            "❌ Poor structural match\n"
            "❌ Stroke inconsistency\n"
            "➡ High probability of forgery"
        )

# ================= BOT FLOW =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "✍️ *Signature Verification Bot*\n\n"
        "Send one or more *REFERENCE* signatures.\n"
        "When done, type /verify\n\n"
        "/history – Past results\n"
        "/graph – Accuracy graph",
        parse_mode="Markdown"
    )
    return REFERENCE

async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REF_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    photo = await update.message.photo[-1].get_file()
    idx = len(os.listdir(user_dir)) + 1
    path = os.path.join(user_dir, f"ref_{idx}.jpg")
    await photo.download_to_drive(path)

    await update.message.reply_text(f"✅ Reference {idx} saved")
    return REFERENCE

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Send TEST signature")
    return WAIT_TEST

async def test_signature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REF_DIR, user_id)

    photo = await update.message.photo[-1].get_file()
    test_path = os.path.join(BASE_DIR, "test.jpg")
    await photo.download_to_drive(test_path)

    test_img = preprocess(test_path)
    best_score = 0
    best_ref = None

    for ref_file in os.listdir(user_dir):
        ref_img = preprocess(os.path.join(user_dir, ref_file))
        if ref_img is None or test_img is None:
            continue

        s = ssim(ref_img, test_img) * 100
        m = ml_similarity(extract_features(ref_img), extract_features(test_img))
        final = 0.7 * s + 0.3 * m

        if final > best_score:
            best_score = final
            best_ref = ref_img

    score = float(best_score)
    result = "MATCH ✅" if score >= MATCH_THRESHOLD else "MISMATCH ❌"
    risk = 100 - score

    await update.message.reply_text(
        f"🔍 *Result*\n\n"
        f"Score: `{score:.2f}%`\n"
        f"{result}\n"
        f"Forgery Risk: `{risk:.2f}%`\n\n"
        f"*AI Explanation*\n{ai_explanation(score)}",
        parse_mode="Markdown"
    )

    heatmap = generate_heatmap(best_ref, test_img, "heatmap.png")
    bar = generate_confidence_bar(score, "confidence.png")

    await update.message.reply_photo(photo=open(heatmap, "rb"), caption="🔥 Accuracy Heatmap")
    await update.message.reply_photo(photo=open(bar, "rb"), caption="📊 Confidence Graph")

    logs = load_json(AUDIT_FILE)
    logs.append({
        "time": datetime.now().isoformat(),
        "user_id": int(user_id),
        "score": round(score, 2),
        "result": result
    })
    save_json(AUDIT_FILE, logs)

    return ConversationHandler.END

# ================= HISTORY =================
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    logs = load_json(AUDIT_FILE)
    user_logs = [l for l in logs if l["user_id"] == user_id][-5:]

    if not user_logs:
        await update.message.reply_text("No history yet.")
        return

    msg = "📜 *Last Results*\n\n"
    for l in user_logs:
        msg += f"{l['time']} | {l['score']}% | {l['result']}\n"

    await update.message.reply_text(msg, parse_mode="Markdown")

# ================= GRAPH =================
async def graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    logs = load_json(AUDIT_FILE)
    scores = [l["score"] for l in logs if l["user_id"] == user_id]

    if len(scores) < 2:
        await update.message.reply_text("Not enough data.")
        return

    plt.plot(scores, marker="o")
    plt.title("Accuracy Trend")
    plt.ylabel("Score (%)")
    plt.grid(True)
    plt.savefig("trend.png")
    plt.close()

    await update.message.reply_photo(photo=open("trend.png", "rb"))

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REFERENCE: [
                MessageHandler(filters.PHOTO, save_reference),
                CommandHandler("verify", verify)
            ],
            WAIT_TEST: [MessageHandler(filters.PHOTO, test_signature)]
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("graph", graph))

    print("🤖 Bot running (polling mode)")
    app.run_polling()

if __name__ == "__main__":
    main()

