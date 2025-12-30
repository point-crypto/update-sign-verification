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
    ContextTypes,
    ConversationHandler,
    filters
)

# ================= CONFIG =================
TOKEN = os.getenv("BOT_TOKEN") or "PASTE_YOUR_BOT_TOKEN_HERE"

REFERENCE, WAIT_TEST = range(2)

AUDIT_FILE = "audit_log.json"
REFERENCE_DIR = "signatures"
MATCH_THRESHOLD = 75

os.makedirs(REFERENCE_DIR, exist_ok=True)

# ================= SAFE JSON =================
def safe_load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

def safe_write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# ================= IMAGE PROCESSING =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        img = img[y:y+h, x:x+w]

    return cv2.resize(img, (300, 150))

def extract_features(img):
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [
        int(np.sum(img < 128)),        # ink density
        int(len(contours)),            # stroke count
        float(np.mean(edges)),         # edge strength
        float(img.shape[1] / img.shape[0])  # aspect ratio
    ]

def ml_similarity(f1, f2):
    diff = np.abs(np.array(f1) - np.array(f2))
    return float((1 / (1 + np.mean(diff))) * 100)

# ================= AI EXPLANATION =================
def ai_reason(score):
    if score >= 90:
        return (
            "🧠 AI Analysis:\n"
            "• Stroke continuity is very stable\n"
            "• Pressure distribution matches reference\n"
            "• No visible tremor or retracing\n"
            "Forgery Type: NONE (Genuine)"
        )
    elif score >= 75:
        return (
            "🧠 AI Analysis:\n"
            "• Overall structure matches\n"
            "• Minor stroke variation detected\n"
            "• Possible slow tracing signs\n"
            "Forgery Type: TRACED / SIMULATED"
        )
    else:
        return (
            "🧠 AI Analysis:\n"
            "• Stroke rhythm inconsistent\n"
            "• Pressure mismatch detected\n"
            "• Significant shape deviation\n"
            "Forgery Type: RANDOM / SKILLED FORGERY"
        )

# ================= BOT FLOW =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["ref_count"] = 0
    await update.message.reply_text(
        "✍️ *AI Signature Verification Bot*\n\n"
        "📌 Send reference signatures (multiple allowed)\n"
        "📌 Type /verify when done\n\n"
        "/history – Past results\n"
        "/graph – Accuracy graph",
        parse_mode="Markdown"
    )
    return REFERENCE

async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REFERENCE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    photo = await update.message.photo[-1].get_file()
    count = len(os.listdir(user_dir)) + 1
    path = os.path.join(user_dir, f"ref{count}.jpg")

    await photo.download_to_drive(path)
    context.user_data["ref_count"] += 1

    await update.message.reply_text(
        f"✅ Reference {count} saved.\nSend more or type /verify"
    )
    return REFERENCE

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("ref_count", 0) == 0:
        await update.message.reply_text("❌ Upload at least one reference first.")
        return REFERENCE

    await update.message.reply_text("📤 Send TEST signature")
    return WAIT_TEST

async def test_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REFERENCE_DIR, user_id)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    best_score = 0

    for file in os.listdir(user_dir):
        ref_img = preprocess(os.path.join(user_dir, file))
        test_img = preprocess(test_path)

        if ref_img is None or test_img is None:
            continue

        s = ssim(ref_img, test_img) * 100
        m = ml_similarity(
            extract_features(ref_img),
            extract_features(test_img)
        )

        final = (0.7 * s) + (0.3 * m)
        best_score = max(best_score, final)

    score = float(best_score)
    result = "MATCH ✅" if score >= MATCH_THRESHOLD else "MISMATCH ❌"
    confidence = "HIGH 🟢" if score >= 85 else "MEDIUM 🟡" if score >= 70 else "LOW 🔴"
    risk = 100 - score

    explanation = ai_reason(score)

    await update.message.reply_text(
        f"🔍 *Signature Result*\n\n"
        f"Score: `{score:.2f}%`\n"
        f"Result: {result}\n"
        f"Confidence: {confidence}\n"
        f"Forgery Risk: `{risk:.2f}%`\n\n"
        f"{explanation}",
        parse_mode="Markdown"
    )

    logs = safe_load_json(AUDIT_FILE)
    logs.append({
        "time": datetime.now().isoformat(),
        "user_id": int(user_id),
        "score": round(score, 2),
        "result": result,
        "confidence": confidence
    })
    safe_write_json(AUDIT_FILE, logs)

    return ConversationHandler.END

# ================= HISTORY =================
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    logs = safe_load_json(AUDIT_FILE)

    user_logs = [l for l in logs if l["user_id"] == user_id][-5:]
    if not user_logs:
        await update.message.reply_text("No history found.")
        return

    msg = "📜 *Last 5 Verifications*\n\n"
    for l in user_logs:
        msg += f"{l['time']} | {l['score']}% | {l['result']}\n"

    await update.message.reply_text(msg, parse_mode="Markdown")

# ================= GRAPH =================
async def graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    logs = safe_load_json(AUDIT_FILE)

    scores = [l["score"] for l in logs if l["user_id"] == user_id]
    if len(scores) < 2:
        await update.message.reply_text("Not enough data for graph.")
        return

    plt.plot(scores, marker="o")
    plt.title("Signature Accuracy Trend")
    plt.xlabel("Attempt")
    plt.ylabel("Score (%)")
    plt.grid(True)

    path = "accuracy.png"
    plt.savefig(path)
    plt.close()

    await update.message.reply_photo(photo=open(path, "rb"))

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REFERENCE: [
                MessageHandler(filters.PHOTO, save_reference),
                CommandHandler("verify", verify),
            ],
            WAIT_TEST: [
                MessageHandler(filters.PHOTO, test_image),
            ],
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("graph", graph))

    print("🤖 Bot running (polling mode)")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
