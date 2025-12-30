import os
import cv2
import json
import hashlib
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

# ================= UTILITIES =================
def image_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def safe_load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
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

    return {
        "ink": int(np.sum(img < 128)),
        "strokes": int(len(contours)),
        "edge": float(np.mean(edges)),
        "ratio": float(img.shape[1] / img.shape[0])
    }

def ml_similarity(f1, f2):
    diff = np.mean([
        abs(f1["ink"] - f2["ink"]),
        abs(f1["strokes"] - f2["strokes"]),
        abs(f1["edge"] - f2["edge"]),
        abs(f1["ratio"] - f2["ratio"])
    ])
    return float((1 / (1 + diff)) * 100)

# ================= AI EXPLANATION =================
def ai_explain(score, ref_f, test_f):
    explanation = []

    if abs(ref_f["strokes"] - test_f["strokes"]) > 5:
        explanation.append("• Stroke count mismatch detected")

    if abs(ref_f["edge"] - test_f["edge"]) > 10:
        explanation.append("• Writing pressure variation found")

    if score >= 90:
        verdict = "Forgery Type: NONE (Genuine)"
    elif score >= 75:
        verdict = "Forgery Type: TRACED / SIMULATED"
    else:
        verdict = "Forgery Type: RANDOM / SKILLED FORGERY"

    if not explanation:
        explanation.append("• Stroke flow and pressure are consistent")

    return "🧠 AI Explanation:\n" + "\n".join(explanation) + f"\n{verdict}"

# ================= BOT FLOW =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()

    await update.message.reply_text(
        "✍️ *AI Signature Verification Bot*\n\n"
        "📌 Send reference signatures (duplicates will be ignored)\n"
        "📌 Type /verify when done\n\n"
        "/history – Previous results\n"
        "/graph – Accuracy graph",
        parse_mode="Markdown"
    )

    context.user_data["hashes"] = set()
    return REFERENCE

# ================= SAVE REFERENCE =================
async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REFERENCE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    photo = await update.message.photo[-1].get_file()
    temp_path = "temp_ref.jpg"
    await photo.download_to_drive(temp_path)

    h = image_hash(temp_path)

    if h in context.user_data["hashes"]:
        await update.message.reply_text("⚠️ Sample already saved (duplicate ignored)")
        return REFERENCE

    context.user_data["hashes"].add(h)

    count = len(os.listdir(user_dir)) + 1
    final_path = os.path.join(user_dir, f"ref{count}.jpg")
    os.replace(temp_path, final_path)

    await update.message.reply_text(f"✅ Reference sample {count} saved")
    return REFERENCE

# ================= VERIFY =================
async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Send TEST signature")
    return WAIT_TEST

# ================= TEST IMAGE =================
async def test_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REFERENCE_DIR, user_id)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    test_img = preprocess(test_path)
    test_f = extract_features(test_img)

    best_score = 0
    best_ref_f = None

    for f in os.listdir(user_dir):
        ref_img = preprocess(os.path.join(user_dir, f))
        ref_f = extract_features(ref_img)

        s = ssim(ref_img, test_img) * 100
        m = ml_similarity(ref_f, test_f)
        final = 0.7 * s + 0.3 * m

        if final > best_score:
            best_score = final
            best_ref_f = ref_f

    result = "MATCH ✅" if best_score >= MATCH_THRESHOLD else "MISMATCH ❌"
    explanation = ai_explain(best_score, best_ref_f, test_f)

    await update.message.reply_text(
        f"🔍 *Result*\n\n"
        f"Score: `{best_score:.2f}%`\n"
        f"{result}\n\n"
        f"{explanation}",
        parse_mode="Markdown"
    )

    logs = safe_load_json(AUDIT_FILE)
    logs.append({
        "time": datetime.now().isoformat(),
        "user_id": int(user_id),
        "score": round(best_score, 2),
        "result": result
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

    msg = "📜 *Last 5 Results*\n\n"
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
            WAIT_TEST: [MessageHandler(filters.PHOTO, test_image)],
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("graph", graph))

    print("🤖 Bot running (stable polling mode)")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
