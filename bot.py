import os, cv2, json, hashlib
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")

from skimage.metrics import structural_similarity as ssim
from fpdf import FPDF

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, ContextTypes,
    CallbackQueryHandler, filters
)

# ================= CONFIG =================
TOKEN = "PASTE_YOUR_BOT_TOKEN_HERE"

REFERENCE, WAIT_TEST = range(2)

BASE_DIR = "data"
REF_DIR = os.path.join(BASE_DIR, "refs")
HASH_FILE = os.path.join(BASE_DIR, "used_hashes.json")
REPORT_DIR = "reports"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ================= JSON =================
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# ================= IMAGE =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, b = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
    img = img[y:y+h, x:x+w]
    return cv2.resize(img, (300, 150))

# ================= AI =================
def classify_forgery(score):
    if score >= 85:
        return "Genuine Signature"
    elif score >= 70:
        return "Skilled Forgery"
    else:
        return "Random Forgery"

def ai_explanation(score):
    if score >= 85:
        return (
            "High similarity detected across structure and stroke flow. "
            "Minor variations fall within natural handwriting limits."
        )
    elif score >= 70:
        return (
            "Moderate similarity observed. Certain stroke deviations indicate "
            "possible skilled imitation."
        )
    else:
        return (
            "Low similarity with inconsistent stroke alignment. "
            "Strong indicators of random forgery detected."
        )

# ================= PDF =================
def generate_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Signature Verification Report", ln=True)
    pdf.ln(4)

    for k, v in report.items():
        pdf.multi_cell(0, 8, f"{k}: {v}")

    path = f"{REPORT_DIR}/report_{int(datetime.now().timestamp())}.pdf"
    pdf.output(path)
    return path

# ================= HASH =================
def image_hash(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()

# ================= BOT =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Add Reference Signatures", callback_data="add_ref")],
        [InlineKeyboardButton("🔍 Start Verification", callback_data="start_verify")],
        [InlineKeyboardButton("ℹ️ Help / Instructions", callback_data="help")]
    ])

    await update.message.reply_text(
        "✍️ *SignaGuard AI – Signature Verification System*\n\n"
        "📌 *Step-by-Step Guide*\n"
        "1️⃣ Upload **2–5 reference signatures** (same person)\n"
        "2️⃣ Click **Start Verification**\n"
        "3️⃣ Upload **ONE test signature**\n"
        "4️⃣ Get score, forgery type & AI explanation\n\n"
        "🔐 Each test image can be verified only once.\n\n"
        "👇 Choose an option below to begin:",
        parse_mode="Markdown",
        reply_markup=keyboard
    )
    return REFERENCE

async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    udir = os.path.join(REF_DIR, uid)
    os.makedirs(udir, exist_ok=True)

    photo = await update.message.photo[-1].get_file()
    idx = len(os.listdir(udir)) + 1
    path = os.path.join(udir, f"ref_{idx}.jpg")
    await photo.download_to_drive(path)

    await update.message.reply_text(
        f"✅ Reference {idx} saved.\n"
        f"📤 Upload more or press *Start Verification*.",
        parse_mode="Markdown"
    )
    return REFERENCE

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    udir = os.path.join(REF_DIR, uid)

    if not os.path.exists(udir) or len(os.listdir(udir)) < 2:
        await update.message.reply_text("⚠️ Please upload at least 2 reference signatures.")
        return REFERENCE

    await update.message.reply_text(
        "📤 *Verification Mode*\n\n"
        "Now upload **ONE test signature image**.\n"
        "This image cannot be reused.",
        parse_mode="Markdown"
    )
    return WAIT_TEST

async def test_signature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    udir = os.path.join(REF_DIR, uid)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    used = load_json(HASH_FILE)
    h = image_hash(test_path)
    if h in used:
        await update.message.reply_text("⚠️ This image was already verified.")
        return ConversationHandler.END

    used.append(h)
    save_json(HASH_FILE, used)

    test_img = preprocess(test_path)
    if test_img is None:
        await update.message.reply_text("❌ Unable to process test image.")
        return ConversationHandler.END

    scores = []
    for r in os.listdir(udir):
        ref_img = preprocess(os.path.join(udir, r))
        if ref_img is None:
            continue
        scores.append(ssim(ref_img, test_img) * 100)

    final_score = float(np.mean(scores))
    forgery = classify_forgery(final_score)
    explanation = ai_explanation(final_score)

    report = {
        "Score (%)": round(final_score, 2),
        "Forgery Type": forgery,
        "AI Explanation": explanation,
        "Verified At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    context.user_data["report"] = report

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Download PDF Report", callback_data="pdf")],
        [InlineKeyboardButton("🔄 Start New Verification", callback_data="restart")]
    ])

    await update.message.reply_text(
        f"🔍 *Verification Result*\n\n"
        f"Score: `{final_score:.2f}%`\n"
        f"Forgery Type: `{forgery}`\n\n"
        f"*AI Explanation*\n{explanation}",
        parse_mode="Markdown",
        reply_markup=keyboard
    )

    return ConversationHandler.END

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "pdf":
        path = generate_pdf(context.user_data["report"])
        await q.message.reply_document(open(path, "rb"))

    elif q.data == "restart":
        await q.message.reply_text("🔁 Restarting… Type /start to begin again.")

    elif q.data == "help":
        await q.message.reply_text(
            "ℹ️ *Help*\n\n"
            "• Upload clear signature images\n"
            "• Same signer for all references\n"
            "• Avoid cropped or blurry images\n"
            "• Each test image is verified once",
            parse_mode="Markdown"
        )

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
    app.add_handler(CallbackQueryHandler(callbacks))

    print("🤖 SignaGuard AI running")
    app.run_polling()

if __name__ == "__main__":
    main()
