import os, cv2, json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
REPORT_DIR = "reports"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

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

# ================= VISUALS =================
def generate_heatmap(ref, test, path="heatmap.png"):
    diff = cv2.absdiff(ref, test)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    cv2.imwrite(path, heat)
    return path

def generate_confidence_graph(score, path="confidence.png"):
    plt.figure()
    plt.bar(["Confidence", "Forgery Risk"], [score, 100-score])
    plt.ylim(0, 100)
    plt.ylabel("Percentage")
    plt.title("Verification Confidence")
    plt.grid(axis="y")
    plt.savefig(path)
    plt.close()
    return path

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
        return "High similarity and stable stroke flow detected."
    elif score >= 70:
        return "Moderate similarity with minor stroke deviations."
    else:
        return "Low similarity and inconsistent stroke structure."

# ================= PDF =================
def generate_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Signature Verification Report", ln=True)
    pdf.ln(5)
    for k, v in report.items():
        pdf.multi_cell(0, 8, f"{k}: {v}")
    path = f"{REPORT_DIR}/report_{int(datetime.now().timestamp())}.pdf"
    pdf.output(path)
    return path

# ================= BOT =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Add Reference Signatures", callback_data="add_ref")],
        [InlineKeyboardButton("▶️ Start Verification", callback_data="start_verify")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")]
    ])
    await update.message.reply_text(
        "✍️ *SignaGuard AI – Signature Verification*\n\n"
        "1️⃣ Upload 2–5 reference signatures\n"
        "2️⃣ Click *Start Verification*\n"
        "3️⃣ Upload ONE test signature\n\n"
        "👇 Choose an option:",
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
        f"✅ Reference {idx} saved.\nUpload more or press *Start Verification*.",
        parse_mode="Markdown"
    )
    return REFERENCE

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    udir = os.path.join(REF_DIR, uid)

    if not os.path.exists(udir) or len(os.listdir(udir)) < 2:
        await update.message.reply_text("⚠️ Upload at least 2 reference signatures.")
        return REFERENCE

    context.user_data["verified_once"] = False
    await update.message.reply_text("📤 Upload ONE test signature.")
    return WAIT_TEST

async def test_signature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("verified_once"):
        await update.message.reply_text("⚠️ Test already completed. Type /start to retry.")
        return ConversationHandler.END

    uid = str(update.message.from_user.id)
    udir = os.path.join(REF_DIR, uid)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    test_img = preprocess(test_path)
    if test_img is None:
        await update.message.reply_text("❌ Unable to process test image.")
        return ConversationHandler.END

    scores = []
    best_ref = None
    best_score = 0

    for r in os.listdir(udir):
        ref_img = preprocess(os.path.join(udir, r))
        if ref_img is None:
            continue
        s = ssim(ref_img, test_img) * 100
        scores.append(s)
        if s > best_score:
            best_score = s
            best_ref = ref_img

    final_score = float(np.mean(scores))
    forgery = classify_forgery(final_score)
    explanation = ai_explanation(final_score)

    heatmap_path = generate_heatmap(best_ref, test_img)
    graph_path = generate_confidence_graph(final_score)

    report = {
        "Score (%)": round(final_score, 2),
        "Forgery Type": forgery,
        "AI Explanation": explanation,
        "Verified At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    context.user_data["report"] = report
    context.user_data["verified_once"] = True

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Download PDF", callback_data="pdf")],
        [InlineKeyboardButton("🔁 New Verification", callback_data="restart")]
    ])

    await update.message.reply_text(
        f"🔍 *Result*\n\n"
        f"Score: `{final_score:.2f}%`\n"
        f"Forgery Type: `{forgery}`\n\n"
        f"*AI Explanation*\n{explanation}",
        parse_mode="Markdown",
        reply_markup=keyboard
    )

    await update.message.reply_photo(photo=open(heatmap_path, "rb"), caption="🔥 Stroke Deviation Heatmap")
    await update.message.reply_photo(photo=open(graph_path, "rb"), caption="📊 Confidence Graph")

    return ConversationHandler.END

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "start_verify":
        await q.message.reply_text("Type /verify to continue.")

    elif q.data == "pdf":
        path = generate_pdf(context.user_data["report"])
        await q.message.reply_document(open(path, "rb"))

    elif q.data == "restart":
        await q.message.reply_text("🔄 Restarting… Type /start")

    elif q.data == "help":
        await q.message.reply_text(
            "ℹ️ *Help*\n\n"
            "• Use clear signature images\n"
            "• Same signer for all references\n"
            "• One test per verification",
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

    print("🤖 SignaGuard AI running with heatmap & graph")
    app.run_polling()

if __name__ == "__main__":
    main()

