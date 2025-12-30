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
VIS_DIR = "visuals"
REPORT_DIR = "reports"
HISTORY_FILE = "history.json"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ================= IMAGE UTILS =================
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
def accuracy_heatmap(ref, test):
    diff = cv2.absdiff(ref, test)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    path = f"{VIS_DIR}/accuracy_heatmap.png"
    cv2.imwrite(path, heat)
    return path

def stroke_heatmap(ref, test):
    edges1 = cv2.Canny(ref, 50, 150)
    edges2 = cv2.Canny(test, 50, 150)
    diff = cv2.absdiff(edges1, edges2)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
    path = f"{VIS_DIR}/stroke_heatmap.png"
    cv2.imwrite(path, heat)
    return path

def confidence_graph(score):
    path = f"{VIS_DIR}/confidence.png"
    plt.figure()
    plt.bar(["Confidence", "Forgery Risk"], [score, 100-score])
    plt.ylim(0, 100)
    plt.savefig(path)
    plt.close()
    return path

def accuracy_trend(user_id, score):
    data = {}
    if os.path.exists(HISTORY_FILE):
        data = json.load(open(HISTORY_FILE))
    data.setdefault(user_id, []).append(score)
    json.dump(data, open(HISTORY_FILE, "w"), indent=4)

    path = f"{VIS_DIR}/trend.png"
    plt.figure()
    plt.plot(data[user_id], marker="o")
    plt.ylim(0, 100)
    plt.title("Accuracy Trend")
    plt.ylabel("Score (%)")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    return path

# ================= AI =================
def ai_text(score):
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

def score_color(score):
    return "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"

# ================= PDF =================
def generate_pdf(report, images):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Signature Verification Report", ln=True)
    pdf.ln(5)

    for k, v in report.items():
        pdf.multi_cell(0, 8, f"{k}: {v}")
    pdf.ln(3)

    for img in images:
        pdf.image(img, w=170)
        pdf.ln(5)

    path = f"{REPORT_DIR}/report_{int(datetime.now().timestamp())}.pdf"
    pdf.output(path)
    return path

# ================= BOT =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "✍️ *Signature Verification Bot*\n\n"
        "1️⃣ Upload reference signatures\n"
        "2️⃣ Type /verify\n"
        "3️⃣ Upload ONE test signature",
        parse_mode="Markdown"
    )
    return REFERENCE

async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    udir = os.path.join(REF_DIR, uid)
    os.makedirs(udir, exist_ok=True)

    photo = await update.message.photo[-1].get_file()
    path = os.path.join(udir, f"ref_{len(os.listdir(udir))+1}.jpg")
    await photo.download_to_drive(path)
    await update.message.reply_text("✅ Reference saved")
    return REFERENCE

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Upload ONE test signature")
    return WAIT_TEST

async def test_signature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    udir = os.path.join(REF_DIR, uid)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    test_img = preprocess(test_path)

    scores, best_ref, best = [], None, 0
    for r in os.listdir(udir):
        ref_img = preprocess(os.path.join(udir, r))
        s = ssim(ref_img, test_img) * 100
        scores.append(s)
        if s > best:
            best = s
            best_ref = ref_img

    score = float(np.mean(scores))
    risk = 100 - score
    result = "MATCH ✅" if score >= 75 else "MISMATCH ❌"

    # Visuals
    h1 = accuracy_heatmap(best_ref, test_img)
    h2 = stroke_heatmap(best_ref, test_img)
    cg = confidence_graph(score)
    tg = accuracy_trend(uid, score)

    report = {
        "Score": f"{score:.2f}%",
        "Result": result,
        "Forgery Risk": f"{risk:.2f}%",
        "AI Explanation": ai_text(score)
    }

    context.user_data["pdf"] = generate_pdf(
        report, [h1, h2, cg]
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Start New Verification", callback_data="restart")],
        [InlineKeyboardButton("📄 Download PDF", callback_data="pdf")]
    ])

    await update.message.reply_text(
        f"🔍 *Result*\n\n"
        f"{score_color(score)} Score: `{score:.2f}%`\n"
        f"{result}\n"
        f"Forgery Risk: `{risk:.2f}%`\n\n"
        f"*AI Explanation*\n{ai_text(score)}",
        parse_mode="Markdown",
        reply_markup=keyboard
    )

    await update.message.reply_photo(open(h1, "rb"), caption="🔥 Accuracy Heatmap")
    await update.message.reply_photo(open(h2, "rb"), caption="🔥 Stroke Deviation Heatmap")
    await update.message.reply_photo(open(cg, "rb"), caption="📊 Confidence Graph")
    await update.message.reply_photo(open(tg, "rb"), caption="📈 Accuracy Trend")

    return ConversationHandler.END

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "restart":
        await q.message.reply_text("🔄 Restarting… Type /start")

    elif q.data == "pdf":
        await q.message.reply_document(open(context.user_data["pdf"], "rb"))

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REFERENCE: [MessageHandler(filters.PHOTO, save_reference),
                        CommandHandler("verify", verify)],
            WAIT_TEST: [MessageHandler(filters.PHOTO, test_signature)]
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(callbacks))

    print("🤖 Bot running with ALL features")
    app.run_polling()

if __name__ == "__main__":
    main()
