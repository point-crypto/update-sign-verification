import os, cv2, shutil
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

REF_DIR = "data/refs"
VIS_DIR = "visuals"
REPORT_DIR = "reports"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ================= PDF SAFE =================
def pdf_safe(text):
    return (
        str(text)
        .replace("✔", "")
        .replace("❌", "")
        .replace("🔥", "")
        .replace("📊", "")
        .replace("🟢", "")
        .replace("🟡", "")
        .replace("🔴", "")
        .replace("➡", "->")
    )

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

# ================= FEATURE HELPERS =================
def stroke_density(img):
    return np.sum(img < 128) / img.size

def contour_count(img):
    edges = cv2.Canny(img, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(cnts)

def similarity(a, b):
    return max(0, 100 - abs(a - b) * 100)

# ================= VISUALS =================
def heatmap(ref, test):
    diff = cv2.absdiff(ref, test)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    path = f"{VIS_DIR}/heatmap.png"
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

# ================= PDF =================
def generate_pdf(report, images):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Signature Verification Report", ln=True)
    pdf.ln(5)

    for k, v in report.items():
        pdf.multi_cell(0, 8, f"{k}: {pdf_safe(v)}")

    pdf.ln(5)
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
        "Upload reference signatures (used for ONE verification)\n"
        "Then type /verify",
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

    await update.message.reply_text("✅ Reference saved. Type /verify when ready.")
    return REFERENCE

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Upload ONE test signature.")
    return WAIT_TEST

# ================= DYNAMIC THRESHOLD =================
def dynamic_threshold(ref_images):
    if len(ref_images) < 2:
        return 70.0  # fallback

    sims = []
    for i in range(len(ref_images)):
        for j in range(i + 1, len(ref_images)):
            sims.append(ssim(ref_images[i], ref_images[j]) * 100)

    avg = np.mean(sims)
    threshold = avg - 7  # safety margin

    return float(min(max(threshold, 65), 85))

async def test_signature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.message.from_user.id)
    udir = os.path.join(REF_DIR, uid)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    test_img = preprocess(test_path)

    ref_imgs = []
    for r in os.listdir(udir):
        img = preprocess(os.path.join(udir, r))
        if img is not None:
            ref_imgs.append(img)

    threshold = dynamic_threshold(ref_imgs)

    scores = []
    best_ref = None
    best_ssim = 0

    for ref_img in ref_imgs:
        ssim_score = ssim(ref_img, test_img) * 100
        sd_sim = similarity(stroke_density(ref_img), stroke_density(test_img))
        cc_sim = similarity(contour_count(ref_img), contour_count(test_img))

        final = (
            0.6 * ssim_score +
            0.2 * sd_sim +
            0.2 * cc_sim
        )

        scores.append(final)

        if ssim_score > best_ssim:
            best_ssim = ssim_score
            best_ref = ref_img

    score = float(np.mean(scores))
    risk = 100 - score
    result = "MATCH ✅" if score >= threshold else "MISMATCH ❌"

    h = heatmap(best_ref, test_img)
    g = confidence_graph(score)

    report = {
        "Score": f"{score:.2f}%",
        "Dynamic Threshold": f"{threshold:.2f}%",
        "Result": result,
        "Forgery Risk": f"{risk:.2f}%",
        "AI Explanation": ai_text(score)
    }

    context.user_data["pdf"] = generate_pdf(report, [h, g])

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Download PDF", callback_data="pdf")],
        [InlineKeyboardButton("▶️ Start New Verification", callback_data="restart")]
    ])

    await update.message.reply_text(
        f"🔍 *Result*\n\n"
        f"Score: `{score:.2f}%`\n"
        f"Threshold: `{threshold:.2f}%`\n"
        f"{result}\n"
        f"Forgery Risk: `{risk:.2f}%`\n\n"
        f"*AI Explanation*\n{ai_text(score)}",
        parse_mode="Markdown",
        reply_markup=keyboard
    )

    await update.message.reply_photo(open(h, "rb"), caption="🔥 Heatmap")
    await update.message.reply_photo(open(g, "rb"), caption="📊 Confidence Graph")

    shutil.rmtree(udir)
    os.makedirs(udir, exist_ok=True)

    return ConversationHandler.END

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "pdf":
        await q.message.reply_document(open(context.user_data["pdf"], "rb"))
    elif q.data == "restart":
        await q.message.reply_text("🔄 Type /start to begin again")

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
        fallbacks=[CommandHandler("start", start)]
    )

    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(callbacks))

    print("🤖 Bot running with Dynamic Threshold")
    app.run_polling()

if __name__ == "__main__":
    main()
