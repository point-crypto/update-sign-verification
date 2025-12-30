import os, cv2, json, hashlib
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from sklearn.svm import SVC
from fpdf import FPDF

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters
)

# ================= CONFIG =================
TOKEN = "PASTE_YOUR_BOT_TOKEN_HERE"

REFERENCE, WAIT_TEST = range(2)

BASE_DIR = "data"
REF_DIR = os.path.join(BASE_DIR, "refs")
AUDIT_FILE = os.path.join(BASE_DIR, "audit.json")
HASH_FILE = os.path.join(BASE_DIR, "used_hashes.json")
REPORT_DIR = "reports"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ================= JSON UTILS =================
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# ================= IMAGE UTILS =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, b = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

# ================= AI LOGIC =================
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
            "The test signature shows strong structural similarity with reference samples. "
            "Stroke alignment and spatial consistency are stable. "
            "The signature is classified as genuine."
        )
    elif score >= 70:
        return (
            "Moderate similarity detected with minor stroke deviations. "
            "These deviations may indicate a skilled imitation. "
            "Further verification is recommended."
        )
    else:
        return (
            "Low similarity observed with inconsistent stroke patterns. "
            "Significant deviations suggest a high probability of forgery."
        )

# ================= PDF REPORT =================
def generate_pdf(report_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Signature Verification Report", ln=True)
    pdf.ln(4)

    for k, v in report_data.items():
        pdf.multi_cell(0, 8, f"{k}: {v}")
        pdf.ln(1)

    path = f"{REPORT_DIR}/report_{int(datetime.now().timestamp())}.pdf"
    pdf.output(path)
    return path

# ================= HASH =================
def image_hash(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()

# ================= BOT FLOW =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "✍️ *SignaGuard AI – Signature Verification Bot*\n\n"
        "📌 *How to use:*\n"
        "1️⃣ Send **2–5 reference signatures** (same person)\n"
        "2️⃣ After uploading references, type /verify\n"
        "3️⃣ Send the **test signature** for verification\n\n"
        "ℹ️ The system will analyze similarity, forgery risk, "
        "and generate an AI explanation.\n\n"
        "➡️ Start by sending reference signatures now.",
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

    await update.message.reply_text(
        f"✅ Reference Signature {idx} saved.\n"
        f"📤 Upload more or type /verify to continue."
    )
    return REFERENCE

async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📤 *Verification Step*\n\n"
        "Now send **ONE test signature image**.\n"
        "⚠️ This image can be verified only once.",
        parse_mode="Markdown"
    )
    return WAIT_TEST

async def test_signature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REF_DIR, user_id)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    # One-time verification
    img_hash = image_hash(test_path)
    used = load_json(HASH_FILE)
    if img_hash in used:
        await update.message.reply_text("⚠️ This test signature was already verified.")
        return ConversationHandler.END

    used.append(img_hash)
    save_json(HASH_FILE, used)

    test_img = preprocess(test_path)

    scores = []
    features = []

    for ref in os.listdir(user_dir):
        ref_img = preprocess(os.path.join(user_dir, ref))
        s = ssim(ref_img, test_img) * 100
        scores.append(s)
        features.append(extract_features(ref_img))

    final_score = float(np.mean(scores))
    forgery_type = classify_forgery(final_score)

    # ML Classifier
    X = np.array(features)
    y = np.ones(len(X))
    clf = SVC()
    clf.fit(X, y)
    ml_result = "Genuine" if clf.predict([extract_features(test_img)])[0] == 1 else "Forgery"

    explanation = ai_explanation(final_score)

    report = {
        "Verification Score (%)": round(final_score, 2),
        "Forgery Classification": forgery_type,
        "ML Decision": ml_result,
        "AI Explanation": explanation,
        "Verified At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    context.user_data["report"] = report

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📄 Download PDF Report", callback_data="pdf"),
            InlineKeyboardButton("🔁 Verify Again", callback_data="restart")
        ]
    ])

    await update.message.reply_text(
        f"🔍 *Verification Result*\n\n"
        f"Score: `{final_score:.2f}%`\n"
        f"Forgery Type: `{forgery_type}`\n"
        f"ML Decision: `{ml_result}`\n\n"
        f"*AI Explanation*\n{explanation}",
        parse_mode="Markdown",
        reply_markup=keyboard
    )

    return ConversationHandler.END

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "pdf":
        path = generate_pdf(context.user_data["report"])
        await query.message.reply_document(open(path, "rb"))

    elif query.data == "restart":
        context.user_data.clear()
        await query.message.reply_text("🔄 Restarted. Send new reference signatures.")
        return REFERENCE

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
            WAIT_TEST: [
                MessageHandler(filters.PHOTO, test_signature)
            ]
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(callbacks))

    print("🤖 SignaGuard AI is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
