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
def load_json(p): return json.load(open(p)) if os.path.exists(p) else []
def save_json(p, d): json.dump(d, open(p, "w"), indent=4)

# ================= IMAGE UTILS =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    if cv2.Laplacian(img, cv2.CV_64F).var() < 50:
        return None  # Quality check (blur)
    _, b = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    cnts,_ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x,y,w,h = cv2.boundingRect(np.vstack(cnts))
        img = img[y:y+h, x:x+w]
    return cv2.resize(img, (300,150))

def extract_features(img):
    edges = cv2.Canny(img,50,150)
    cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return np.array([
        np.sum(img<128),
        len(cnts),
        np.mean(edges),
        img.shape[1]/img.shape[0]
    ])

# ================= AI LOGIC =================
def classify_forgery(score):
    if score >= 85: return "Genuine"
    if score >= 70: return "Skilled Forgery"
    return "Random Forgery"

def ai_explanation(score):
    if score >= 85:
        return "Signature exhibits high structural similarity and stable stroke flow. Authenticity is strongly confirmed."
    elif score >= 70:
        return "Partial similarity detected with minor stroke deviations. Possible skilled forgery."
    else:
        return "Low similarity with inconsistent stroke patterns. High forgery probability."

# ================= VISUALS =================
def heatmap(ref, test):
    diff = cv2.absdiff(ref,test)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)

def bar_graph(score, path):
    plt.figure()
    plt.bar(["Confidence","Risk"],[score,100-score])
    plt.ylim(0,100)
    plt.savefig(path)
    plt.close()

# ================= PDF REPORT =================
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=12)
    for k,v in data.items():
        pdf.multi_cell(0,8,f"{k}: {v}")
    path = f"{REPORT_DIR}/report_{datetime.now().timestamp()}.pdf"
    pdf.output(path)
    return path

# ================= HASH CHECK =================
def image_hash(path):
    return hashlib.md5(open(path,"rb").read()).hexdigest()

# ================= BOT =================
async def start(update:Update, context:ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "✍️ *SignaGuard AI*\n\nSend reference signatures.\nThen /verify",
        parse_mode="Markdown"
    )
    return REFERENCE

async def save_reference(update:Update, context:ContextTypes.DEFAULT_TYPE):
    uid=str(update.message.from_user.id)
    udir=os.path.join(REF_DIR,uid)
    os.makedirs(udir,exist_ok=True)

    f=await update.message.photo[-1].get_file()
    path=os.path.join(udir,f"ref_{len(os.listdir(udir))+1}.jpg")
    await f.download_to_drive(path)
    await update.message.reply_text("✅ Reference saved")
    return REFERENCE

async def verify(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Send test signature")
    return WAIT_TEST

async def test(update:Update, context:ContextTypes.DEFAULT_TYPE):
    uid=str(update.message.from_user.id)
    udir=os.path.join(REF_DIR,uid)

    f=await update.message.photo[-1].get_file()
    test_path="test.jpg"
    await f.download_to_drive(test_path)

    # One-time verification
    h=image_hash(test_path)
    used=load_json(HASH_FILE)
    if h in used:
        await update.message.reply_text("⚠️ This image was already verified.")
        return ConversationHandler.END
    used.append(h); save_json(HASH_FILE,used)

    test_img=preprocess(test_path)
    if test_img is None:
        await update.message.reply_text("❌ Image quality too low.")
        return ConversationHandler.END

    scores=[]
    features=[]
    for r in os.listdir(udir):
        ref_img=preprocess(os.path.join(udir,r))
        if ref_img is None: continue
        s=ssim(ref_img,test_img)*100
        scores.append(s)
        features.append(extract_features(ref_img))

    mean_score=float(np.mean(scores))
    threshold=max(70, np.mean(scores)-5)
    forgery=classify_forgery(mean_score)

    # ML Classifier
    X=np.array(features)
    y=np.ones(len(X))
    clf=SVC()
    clf.fit(X,y)
    ml_decision="Genuine" if clf.predict([extract_features(test_img)])[0]==1 else "Forgery"

    msg=(
        f"*Score:* `{mean_score:.2f}%`\n"
        f"*Threshold:* `{threshold:.2f}%`\n"
        f"*Type:* `{forgery}`\n"
        f"*ML Decision:* `{ml_decision}`\n\n"
        f"*AI Explanation*\n{ai_explanation(mean_score)}"
    )

    buttons=InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Download PDF",callback_data="pdf")]
    ])

    context.user_data["report"]={
        "Score":mean_score,
        "Forgery Type":forgery,
        "ML Decision":ml_decision,
        "Explanation":ai_explanation(mean_score)
    }

    await update.message.reply_text(msg,parse_mode="Markdown",reply_markup=buttons)
    return ConversationHandler.END

async def pdf_cb(update:Update, context:ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    await q.answer()
    path=generate_pdf(context.user_data["report"])
    await q.message.reply_document(open(path,"rb"))

# ================= MAIN =================
def main():
    app=Application.builder().token(TOKEN).build()

    conv=ConversationHandler(
        entry_points=[CommandHandler("start",start)],
        states={
            REFERENCE:[MessageHandler(filters.PHOTO,save_reference),CommandHandler("verify",verify)],
            WAIT_TEST:[MessageHandler(filters.PHOTO,test)]
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(pdf_cb,pattern="pdf"))
    print("🤖 SignaGuard AI Running")
    app.run_polling()

if __name__=="__main__":
    main()


