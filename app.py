import os
import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# =========================
# Flask Configuration
# =========================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "supersecretkey")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VGG_PATH = os.path.join(BASE_DIR, "vgg_normalised.pth")
DECODER_PATH = os.path.join(
    BASE_DIR,
    "experiment",
    "final_exp",
    "decoder_final.pth"
)


# =========================
# Form
# =========================

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField("Alpha", default=1.0)

    submit = SubmitField("Transfer Style")


# =========================
# Device
# =========================

device = torch.device("cpu")


# =========================
# Load Models
# =========================

encoder = VGGEncoder(VGG_PATH).to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device
    )
)

encoder.eval()
decoder.eval()

for param in encoder.parameters():
    param.requires_grad = False

for param in decoder.parameters():
    param.requires_grad = False


# =========================
# Image Transforms
# =========================

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# =========================
# Helper Functions
# =========================

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def style_transfer(content_image, style_image, alpha):
    content_tensor = image_transform(content_image).unsqueeze(0).to(device)
    style_tensor = image_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        stylized_feats = (
            alpha * stylized_feats
            + (1 - alpha) * content_feats
        )

        output = decoder(stylized_feats)

    return output


def save_image(image_tensor, path):
    image = image_tensor.cpu().squeeze(0).clamp(0, 1)

    image = transforms.ToPILImage()(image)
    image.save(path)


# =========================
# Routes
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        if not form.content.data or not form.style.data:
            error = "Please upload both content and style images."

        elif (
            not allowed_file(form.content.data.filename)
            or not allowed_file(form.style.data.filename)
        ):
            error = "Only PNG, JPG, and JPEG files are allowed."

        else:
            try:
                content_filename = secure_filename(
                    form.content.data.filename
                )

                style_filename = secure_filename(
                    form.style.data.filename
                )

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

                form.content.data.save(content_path)
                form.style.data.save(style_path)

                content_image = Image.open(content_path).convert("RGB")
                style_image = Image.open(style_path).convert("RGB")

                alpha = form.alpha.data or 1.0
                alpha = max(0.0, min(1.0, float(alpha)))

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )

                result_filename = f"stylized_{content_filename}"

                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )

                save_image(stylized_image, result_path)

                result_image = result_filename

            except Exception as e:
                error = f"Style transfer failed: {str(e)}"

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("examples", filename)


# =========================
# Local Development
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)