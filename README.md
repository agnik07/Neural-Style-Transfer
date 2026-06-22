# NeuralArt (AdaIN Neural Style Transfer)

This project provides an **AdaIN-based neural style transfer** implementation using **PyTorch** and a simple **Flask** web app to run style transfer on two user-provided images (content + style).

## Features
- Train an AdaIN style-transfer decoder using a fixed VGG encoder.
- Web UI to upload **content** and **style** images and generate a stylized output.
- Adjustable **alpha** (style strength vs content preservation).

## Project Structure
- `app.py` – Flask web app (uploads + inference)
- `train.py` – training script for the decoder
- `utils/models.py` – VGG encoder + decoder network definitions
- `utils/utils.py` – dataset loader, transforms, AdaIN + mean/std utilities
- `templates/index.html` – UI
- `static/uploads/` – uploaded images + generated outputs
- `content_data/`, `style_data/` – training image folders
- `experiment/`, `experiments/` – training outputs/checkpoints

## Requirements
See `requirements.txt`:
- `torch`, `torchvision`
- `flask`, `flask_bootstrap`, `flask_wtf`
- `Pillow`, `numpy`, `tqdm`, etc.

## Setup
### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Required model files
The web app loads:
- `vgg_normalised.pth` (VGG encoder weights)
- `experiment/final_exp/decoder_final.pth` (decoder weights)

Ensure these exist at the paths expected by `app.py`.

> Note: `app.py` currently includes an absolute path for the decoder checkpoint. If you move the project directory, update that path accordingly.

## Running the Web App
```bash
python app.py
```
Then open:
- `http://127.0.0.1:8080`

### How inference works
1. User uploads a **content** image and a **style** image.
2. The VGG encoder extracts features (test mode returns only `h4`).
3. AdaIN mixes content and style statistics.
4. The trained decoder reconstructs the stylized image.
5. The result is saved into `static/uploads/` and shown in the UI.

## Training the Decoder
### Data layout
Provide two folders:
- `content_data/` – content images
- `style_data/` – style images

The training dataset expects each folder contains image files with extensions `.jpg`, `.png`, `.jpeg`.

### Train
```bash
python train.py \
  --content_dir content_data \
  --style_dir style_data \
  --vgg vgg_normalised.pth \
  --experiment experiment1 \
  --epochs 1
```

Common flags:
- `--content_size`, `--style_size`, `--final_size`
- `--batch_size`, `--lr`, `--lr_decay`
- `--content_weight`, `--style_weight`
- `--save_interval` (saves decoder/optimizer checkpoints)

Checkpoints are written under `experiment/<experiment_name>/`.

## Alpha (style strength)
In `app.py`, `alpha` controls the interpolation:
- `alpha = 1` → more style transfer (more stylized)
- `alpha = 0` → original content features

## Notes / Known Details
- The encoder (VGG) parameters are frozen (`requires_grad=False`).
- The training updates only the decoder.
- Device selection:
  - uses `mps` if available, else `cuda`, else `cpu`.

## License
Add your license information here (or remove this section).
