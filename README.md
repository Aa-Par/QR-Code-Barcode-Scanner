# QR Code & Barcode Scanner (Single-Image, CLI-Only)

A command-line computer vision tool that scans a **single image file** for
QR codes and 1D barcodes (EAN-13, EAN-8, UPC-A, UPC-E, CODE-128, etc.) and
prints the decoded content to the terminal.

**No webcam or live camera is used anywhere in this project.** You provide
one image (a file path), and the program returns the decoded text.

Detection and decoding are done
with:
- `cv2.QRCodeDetector` — OpenCV's native multi-QR detector/decoder.
- `cv2.barcode.BarcodeDetector` — OpenCV's native 1D barcode detector/decoder
  (from the `opencv-contrib-python` package).
- A hand-written classical CV localization stage (Scharr gradients +
  morphology) that highlights candidate barcode regions on the output image,
  demonstrating the underlying image-processing technique rather than
  treating the decoder as a black box.

---

## 1. Project Structure

```
qr-barcode-scanner/
├── scanner.py                 
├── requirements.txt          
├── sample_images/             
│   ├── sample_qr.png
│   └── sample_ean13.png
├── outputs/                   
└── README.md
```

---

## 2. Environment Setup

### Step 2.1 — Prerequisites
- Python **3.9 or newer** installed on your system.

### Step 2.2 — Clone the repository
```bash
git clone https://github.com/Aa-Par/QR-Code-Barcode-Scanner.git
cd QR-Code-Barcode-Scanner
```

### Step 2.3 — Create and activate a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 2.4 — Install dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `opencv-contrib-python` — provides both `cv2.QRCodeDetector` and the
  contrib-only `cv2.barcode.BarcodeDetector` module.
- `numpy` — array operations used in the classical localization step.

> **Important:** Do not install plain `opencv-python` instead of
> `opencv-contrib-python`. The barcode decoder (`cv2.barcode`) only ships in
> the `contrib` build. If both are somehow installed together they can
> conflict — run `pip uninstall opencv-python opencv-python-headless` first
> if you hit an `AttributeError: module 'cv2' has no attribute 'barcode'`.

---

## 3. Configuration

No configuration files, API keys, or environment variables are required.
Everything runs fully offline and locally once the dependencies above are
installed.

---

## 4. Running the Project

The script accepts an image in two ways:

### Option A — Pass the image path as a command-line argument
```bash
python3 scanner.py --image sample_images/yt.jpg
python3 scanner.py --image sample_images/ean13.png
```

### Option B — Run without arguments and be prompted interactively
```bash
python3 scanner.py
```
```
Enter path to the QR/Barcode image: sample_images/yt.jpg
```

### Optional flag
```bash
python3 scanner.py --image sample_images/yt.jpg --output-dir my_results
```
`-o / --output-dir` controls where the annotated result image is saved
(defaults to `outputs/`).

### Expected terminal output
```
[INFO] Loaded image: D:\#VIT B\Projects\qr-barcode-scanner\sample_images\yt.jpg  (shape: 474x474)
[INFO] Running classical gradient-based barcode localization ...
[INFO]   -> 16 candidate region(s) found.
[INFO] Running QR code detection & decoding ...
[INFO] Running barcode detection & decoding ...

============================================================
SCAN RESULTS
============================================================
[QR #1]  Type: QRCODE  |  Data: https://www.youtube.com/watch?v=dQw4w9WgXcQ
============================================================

[INFO] Annotated result image saved to: d:\qr-barcode-scanner\outputs\yt_annotated_20260918_133547.png
```

An annotated copy of the input image — with green boxes around detected QR
codes, blue boxes around detected/decoded barcodes, and light gray boxes
marking the classical gradient-based candidate regions — is written to the
`outputs/` folder for visual verification.

### Testing with your own images
Any standard image format supported by OpenCV works: `.png`, `.jpg`,
`.jpeg`, `.bmp`, `.tiff`. Just point `--image` at the file. For best
results:
- Make sure the code fills a reasonable portion of the frame.
- Avoid extreme blur or very low resolution.
- A small amount of quiet/white space around the code helps decoding.

The sample files in `sample_images/` (`yt.jpg`, `ean13.png`,) are provided                                                                                                  as ready-to-use test fixtures — no extra setup or
generation step is needed to use them.

---

## 5. How It Works:

1. **Load** the image from disk with OpenCV (`cv2.imread`).
2. **QR branch:** `cv2.QRCodeDetector().detectAndDecodeMulti()` finds and
   decodes every QR code in the frame in one pass, with a single-code
   fallback (`detectAndDecode`) for edge cases.
3. **Barcode branch:**
   - Localization: a Scharr-gradient-difference +
     morphological-closing pipeline highlights regions with the strong,
     closely spaced vertical edges characteristic of 1D barcodes.
   - Decoding: `cv2.barcode.BarcodeDetector().detectAndDecodeWithType()`
     locates and decodes the symbol, reporting both the payload and the
     symbology (EAN-13, EAN-8, UPC-A, UPC-E, CODE-128, ...). A very light
     Gaussian blur is applied before detection (with a raw-image fallback)
     because OpenCV's gradient-orientation detector is tuned for the soft
     edges of real camera photos rather than perfectly crisp synthetic
     bars — this measurably improves the detection rate on clean/scanned
     inputs without affecting decoding accuracy.
4. **Annotate** every detected symbol with its bounding polygon and decoded
   text, and save the result image.
5. **Report** all decoded values to the terminal.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `AttributeError: module 'cv2' has no attribute 'barcode'` | You have plain `opencv-python` installed instead of `opencv-contrib-python`. Run `pip uninstall opencv-python opencv-python-headless -y` then `pip install -r requirements.txt` again. |
| `No QR codes or barcodes could be decoded` | Try a sharper / higher-resolution photo, ensure good lighting/contrast, and leave some white margin around the code. |
| `[ERROR] File not found` | Double-check the path you passed to `--image`; use an absolute path if unsure. |
