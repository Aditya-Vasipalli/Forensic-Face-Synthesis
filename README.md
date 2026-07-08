# ForenSynth — Forensic Face Synthesis

A comprehensive platform for generating, comparing, and enhancing forensic facial images using state-of-the-art generative models. **ForenSynth** combines sketch-to-face synthesis, face aging, and image restoration to support forensic investigations and research.

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Language](https://img.shields.io/badge/Language-Python%2057%25%20%7C%20JavaScript%2016%25%20%7C%20CSS%2014%25%20%7C%20HTML%2013%25-blue.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-July%202026-brightgreen.svg)

---

## Features

### 🎨 **Sketch-to-Face Generation**
Convert forensic sketches into realistic facial images using multiple GAN architectures:
- **CycleGAN**: Unpaired image-to-image translation
- **Pix2Pix**: Paired conditional image translation
- **Pix2Pix + ArcFace**: Identity-preserving generation with facial feature matching

### 🔍 **Face Comparison**
Side-by-side comparison tool to evaluate synthesis quality and match generated faces with reference images.

### ✨ **Face Restoration & Clarification**
AI-powered face enhancement and restoration using **GFPGAN**:
- Upscaling (1–4×)
- Denoising
- Sharpening
- Artifact removal

### 👤 **Face Aging Simulation**
Realistic age progression and regression using **UNet-based** age transformation:
- Morphing videos between age states
- Forensic age estimation aid
- Age-specific facial variation synthesis

---

## Architecture

### Backend Stack
- **Python 3.8+** with PyTorch for deep learning
- **Flask** REST API for model inference
- **Gradio** for interactive age transformation UI
- **FastAPI** (optional) for async endpoints

### Frontend
- **HTML5** + **CSS3** responsive design
- **Vanilla JavaScript** for dynamic UI and API communication
- Multi-tab interface for different synthesis modes

### Models
1. **Sketch-to-Face Generators**: U-Net–based encoder-decoder architecture with skip connections
2. **ArcFace Identity Loss**: Pre-trained w600k_r50 model for identity preservation
3. **GFPGAN v1.4**: Face restoration and upscaling
4. **Age Transformation UNet**: Specialized for age progression/regression

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- ~5 GB storage for pre-trained models

### Step 1: Clone the Repository
```bash
git clone https://github.com/Aditya-Vasipalli/Forensic-Face-Synthesis.git
cd Forensic-Face-Synthesis
```

### Step 2: Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

**For the aging module:**
```bash
cd aging
pip install -r requirements.txt
cd ..
```

**For the main synthesis models:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask pillow opencv-python gfpgan facexlib insightface onnx onnx2torch
```

### Step 4: Download Pre-trained Models
Models are auto-downloaded on first run. For the **ArcFace** model:
```python
python -c "from arcface import _ensure_onnx_downloaded; _ensure_onnx_downloaded()"
```

---

## Quick Start

### Option A: Web Interface

**1. Start the aging API server:**
```bash
cd aging
python app.py
```
This launches a Gradio interface at `http://localhost:7860`

**2. Start the synthesis API server (in a new terminal):**
```bash
python flask_api_clarify.py
```
The API listens on `http://localhost:7003`

**3. Serve the frontend:**
```bash
cd frontend
python -m http.server 8000
```
Open `http://localhost:8000` in your browser.

### Option B: Python API

```python
import torch
from model_pix2pix import Generator
from PIL import Image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
model.load_state_dict(torch.load("output_pix2pix/checkpoint_epoch_200.pt")["G"])
model.eval()

# Load sketch
sketch = Image.open("path/to/sketch.jpg").convert("RGB")
sketch_tensor = transforms.ToTensor()(sketch).unsqueeze(0).to(device)

# Generate face
with torch.no_grad():
    generated_face = model(sketch_tensor)
    
# Save result
output = transforms.ToPILImage()(generated_face.squeeze(0).cpu())
output.save("generated_face.jpg")
```

---

## Dataset Structure

Organize your training data as follows:
```
dataset/
├── sketches/
│   ├── F2-person1-sz1.jpg
│   ├── F2-person2-sz1.jpg
│   ├── M2-person3-sz1.jpg
│   └── ...
└── photos/
    ├── f-person1.jpg
    ├── f-person2.jpg
    ├── m-person3.jpg
    └── ...
```

**Naming Convention**: The dataset loader maps sketches to photos automatically:
- `F2-` prefix → `f-` prefix
- `M2-` prefix → `m-` prefix
- `-sz1` suffix is removed during mapping

---

## Training

### Train Pix2Pix Model

```bash
python train_pix2pix.py
```

**Configuration** (edit `train_pix2pix.py`):
```python
SKETCH_DIR = "dataset/sketches"
PHOTO_DIR = "dataset/photos"
BATCH_SIZE = 4
EPOCHS = 200
LR = 2e-4
LAMBDA_L1 = 100
```

**Output**:
- Checkpoints: `output_pix2pix/checkpoint_epoch_*.pt`
- Sample visualizations: `output_pix2pix/samples/epoch_*.png`

---

## API Endpoints

### Generate Faces
```
POST /api/cyclegan/generate
POST /api/pix2pix/generate
POST /api/pix2pix-arcface/generate
```
**Body**: `multipart/form-data { image: File }`  
**Response**: `{ output_url: string, inference_ms: number }`

### Enhance & Restore
```
POST /api/clarify/enhance
```
**Body**:
```
multipart/form-data {
  image: File,
  scale: 1-4,
  denoise: boolean,
  sharpen: boolean
}
```
**Response**: `{ output_url: "data:image/png;base64,..." }`

### Age Transformation
```
POST /api/aging/generate
```
**Body**:
```
multipart/form-data {
  image: File,
  source_age: 18-80,
  target_age: 18-80
}
```
**Response**: `{ output_url: string }`

---

## Project Structure

```
Forensic-Face-Synthesis/
├── arcface.py                 # ArcFace identity loss module
├── model_pix2pix.py           # Pix2Pix U-Net generator & discriminator
├── dataset_loader.py          # Sketch-photo dataset loader
├── train_pix2pix.py           # Training script for Pix2Pix
├── flask_api_clarify.py       # GFPGAN face restoration API
├── aging/
│   ├── app.py                 # Gradio interface for age transformation
│   ├── models.py              # UNet model for aging
│   ├── test_functions.py      # Image processing utilities
│   ├── test_compress.py       # Age transformation core logic
│   └── requirements.txt        # Aging module dependencies
├── frontend/
│   ├── index.html             # Main web interface
│   ├── app.js                 # Frontend logic & API calls
│   ├── style.css              # UI styling
│   └── ...
└── README.md
```

---

## Key Technologies

| Component | Technology |
|-----------|------------|
| **Deep Learning** | PyTorch, TorchVision |
| **Generative Models** | Pix2Pix, CycleGAN, GFPGAN |
| **Identity Preservation** | ArcFace w600k_r50 (insightface) |
| **Face Detection** | dlib, facexlib |
| **APIs** | Flask, FastAPI, Gradio |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |

---

## Performance

- **Sketch-to-Face Generation**: ~200–500ms per image (GPU) / 1–2s (CPU)
- **Face Enhancement**: ~300–800ms per image depending on scale
- **Age Transformation**: ~400–600ms per image
- **Batch Processing**: Supported for improved throughput

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Roadmap

- [ ] Real-time video processing for age progression
- [ ] Multi-face batch processing
- [ ] Extended age range support (0–100 years)
- [ ] Emotion and expression transfer
- [ ] Docker containerization for easy deployment
- [ ] REST API documentation (OpenAPI/Swagger)

---

## Support & Issues

If you encounter issues or have questions:
1. Check existing [Issues](https://github.com/Aditya-Vasipalli/Forensic-Face-Synthesis/issues)
2. Review the [Discussions](https://github.com/Aditya-Vasipalli/Forensic-Face-Synthesis/discussions)
3. Open a new issue with detailed reproduction steps

---

## Citation

If you use ForenSynth in your research, please cite:

```bibtex
@software{forensynth2026,
  author = {Vasipalli, Aditya},
  title = {ForenSynth: Forensic Face Synthesis Platform},
  year = {2026},
  url = {https://github.com/Aditya-Vasipalli/Forensic-Face-Synthesis}
}
```

---

## Acknowledgments

- **ArcFace** by Jiankang Deng and team ([insightface](https://github.com/deepinsight/insightface))
- **GFPGAN** by Xintao Wang et al.
- **Pix2Pix** by Phillip Isola et al.
- **CycleGAN** by Jun-Yan Zhu et al.

---

**Made with ❤️ for forensic research and development.**
