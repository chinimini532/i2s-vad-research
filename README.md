# I2S VAD Research
Voice Activity Detection under G.711 A-law telephony codec constraints,
deployed on Raspberry Pi CM5 via custom I2S kernel driver.

## Paper Target
MDPI Sensors / Electronics

## Project Structure
- `src/data/` — dataset download, preprocessing, splitting
- `src/models/` — model architectures
- `src/training/` — training loop and config
- `src/utils/` — shared utilities
- `notebooks/` — evaluation and visualization
- `outputs/stats/` — model performance CSVs (tracked by git)
- `outputs/figures/` — plots and confusion matrices as PDF (tracked by git)

## Setup

### 1. Clone and install
```bash
git clone <your-repo-url>
cd i2s-vad-research
pip install -r requirements.txt
```

### 2. GPU Setup in VS Code (IMPORTANT for MSI RTX 3050)
VS Code terminal does not always detect GPU automatically. Do this:

**Check GPU is visible:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**If it shows False, run this before training:**
```bash
# Windows (MSI)
set CUDA_VISIBLE_DEVICES=0
python src/training/train.py

# Or set permanently in VS Code:
# File → Preferences → Settings → search "terminal env"
# Add to settings.json:
# "terminal.integrated.env.windows": {"CUDA_VISIBLE_DEVICES": "0"}
```

**Verify CUDA version matches PyTorch:**
```bash
nvidia-smi  # check your CUDA version
# Then install matching PyTorch from https://pytorch.org
# Example for CUDA 11.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download dataset (LG Gram — small test)
```bash
python src/data/download.py --mode test
```

### 4. Download dataset (MSI — full dataset)
```bash
python src/data/download.py --mode full
```

### 5. Preprocess (A-law simulation)
```bash
python src/data/preprocess.py
```

### 6. Create splits
```bash
python src/data/split.py
```

### 7. Train all models
```bash
python src/training/train.py
```

### 8. Evaluate
Open `notebooks/evaluation.ipynb` and run all cells.
Results saved automatically to `outputs/stats/` and `outputs/figures/`.

## Workflow (Two-laptop setup)
- **LG Gram (weekdays):** Code, debug with `--mode test` (small data subset)
- **MSI RTX 3050 (weekend):** Pull from GitHub, run full training with GPU

```bash
# On MSI after pulling
git pull origin main
python src/data/download.py --mode full
python src/data/preprocess.py
python src/training/train.py
```
