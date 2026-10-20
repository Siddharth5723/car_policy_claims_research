# Car Damage Estimation

An AI-powered system that detects vehicle damage from photos, classifies its type and severity, and produces an estimated repair cost, reducing manual inspection time for insurers, repair shops and fleet operators.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
7. [Usage](#usage)
8. [API Reference](#api-reference)
9. [Dataset](#dataset)
10. [Model & Training](#model--training)
11. [Evaluation](#evaluation)
12. [Roadmap](#roadmap)
13. [Limitations](#limitations)
14. [References](#references)
15. [Contributing](#contributing)
16. [License](#license)

---

## Overview

Manual vehicle damage assessment is slow, subjective and inconsistent. This project automates it:

1. A user uploads one or more photos of a damaged vehicle.
2. The system **detects and localizes** damaged regions.
3. Each region is **classified** by damage type (dent, scratch, crack, broken part, etc.) and **severity** (minor / moderate / severe).
4. A cost-estimation module maps damage + affected part to an **estimated repair range**.
5. A structured report is returned (JSON and/or PDF).

The design of this project follows the multi-stage approach proposed by Mallios et al. in *"Vehicle Damage Severity Estimation for Insurance Operations Using In-The-Wild Mobile Images"* (IEEE Access, 2023) [[1]](#references). Rather than training one end-to-end model, the paper decomposes the problem into several specialized computer-vision models (vehicle detection, view angle, parts and damage segmentation, airbag detection) whose outputs are fused with structured claim data and passed to a gradient-boosted model for the final severity prediction. This keeps the vision pipeline stable while the cost model can be updated cheaply, and makes every intermediate output inspectable.

## Features

- Multi-image upload per claim
- Damage localization (bounding boxes / segmentation masks)
- Damage type and severity classification
- Affected car-part identification (bumper, hood, door, fender, etc.)
- Repair cost estimation with min/max range
- Annotated output images with overlays
- REST API for integration into claims or workshop systems
- Confidence scores on every prediction, with low-confidence cases flagged for human review

## System Architecture

```
 Image(s) ──► Preprocessing ──► Damage Detection/Segmentation
                                        │
                                        ▼
                         Part Identification + Severity Classifier
                                        │
                                        ▼
                              Cost Estimation Engine
                                        │
                                        ▼
                        Report Generator (JSON / PDF / Overlay)
```

The reference pipeline in the paper [[1]](#references) runs these models independently on every image, then fuses the results:

| Stage | Model in the paper |
|---|---|
| Vehicle detector | YOLOv4, pre-trained on COCO |
| Interior / exterior classifier | VGG-16 |
| View-angle classifier (8 bins of 45°) | ResNet-18, regressing (x, y) on the unit circle |
| Parts detection (24 classes) and damage detection (5 classes) | Mask R-CNN with deformable convolutions (ResNet-50 + FPN) |
| Airbag deployment classifier | ResNet-34 |
| Feature fusion | Per-panel damage IoU (max across images) + airbag probability + structured data, giving a 50-dimension vector |
| Severity / cost model | XGBoost (via LightGBM) |

> Adapt this table to what your implementation actually uses. The paper's code and data were withheld for commercial reasons, so this project is an independent implementation inspired by its design.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch, torchvision |
| Detection / segmentation | YOLO or Mask R-CNN (configurable) |
| Image processing | OpenCV, Albumentations |
| API | FastAPI, Uvicorn |
| Storage | MongoDB (claims, results), local or cloud object storage (images) |
| Deployment | Docker, GCP (Cloud Run / Compute Engine) |

> Adjust this table to match what you actually use.

## Project Structure

```
car-damage-estimation/
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and split data
│   └── annotations/        # COCO / YOLO format labels
├── notebooks/              # EDA and experiments
├── src/
│   ├── preprocessing/      # Resize, normalize, augment
│   ├── models/             # Detection, severity, part models
│   ├── training/           # Training and validation loops
│   ├── inference/          # Prediction pipeline
│   ├── cost/               # Cost estimation logic
│   ├── reports/            # Report and overlay generation
│   └── api/                # FastAPI app
├── configs/                # YAML configs for models and training
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- (Recommended) NVIDIA GPU with CUDA for training
- MongoDB instance, if using persistence

### Installation

```bash
git clone https://github.com/<your-username>/car-damage-estimation.git
cd car-damage-estimation

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

```env
MODEL_PATH=weights/best.pt
MONGO_URI=mongodb://localhost:27017
CONFIDENCE_THRESHOLD=0.4
```

## Usage

### Run inference on a single image

```bash
python -m src.inference.predict --image path/to/car.jpg --output results/
```

### Start the API server

```bash
uvicorn src.api.main:app --reload --port 8000
```

Interactive docs are available at `http://localhost:8000/docs`.

### Train a model

```bash
python -m src.training.train --config configs/train.yaml
```

## API Reference

### `POST /estimate`

Upload one or more images and receive a damage assessment.

**Request:** `multipart/form-data` with `files` (one or more images).

**Example response:**

```json
{
  "claim_id": "a1b2c3",
  "damages": [
    {
      "part": "front_bumper",
      "type": "dent",
      "severity": "moderate",
      "confidence": 0.87,
      "bbox": [120, 340, 410, 520],
      "estimated_cost": { "min": 8000, "max": 14000, "currency": "INR" }
    }
  ],
  "total_estimated_cost": { "min": 8000, "max": 14000, "currency": "INR" },
  "needs_human_review": false
}
```

### `GET /claims/{claim_id}`

Retrieve a stored assessment.

### `GET /health`

Service health check.

## Dataset

Describe the data used for training:

- **Source:** (e.g., public Kaggle / Roboflow car-damage datasets, or custom collected)
- **Size:** number of images and annotated instances
- **Classes:** damage types, car parts, severity levels
- **Split:** train / validation / test ratios
- **Annotation format:** COCO or YOLO

Add the licence and attribution for any third-party dataset.

## Model & Training

- **Base model:** pretrained backbone fine-tuned on car-damage data
- **Augmentation:** flips, brightness/contrast, blur, random crops
- **Loss:** detection loss plus cross-entropy for severity
- **Optimizer / schedule:** fill in once finalized
- **Cost model:** rule-based lookup (part × damage type × severity) or a regression model trained on repair records

## Evaluation

| Metric | Value |
|---|---|
| mAP@0.5 | _TBD_ |
| mAP@0.5:0.95 | _TBD_ |
| Severity accuracy | _TBD_ |
| Part identification accuracy | _TBD_ |
| Cost MAE | _TBD_ |

Replace the placeholders with your measured results and include a few sample prediction images.

## Roadmap

- [ ] Video input (walk-around capture)
- [ ] Multi-view fusion to avoid double-counting damage
- [ ] Regional pricing and parts database integration
- [ ] Fraud and pre-existing damage detection
- [ ] Mobile capture app with guided photo prompts
- [ ] Explainability overlays (Grad-CAM)

## Limitations

- Accuracy drops with poor lighting, reflections, mud or heavy occlusion.
- Internal or structural damage cannot be seen in photos.
- Cost estimates are indicative only and do not replace a professional inspection.
- Performance depends on how well the training data matches the target vehicles and region.

## References

[1] D. Mallios, L. Xiaofei, N. McLaughlin, J. Martinez Del Rincon, C. Galbraith, and R. Garland, "Vehicle damage severity estimation for insurance operations using in-the-wild mobile images," *IEEE Access*, vol. 11, pp. 78644–78655, 2023. DOI: [10.1109/ACCESS.2023.3299223](https://doi.org/10.1109/ACCESS.2023.3299223)

BibTeX:

```bibtex
@article{mallios2023vehicle,
  author  = {Mallios, Dimitrios and Xiaofei, Li and McLaughlin, Niall and Martinez Del Rincon, Jesus and Galbraith, Clare and Garland, Rory},
  title   = {Vehicle Damage Severity Estimation for Insurance Operations Using In-The-Wild Mobile Images},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {78644--78655},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.3299223}
}
```

The paper is open access under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch and open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for details.

## Author

**Sidd** — B.Tech CSE (Machine Learning & Generative AI), Amity University Rajasthan
