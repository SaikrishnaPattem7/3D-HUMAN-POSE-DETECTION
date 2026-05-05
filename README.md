# 3D Human Pose Estimation

Lifts 2D human joint coordinates(detected from images/video) into a full 3D skeleton using deep learning — no motion-capture suits or multi-camera setups required.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Train on synthetic data (no dataset required)**
```bash
python main.py --mode train --model fcn --epochs 10
```

**Train on Human3.6M**
```bash
python main.py --mode train --model fcn --data-path data/human36m/ --epochs 50
```

**Run on a video**
```bash
python main.py --mode video --input video.mp4 --checkpoint checkpoints/fcn_best.pth
```

**Run on pilot study Excel data**
```bash
python main.py --mode excel --input raw_data/pilot_study.xlsx --checkpoint checkpoints/fcn_best.pth

# Filter to one participant
python main.py --mode excel --input raw_data/pilot_study.xlsx --checkpoint checkpoints/fcn_best.pth --participant P1

# Save output to file
python main.py --mode excel --input raw_data/pilot_study.xlsx --checkpoint checkpoints/fcn_best.pth --save-fig output.png
```

## Models

- **FCN** — residual fully-connected network (Martinez et al. 2017)
- **GCN** — graph convolutional network over the COCO skeleton adjacency matrix

Checkpoints are saved to `checkpoints/<model_type>_best.pth` and include the normalization stats needed for inference.

## Data

- **Human3.6M**: preprocessed `.npz` files with 2D and 3D joint positions. Place in `data/human36m/`.
- **pilot_study.xlsx**: pre-extracted 2D COCO keypoints for 78,877 rows across participants. Place in `raw_data/`.


All modules use the **COCO 17-joint** format throughout.
