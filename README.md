# SMPL Sensor Annotator

An interactive web-based tool for selecting sensor locations on the SMPL body model and exporting corresponding vertex, UV, and grid mappings.

### Demo

A short demonstration video is available in:

https://github.com/user-attachments/assets/f7962e10-7bc4-4c4f-8728-4cb67214b7b6

## Features

- Interactive 3D visualization of the SMPL body model
- Synchronized 3D mesh and UV map views
- Click directly on the body surface or UV map to place sensors
- Automatic correspondence between 3D vertices and UV coordinates
- Configurable sensor grid (default: 32 × 16)
- Assign sensors to grid locations
- Export annotations as JSON
- Adjustable SMPL parameters
  - Gender
  - Body shape (betas)
  - Body pose
  - Global orientation

---

## Exported Data

Each annotation includes:

- Sensor ID
- SMPL vertex index
- 3D coordinates (x, y, z)
- UV coordinates (u, v)
- Assigned grid row and column
- Complete sensor grid mapping
- Current SMPL parameters

Example:

```json
{
  "sensor_id": 1,
  "vertex_index": 3456,
  "position": {
    "x": 0.021,
    "y": -0.835,
    "z": 0.087
  },
  "uv": {
    "u": 0.356,
    "v": 0.712
  },
  "grid": {
    "row": 4,
    "column": 12
  }
}
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SMPL-Sensor-Annotator.git
cd SMPL-Sensor-Annotator
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Download Required SMPL Assets

This repository **does not** include any SMPL models or UV assets.

Please download the official files from the SMPL website:

https://smpl.is.tue.mpg.de/

After downloading, organise them as follows:

```
SMPL-Sensor-Annotator/
│
├── smpl_sensor_annotator.py
└── body_models/
    ├── smpl/
    │   ├── SMPL_MALE.pkl
    │   ├── SMPL_FEMALE.pkl
    │   └── SMPL_NEUTRAL.pkl
    │
    └── smpl_uv_20200910/
        └── smpl_uv.obj
```

---

## Run

```bash
python smpl_sensor_annotator.py
```

The application will automatically open

```
http://127.0.0.1:5000
```

---

## Typical Workflow

1. Load the SMPL model.
2. Adjust body shape or pose if required.
3. Click on the 3D model or UV map to place sensors.
4. Assign sensors to the sensor grid.
5. Export the annotation as a JSON file.

---

## Repository Structure

```
SMPL-Sensor-Annotator/
│
├── smpl_sensor_annotator.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
└── body_models/      (not included)
```

---

## License

The source code in this repository is released under the MIT License.

This repository **does not** include the SMPL body models or UV assets.

Please obtain the official SMPL files directly from the SMPL website and comply with their license terms.

---

## Citation

If you use this tool in your research, please cite this repository.

```bibtex
@software{smpl_sensor_annotator,
  title={SMPL Sensor Annotator},
  author={Zhen Liang},
  year={2026},
  url={https://github.com/YOUR_USERNAME/SMPL-Sensor-Annotator}
}
```

---

## Acknowledgements

This tool is built upon the excellent **SMPL** body model developed by the Max Planck Institute for Intelligent Systems.

SMPL:

Loper M, Mahmood N, Romero J, Pons-Moll G, Black MJ.
**SMPL: A Skinned Multi-Person Linear Model.**
ACM Transactions on Graphics (SIGGRAPH Asia), 2015.
