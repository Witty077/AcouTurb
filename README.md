0. End-to-end workflow
```mermaid
graph TD
    A[Raw Signal] --> B[Frontend: Feature Extractor]
    B --> C[Neural Encoder: CNN]
    C --> D[Deep SVDD Objective]
    D --> E[Anomaly Score: Distance to Hypersphere Center]
```

1. Repo structure
```text
anomalous-signal-detection/
│
├── configs/
│   ├── default.yaml
│   └── experiments/
│
├── data/
│   └── dataset.py
│
├── frontend/
│   ├── base_frontend.py
│   ├── logmel.py
│   └── physical_filters/      # future physics modules
│
├── models/
│   └──cnn_encoder.py
│
├── losses/
│   └── svdd.py
│
├── evaluation/
│   └── metrics.py
│
├── scripts/
│   ├── train.py
│   └── test.py
│
└── README.md
```
