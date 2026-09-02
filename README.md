# CNN vs Rule-Based Conveyor Image Counter

This repository plans a Raspberry Pi 4 experiment comparing two approaches on
annotated images from a real conveyor:

1. YOLO26n inference through ONNX Runtime on the CPU.
2. Traditional OpenCV color thresholding, morphology, and contour detection.

Both modes receive the same normalized images and return the same bounding-box
contract. Their boxes and per-image object counts are evaluated against the
provided annotations. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete
experimental protocol.

## Dataset

The baseline input is Edge Impulse's
[Cubes on conveyor belt (colors)](https://docs.edgeimpulse.com/datasets/image/cubes-on-conveyor-belt-colors)
object-detection dataset. It contains real, top-down conveyor images with
clearly separated blue, green, red, and yellow cubes on a dark belt. The source
dataset is licensed under
[BSD 3-Clause Clear](https://spdx.org/licenses/BSD-3-Clause-Clear.html).

For reproducible deployment, the downloader uses Edge Impulse's
[Hugging Face mirror](https://huggingface.co/datasets/edgeimpulse/cubes-on-conveyor-belt)
at commit
[`e3d1c8b`](https://huggingface.co/datasets/edgeimpulse/cubes-on-conveyor-belt/commit/e3d1c8b0c4872b70fcd77d86dec3bde7875e6054).
The pinned snapshot has:

| Property | Value |
|---|---:|
| Images | 70 |
| Training images | 55 |
| Testing images | 15 |
| Bounding boxes | 156 |
| Objects per image | 1–4 |
| Source labels | `blue`, `green`, `red`, `yellow` |
| Evaluation label | `cube` (all colors combined) |

### Example data image

The following downloaded training image has four annotated, spatially
separated cubes. Its ground-truth count is **4**.

![Four colored cubes separated on a black conveyor belt](docs/images/cubes-on-conveyor-example.jpg)

Image source: Edge Impulse, *Cubes on conveyor belt (colors)* dataset. A local
copy is included only as a documented preview; the complete dataset is fetched
during deployment.

### Deployment download

The complete dataset is not committed to this repository. Install Git LFS and
run the pinned downloader:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs
git lfs install

python3 scripts/download_dataset.py
```

No account or API key is required. The command downloads the exact pinned
revision, verifies all image/annotation pairs, checks that LFS images were
materialized, and confirms the expected splits, classes, and box count. It
writes non-secret provenance metadata to:

```text
datasets/cubes-on-conveyor-belt/.source.json
```

The command is idempotent: an existing valid dataset is checked and reused.
Dataset facts and limitations are versioned in
[`data/dataset_sources.json`](data/dataset_sources.json), and attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Ground truth and metrics

Each split contains a `bounding_boxes.labels` JSON manifest. For a given image,
the number of objects in its `boundingBoxes[filename]` array is the true cube
count. Color labels are collapsed to a single `cube` class for a fair comparison
because the experiment measures object detection and counting, not color
classification.

This supports:

- bounding-box precision, recall, F1, and mAP;
- mean absolute count error per image;
- percentage of images with an exactly correct count;
- total predicted boxes versus total ground-truth boxes;
- median/P95 latency and sustained images per second.

The annotations are independent of both detector outputs, so neither method
defines its own truth.

## Limitations

These are annotated still images, not a documented ordered video sequence.
They cannot evaluate tracking, line crossings, duplicate temporal counts, or
video playback FPS. The first application version will output annotated images
and a metrics summary.

The 15-image testing split is appropriate for a smoke benchmark but too small
for a strong general claim. Several images also appear to come from related
capture sequences, so the published split may contain near-neighbor leakage.
Report this limitation and use a larger, capture-grouped held-out set before
publishing definitive accuracy conclusions.

## Planned command-line interface

```bash
python app.py --mode opencv --dataset datasets/cubes-on-conveyor-belt --split testing
python app.py --mode yolo --dataset datasets/cubes-on-conveyor-belt --split testing
python app.py --mode opencv --dataset datasets/cubes-on-conveyor-belt --split testing --benchmark --no-render
python app.py --mode yolo --dataset datasets/cubes-on-conveyor-belt --split testing --output results/yolo-testing
```

> The inference application is not implemented yet. The repository currently
> contains the project plan and reproducible dataset-download preparation.
