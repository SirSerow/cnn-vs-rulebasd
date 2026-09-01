# CNN vs Rule-Based Conveyor Counter

This repository contains a Raspberry Pi 4 experiment comparing two object-counting approaches on conveyor-belt video:

1. YOLO26n inference with ONNX Runtime on the CPU.
2. Traditional OpenCV processing using thresholding, morphology, and contour detection.

Both approaches use the same tracking, line-crossing counter, visualization, and benchmarking pipeline so that the comparison is as fair as possible.

The implementation roadmap, architecture, experimental protocol, and acceptance criteria are defined in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Planned command-line interface

```bash
python app.py --mode opencv --source videos/sample.mp4
python app.py --mode yolo --source videos/sample.mp4
python app.py --mode yolo --source 0 --output results/videos/yolo_run.mp4
python app.py --mode opencv --source videos/sample.mp4 --no-render --benchmark
```

> The application has not been implemented yet. The repository currently contains the agreed project plan.
