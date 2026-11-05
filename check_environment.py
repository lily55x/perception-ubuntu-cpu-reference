"""Check dependencies and CPU inference without opening a camera."""
import json
import platform
from pathlib import Path
import sys

import cv2
import torch
import torchvision
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    model = YOLO(str(ROOT / 'models' / 'yolo26n.pt'))
    result = model.predict(str(ROOT / 'assets' / 'bus.jpg'), device='cpu',
                           imgsz=416, conf=0.25, verbose=False)[0]
    if model.predictor.device.type != 'cpu' or result.boxes.data.device.type != 'cpu':
        raise RuntimeError('CPU inference device check failed')
    if len(result.boxes) == 0:
        raise RuntimeError('No objects detected on the sample image')
    output = ROOT / 'runs' / 'environment-check'
    output.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(output / 'bus-detection.jpg'))
    report = {
        'os': platform.platform(), 'python': sys.version, 'executable': sys.executable,
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'opencv': cv2.__version__, 'ultralytics': ultralytics.__version__,
        'model': 'yolo26n.pt', 'inference_device': str(model.predictor.device),
        'cuda_required': False, 'camera_tested': False,
        'detections': [{'class': result.names[int(c)], 'confidence': round(float(s), 4)}
                       for c, s in zip(result.boxes.cls, result.boxes.conf)],
        'cpu_inference': 'passed',
    }
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
