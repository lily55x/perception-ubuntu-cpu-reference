"""Run a local YOLO model on a webcam, image, or video. Q/Esc exits; S saves a screenshot."""
import argparse
from contextlib import contextmanager
import os
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
import tempfile

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WINDOW = 'YOLO Camera | Q/Esc: quit | S: screenshot | H: info'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0', help='Camera index (0, 1...) or a local image/video path')
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / 'yolo26n.pt')
    parser.add_argument('--device', default='cpu', choices=['auto', 'cpu', '0'])
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1), help='PyTorch CPU threads')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold, between 0 and 1')
    parser.add_argument('--imgsz', type=int, default=416, help='Inference input size; multiple of 32')
    parser.add_argument('--width', type=int, default=640, help='Requested camera width')
    parser.add_argument('--height', type=int, default=480, help='Requested camera height')
    parser.add_argument('--fps', type=float, default=15, help='Requested camera capture FPS (hardware may ignore it)')
    parser.add_argument('--hud-opacity', type=float, default=0.3, help='Info panel background opacity, 0 to 1')
    parser.add_argument('--hide-info', action='store_true', help='Start with the info panel hidden; H toggles it')
    parser.add_argument('--backend', choices=['auto', 'v4l2'], default='auto',
                        help='Linux camera backend; auto tries V4L2 first')
    parser.add_argument('--max-frames', type=int, default=0, help='Stop after this many frames; 0 means unlimited')
    parser.add_argument('--no-display', action='store_true', help='Run without a window; specify --max-frames for camera/video')
    parser.add_argument('--save-output', action='store_true', help='Save the last annotated frame on exit')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / 'camera')
    parser.add_argument('--report', type=Path, help='Optional JSON run summary, without captured images')
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error('--conf must be greater than 0 and at most 1')
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error('--imgsz must be a positive multiple of 32')
    if args.max_frames < 0 or args.width <= 0 or args.height <= 0:
        parser.error('Frame limit cannot be negative; width and height must be positive')
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps must be a finite positive number')
    if not 0 <= args.hud_opacity <= 1:
        parser.error('--hud-opacity must be between 0 and 1')
    if args.threads <= 0:
        parser.error('--threads must be positive')
    return args


def draw_info_panel(frame, lines, opacity):
    """Blend only the compact HUD region, keeping the live picture visible."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y, padding = 8, 8, 9
    available = frame.shape[1] - x - padding * 2 - 8
    if available <= 0 or frame.shape[0] < 70:
        return
    scales = []
    widths = []
    for i, line in enumerate(lines):
        scale = 0.52 if i == 0 else 0.46
        width = cv2.getTextSize(line, font, scale, 1)[0][0]
        if width > available:
            scale *= available / width
        scales.append(scale)
        widths.append(cv2.getTextSize(line, font, scale, 1)[0][0])
    width = max(widths) + padding * 2
    height = padding * 2 + 22 * len(lines)
    region = frame[y:y + height, x:x + width]
    tint = np.full_like(region, (24, 24, 24))
    cv2.addWeighted(region, 1 - opacity, tint, opacity, 0, dst=region)
    for i, (line, scale) in enumerate(zip(lines, scales)):
        origin = (x + padding, y + padding + 15 + i * 22)
        shadow = (origin[0] + 1, origin[1] + 1)
        cv2.putText(frame, line, shadow, font, scale, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(frame, line, origin, font, scale, (250, 250, 250), 1, cv2.LINE_AA)


def save_frame(frame, folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'detection-{datetime.now():%Y%m%d-%H%M%S-%f}.jpg'
    ok, encoded = cv2.imencode('.jpg', frame)
    if not ok:
        raise RuntimeError('Could not encode screenshot')
    encoded.tofile(str(target))  # Supports Chinese Windows directory names.
    print(f'Screenshot saved: {target}', flush=True)
    return str(target)


@contextmanager
def camera_guard(source):
    """Use a Linux advisory lock to prevent duplicate copies of this program."""
    if not sys.platform.startswith('linux') or not source.isdecimal():
        yield
        return
    import fcntl
    lock_path = Path(tempfile.gettempdir()) / f'perception-camera-{os.getuid()}-{int(source)}.lock'
    with lock_path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This camera is already running in another instance. Close it with Q/Esc first.') from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def is_black_frame(frame):
    return int(frame.max()) <= 3


def open_camera(index, width, height, backend_name='auto', requested_fps=30):
    if sys.platform.startswith('linux'):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if backend_name == 'auto' else [cv2.CAP_V4L2]
    elif sys.platform == 'win32':
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
    else:
        backends = [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        accepted = False
        try:
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if not capture.set(cv2.CAP_PROP_FPS, requested_fps):
                    print(f'Camera backend could not apply requested {requested_fps:g} FPS; using device settings.', flush=True)
                # A successful read may still be a synthetic all-black frame.
                for _ in range(20):
                    ok, frame = capture.read()
                    if ok and frame is not None and not is_black_frame(frame):
                        accepted = True
                        return capture, frame
                print(f'Camera backend {capture.getBackendName()} returned no usable non-black frames.', flush=True)
        finally:
            if not accepted:
                capture.release()
    raise RuntimeError(
        f'Cannot get a non-black picture from camera {index}. Check the privacy shutter, lighting, '
        'camera permissions, and other camera apps. On Ubuntu, check /dev/video* or try --source 1.'
    )


def run(args):
    torch.set_num_threads(args.threads)
    if not args.model.is_file():
        raise FileNotFoundError(f'Model not found: {args.model}. See README.md for the download URL.')
    camera = args.source.isdecimal()
    source = None if camera else Path(args.source).expanduser().resolve()
    if source is not None and not source.is_file():
        raise FileNotFoundError(f'Input file not found: {source}')
    image = source is not None and source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    if args.no_display and not image and not args.max_frames:
        raise ValueError('--no-display requires --max-frames for camera/video input')
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == '0' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or run check_environment.py.')
    print(f'Loading {args.model.name}; device={device}; source={args.source}', flush=True)
    model = YOLO(str(args.model))
    # Warm up the selected inference device before opening the camera.
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
                  device=device, imgsz=args.imgsz, verbose=False)
    capture = None
    frame = None
    annotated = None
    count = 0
    screenshots = []
    reason = 'end_of_input'
    last_counts = {}
    camera_reported_fps = None
    started = time.perf_counter()
    try:
        if image:
            frame = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f'Cannot decode image: {source}')
        elif camera:
            capture, frame = open_camera(int(args.source), args.width, args.height, args.backend, args.fps)
            reported = capture.get(cv2.CAP_PROP_FPS)
            camera_reported_fps = reported if math.isfinite(reported) and reported > 0 else None
            print(f'Camera opened: {frame.shape[1]}x{frame.shape[0]}, backend={capture.getBackendName()}, '
                  f'requested FPS={args.fps:g}, driver-reported FPS={camera_reported_fps}, '
                  f'initial pixel mean={frame.mean():.2f}', flush=True)
            if camera_reported_fps is not None and abs(camera_reported_fps - args.fps) > max(1, args.fps * 0.05):
                print('Requested FPS differs from driver-reported FPS. Resolution/FPS combinations depend on camera support.', flush=True)
        else:
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open video: {source}')
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Video contains no readable frames')
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, min(frame.shape[1], 1280), min(frame.shape[0], 720))
        fps = None
        black_frames = 0
        previous_tick = None
        first_tick = None
        loop_fps = None
        show_info = not args.hide_info
        started = time.perf_counter()
        while frame is not None:
            tick = time.perf_counter()
            if first_tick is None:
                first_tick = tick
            if previous_tick is not None:
                current_loop_fps = 1 / max(tick - previous_tick, 1e-6)
                loop_fps = current_loop_fps if loop_fps is None else loop_fps * 0.9 + current_loop_fps * 0.1
            previous_tick = tick
            result = model.predict(frame, device=device, imgsz=args.imgsz,
                                   conf=args.conf, verbose=False)[0]
            annotated = result.plot()
            black_frames = black_frames + 1 if camera and is_black_frame(frame) else 0
            count += 1
            last_counts = {}
            for category in result.boxes.cls.cpu().tolist():
                name = result.names[int(category)]
                last_counts[name] = last_counts.get(name, 0) + 1
            # This rate includes inference and annotation; camera/display wait time is excluded.
            instant = 1 / max(time.perf_counter() - tick, 1e-6)
            fps = instant if fps is None else fps * 0.9 + instant * 0.1
            label = f'{args.model.stem} | {model.predictor.device} | processing {fps:.1f} FPS | objects {len(result.boxes)}'
            loop_text = f'{loop_fps:.1f}' if loop_fps is not None else '--'
            camera_text = f'{camera_reported_fps:g}' if camera_reported_fps is not None else 'unknown'
            details = f'frame {frame.shape[1]}x{frame.shape[0]} | loop {loop_text} FPS | imgsz {args.imgsz}'
            if camera:
                details += f' | camera reported {camera_text} FPS'
            if show_info:
                draw_info_panel(annotated, [label, details], args.hud_opacity)
            if black_frames >= 5:
                cv2.putText(annotated, 'Black camera frames: check shutter / other camera apps',
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            if black_frames == 30:
                print('Warning: camera is returning black frames. Check shutter and competing camera apps.', flush=True)
            if not args.no_display:
                cv2.imshow(WINDOW, annotated)
                key = cv2.waitKey(0 if image else 1) & 0xFF
                if count == 1:
                    print('Preview started. Focus the preview window: Q/Esc quits; S saves a screenshot; H toggles info.', flush=True)
                if key in (ord('h'), ord('H')):
                    show_info = not show_info
                    if image:
                        continue
                if key in (ord('s'), ord('S')):
                    screenshots.append(save_frame(annotated, args.output_dir))
                    if image:
                        continue
                if key in (ord('q'), ord('Q'), 27):
                    reason = 'user_quit'
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    reason = 'window_closed'
                    break
            if args.max_frames and count >= args.max_frames:
                reason = 'frame_limit'
                break
            if image:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                if camera:
                    raise RuntimeError('Camera stopped returning frames. Check connection and camera permissions.')
                break
        if args.save_output and annotated is not None:
            screenshots.append(save_frame(annotated, args.output_dir))
        report = {
            'model': args.model.name,
            'cpu_threads': args.threads,
            'source_kind': 'camera' if camera else 'image' if image else 'video',
            'frames_processed': count,
            'inference_device': str(model.predictor.device),
            'frame_size': [int(annotated.shape[1]), int(annotated.shape[0])],
            'confidence_threshold': args.conf,
            'last_frame_counts': last_counts,
            'elapsed_seconds': round(time.perf_counter() - started, 3),
            'exit_reason': reason,
            'screenshots': screenshots,
            'display_enabled': not args.no_display,
            'camera_backend': capture.getBackendName() if camera else None,
            'requested_camera_fps': args.fps if camera else None,
            'driver_reported_camera_fps': camera_reported_fps,
            'observed_loop_fps': round((count - 1) / (previous_tick - first_tick), 2)
                                 if count > 1 and not image and previous_tick > first_tick else None,
            'last_frame_pixel_mean': round(float(frame.mean()), 2) if frame is not None else None,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    try:
        with camera_guard(args.source):
            return run(args)
    except KeyboardInterrupt:
        print('\nStopped by Ctrl+C.', flush=True)
        return 0
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
