"""
3D Human Pose Estimation Pipeline
==================================

Three modes:

  train  — Train the 2D→3D lifting model on Human3.6M (or synthetic data)
  video  — Full pipeline on a video file: MediaPipe 2D detect → lift → visualize
  excel  — Load pre-extracted 2D keypoints from pilot_study.xlsx → lift → visualize

Examples:
  # Train FCN model (with Human3.6M data)
  python main.py --mode train --model fcn --data-path data/human36m/ --epochs 50

  # Train with synthetic data (no dataset required)
  python main.py --mode train --model fcn --epochs 10

  # Run on a video
  python main.py --mode video --input video.mp4 --checkpoint checkpoints/fcn_best.pth

  # Run on pilot study Excel file (all participants)
  python main.py --mode excel --input raw_data/pilot_study.xlsx --checkpoint checkpoints/fcn_best.pth

  # Run on pilot study, filtered to a single participant
  python main.py --mode excel --input raw_data/pilot_study.xlsx --checkpoint checkpoints/fcn_best.pth --participant P1
"""

import argparse
import numpy as np


def run_train(args):
    from lifting.train import train
    train(
        model_type=args.model,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_path=args.save_path,
    )


def run_video(args):
    import os
    import cv2
    from detector.pose_detector import PoseDetector

    print(f"Processing video: {args.input}")

    # Auto-generate output path if not provided
    if not args.output:
        os.makedirs('outputs', exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join('outputs', f'{stem}_annotated.mp4')
        print(f"No --output given, saving to: {args.output}")

    _run_video_render(args, cv2, PoseDetector)


def _run_video_render(args, cv2, PoseDetector):
    """
    Frame-by-frame render: writes the original video with 2D skeleton overlay to args.output.
    """
    # COCO skeleton for 2D overlay
    SKELETON_2D = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    detector = PoseDetector()

    detected = 0

    print(f"Rendering {total} frames → {args.output}")
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        kp = detector.detect(frame)

        if kp is not None:
            detected += 1
            h, w = frame.shape[:2]

            # 2D skeleton overlay — thicker lines for visibility
            for a, b in SKELETON_2D:
                x1, y1 = int(kp[a, 0] * w), int(kp[a, 1] * h)
                x2, y2 = int(kp[b, 0] * w), int(kp[b, 1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 230, 0), 3)
            for jx, jy in kp:
                cv2.circle(frame, (int(jx * w), int(jy * h)), 5, (0, 60, 255), -1)

        writer.write(frame)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total} frames  ({detected} poses detected)")

    cap.release()
    writer.release()
    detector.close()
    print(f"Done. Detected poses in {detected}/{total} frames.")
    print(f"Saved: {args.output}")


def run_excel(args):
    from data.loader import load_pilot_study
    from lifting.predict import load_model, predict
    from visualizer.visualize import animate_sequence, plot_pose_3d

    print(f"Loading pilot study: {args.input}")
    kps_2d, metadata = load_pilot_study(args.input)

    if args.participant:
        mask = metadata['Dataset'].str.startswith(args.participant)
        kps_2d = kps_2d[mask.values]
        print(f"Filtered to {len(kps_2d)} frames for participant '{args.participant}'")
    else:
        print(f"Loaded {len(kps_2d)} frames")

    if len(kps_2d) == 0:
        print("No data after filtering.")
        return

    model, ckpt, device = load_model(args.checkpoint)
    kps_3d = predict(model, kps_2d, ckpt, device)   # (N, 17, 3)

    title = f'3D Pose — {args.participant or "all participants"}'
    if len(kps_3d) == 1:
        plot_pose_3d(kps_3d[0], title=title, save_path=args.save_fig)
    else:
        anim = animate_sequence(kps_3d, title=title, save_path=args.save_fig)  # noqa: F841


def main():
    parser = argparse.ArgumentParser(
        description='3D Human Pose Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode', choices=['train', 'video', 'excel'], required=True,
                        help='Pipeline mode')

    # Shared
    parser.add_argument('--input', type=str,
                        help='Path to input video (.mp4 etc.) or Excel file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fcn_best.pth',
                        help='Path to model checkpoint for inference')
    parser.add_argument('--save-fig', type=str, default=None,
                        help='Save visualisation to file instead of displaying it')
    parser.add_argument('--output', type=str, default=None,
                        help='(video mode) Save side-by-side annotated video to this path')

    # Excel-specific
    parser.add_argument('--participant', type=str, default=None,
                        help='Filter pilot study by participant prefix, e.g. "P1"')

    # Training
    parser.add_argument('--model', choices=['fcn', 'gcn'], default='fcn',
                        help='Lifting model architecture')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to Human3.6M preprocessed data directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--save-path', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')

    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'video':
        if not args.input:
            parser.error('--input is required for video mode')
        run_video(args)
    elif args.mode == 'excel':
        if not args.input:
            parser.error('--input is required for excel mode')
        run_excel(args)


if __name__ == '__main__':
    main()
