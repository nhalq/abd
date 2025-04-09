import torch
import cv2
import argparse

from lib.oed.models import build_model as build_oed_model
from lib.oed.main import get_args_parser as get_oed_args_parser


def get_args_parser():
    eod_parser = get_oed_args_parser()
    parser = argparse.ArgumentParser(
        'OED Scene Graph Generation', add_help=False, parents=[eod_parser])
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)

    return parser


def create_frames(video_path: str):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Can not open: {video_path}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 640))
        yield frame

    capture.release()


def main(args) -> None:
    frames = list(create_frames(args.video_path))[:24]
    samples = torch.tensor(frames, dtype=torch.float).to(args.device)

    oed, _, _ = build_oed_model(args)
    oed(samples)

    pass


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # OED args
    args.dataset_file = 'ag_multi'
    args.dsgg_task = 'sgdet'

    main(args)
