import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time

# Add OED to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib/OED'))

# OED imports
from models import build_model
from engine import evaluate_dsgg
from datasets import build_dataset
from datasets.coco_video_parser import CocoVID
from util.misc import NestedTensor

def get_args_parser():
    parser = argparse.ArgumentParser('OED Scene Graph Generation', add_help=False)

    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers_hopd', default=3, type=int,
                        help="Number of hopd decoding layers in the transformer")
    parser.add_argument('--dec_layers_interaction', default=3, type=int,
                        help="Number of interaction decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Video processing
    parser.add_argument('--video_path', type=str, required=True,
                        help="Path to the input video file")
    parser.add_argument('--output_file', type=str, default='output.txt',
                        help="Path to save the scene graph output")
    parser.add_argument('--frame_interval', type=int, default=1,
                        help="Process every nth frame")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Batch size for processing frames")
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device to use (cuda or cpu)")

    # Model weights
    parser.add_argument('--weights_path', type=str, required=True,
                        help="Path to the pretrained OED model weights")

    return parser

def extract_frames(video_path, frame_interval=1):
    """Extract frames from video at specified intervals"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1

    cap.release()
    return frames

def preprocess_frame(frame):
    """Preprocess frame for OED model"""
    # Resize to standard size (adjust as needed)
    frame = cv2.resize(frame, (800, 600))

    # Normalize
    frame = frame.astype(np.float32) / 255.0

    # Convert to tensor and add batch dimension
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)

    return frame_tensor

def create_dummy_annotation(frames):
    """Create a dummy annotation file for the frames"""
    annotation = {
        'images': [],
        'annotations': [],
        'videos': [{'id': 1, 'name': 'video'}]
    }

    for i, _ in enumerate(frames):
        annotation['images'].append({
            'id': i,
            'file_name': f'frame_{i}.jpg',
            'video_id': 1,
            'frame_id': i
        })

    return annotation

def process_video(args):
    """Process video and generate scene graphs"""
    # Extract frames
    print(f"Extracting frames from {args.video_path}...")
    frames = extract_frames(args.video_path, args.frame_interval)
    print(f"Extracted {len(frames)} frames")

    # Create dummy annotation
    annotation = create_dummy_annotation(frames)

    # Save temporary annotation file
    temp_annotation_path = 'temp_annotation.json'
    with open(temp_annotation_path, 'w') as f:
        json.dump(annotation, f)

    # Initialize dataset
    dataset = build_dataset('video', temp_annotation_path, args)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=0, collate_fn=dataset.collate_fn
    )

    # Initialize model
    device = torch.device(args.device)
    model = build_model(args)
    model.to(device)

    # Load pretrained weights
    checkpoint = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Process frames and generate scene graphs
    scene_graphs = []

    with torch.no_grad():
        for samples, _ in tqdm(dataloader, desc="Generating scene graphs"):
            samples = samples.to(device)
            outputs = model(samples)

            # Process outputs to extract scene graphs
            for output in outputs:
                scene_graph = process_model_output(output)
                scene_graphs.append(scene_graph)

    # Save scene graphs to output file
    with open(args.output_file, 'w') as f:
        for i, graph in enumerate(scene_graphs):
            f.write(f"Frame {i * args.frame_interval}:\n")
            f.write(json.dumps(graph, indent=2))
            f.write("\n\n")

    # Clean up
    os.remove(temp_annotation_path)

    print(f"Scene graphs saved to {args.output_file}")
    return scene_graphs

def process_model_output(output):
    """Process model output to extract scene graph"""
    # This function needs to be customized based on the actual output format of OED
    # For now, we'll extract basic information

    scene_graph = {
        'objects': [],
        'relationships': []
    }

    # Extract objects
    if 'pred_logits' in output and 'pred_boxes' in output:
        scores = output['pred_logits'].softmax(-1)[0, :, :-1].max(-1)[0]
        boxes = output['pred_boxes'][0]

        # Filter by confidence threshold
        keep = scores > 0.5

        for i, (score, box) in enumerate(zip(scores[keep], boxes[keep])):
            scene_graph['objects'].append({
                'id': i,
                'score': float(score),
                'box': box.tolist()
            })

    # Extract relationships (if available)
    if 'pred_rel_logits' in output and 'pred_rel_pairs' in output:
        rel_scores = output['pred_rel_logits'].softmax(-1)[0, :, :-1].max(-1)[0]
        rel_pairs = output['pred_rel_pairs'][0]

        # Filter by confidence threshold
        keep = rel_scores > 0.5

        for i, (score, pair) in enumerate(zip(rel_scores[keep], rel_pairs[keep])):
            scene_graph['relationships'].append({
                'id': i,
                'score': float(score),
                'subject': int(pair[0]),
                'object': int(pair[1])
            })

    return scene_graph

def main(args):
    process_video(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OED Scene Graph Generation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
