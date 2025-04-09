import os
import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# OED imports
from lib.OED.models import build_model
from lib.OED.engine import evaluate_dsgg
from lib.OED.datasets import build_dataset

# TGN imports
from lib.tgn.model.tgn import TGN
from lib.tgn.utils.utils import get_neighbor_finder
from lib.tgn.evaluation.evaluation import eval_node_classification

class VideoClassificationPipeline:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize OED model for scene graph generation
        self.oed_model = build_model(args)
        self.oed_model.to(self.device)

        # Initialize TGN model for graph classification
        self.tgn_model = TGN(
            node_dim=args.node_dim,
            time_dim=args.time_dim,
            n_layers=args.n_layer,
            n_heads=args.n_head,
            dropout=args.drop_out,
            use_memory=args.use_memory,
            embedding_module_type=args.embedding_module,
            message_function_type=args.message_function,
            aggregator_type=args.aggregator
        )
        self.tgn_model.to(self.device)

    def process_video(self, video_path):
        """Process a video through both stages of the pipeline"""
        # Stage 1: Generate scene graphs using OED
        scene_graphs = self.generate_scene_graphs(video_path)

        # Stage 2: Classify scene graphs using TGN
        predictions = self.classify_scene_graphs(scene_graphs)

        return predictions

    def generate_scene_graphs(self, video_path):
        """Generate scene graphs from video frames using OED"""
        # Load video frames
        dataset = build_dataset('video', video_path, self.args)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size,
            num_workers=4, collate_fn=dataset.collate_fn
        )

        scene_graphs = []
        self.oed_model.eval()

        with torch.no_grad():
            for samples in tqdm(dataloader, desc="Generating scene graphs"):
                samples = samples.to(self.device)
                outputs = self.oed_model(samples)
                scene_graphs.extend(self.process_oed_outputs(outputs))

        return scene_graphs

    def classify_scene_graphs(self, scene_graphs):
        """Classify scene graphs using TGN"""
        self.tgn_model.eval()
        predictions = []

        with torch.no_grad():
            for graph in tqdm(scene_graphs, desc="Classifying scene graphs"):
                # Convert scene graph to TGN format
                tgn_input = self.convert_to_tgn_format(graph)

                # Get prediction
                pred = self.tgn_model(tgn_input)
                predictions.append(pred)

        return predictions

    def process_oed_outputs(self, outputs):
        """Process OED outputs into scene graph format"""
        # Implementation depends on OED output format
        # This is a placeholder - you'll need to implement based on actual OED output structure
        return outputs

    def convert_to_tgn_format(self, scene_graph):
        """Convert scene graph to TGN input format"""
        # Implementation depends on scene graph format
        # This is a placeholder - you'll need to implement based on actual graph structure
        return scene_graph

def get_args_parser():
    parser = argparse.ArgumentParser('Video Classification Pipeline', add_help=False)

    # OED arguments
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers_hopd', default=3, type=int)
    parser.add_argument('--dec_layers_interaction', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)

    # TGN arguments
    parser.add_argument('--node_dim', type=int, default=100)
    parser.add_argument('--time_dim', type=int, default=100)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--use_memory', action='store_true')
    parser.add_argument('--embedding_module', type=str, default="graph_attention")
    parser.add_argument('--message_function', type=str, default="identity")
    parser.add_argument('--aggregator', type=str, default="last")

    # General arguments
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--video_path', type=str, required=True)

    return parser

def main(args):
    pipeline = VideoClassificationPipeline(args)
    predictions = pipeline.process_video(args.video_path)
    print(f"Classification results: {predictions}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Video Classification Pipeline', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
