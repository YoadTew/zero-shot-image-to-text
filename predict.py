import os
import tempfile
import sys
sys.path.append('CLIP')
from pathlib import Path
import cog
import argparse
import torch
import clip
from model.ZeroCLIP import CLIPTextGenerator


class Predictor(cog.Predictor):
    def setup(self):
        self.args = get_args()
        self.args.reset_context_delta = True
        self.text_generator = CLIPTextGenerator(**vars(self.args))

    @cog.input(
        "image",
        type=Path,
        help="input image"
    )
    @cog.input(
        "cond_text",
        type=str,
        default='Image of a',
        help="conditional text",
    )
    @cog.input(
        "beam_size",
        type=int,
        default=5, min=1, max=10,
        help="Number of beams to use",
    )
    @cog.input(
        "end_factor",
        type=float,
        default=1.01, min=1.0, max=1.10,
        help="Higher value for shorter captions",
    )
    def predict(self, image, cond_text, beam_size, end_factor):
        self.args.cond_text = cond_text
        self.text_generator.end_factor = end_factor

        image_features = self.text_generator.get_img_feature([str(image)], None)
        captions = self.text_generator.run(image_features, self.args.cond_text, beam_size=beam_size)
        encoded_captions = [self.text_generator.clip.encode_text(clip.tokenize(c).to(self.text_generator.device))
                            for c in captions]
        encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
        best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

        return self.args.cond_text + captions[best_clip_idx]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--reset_context_delta", action="store_true",
                        help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    args = parser.parse_args('')
    return args
