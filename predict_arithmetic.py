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

def perplexity_score(text, lm_model, lm_tokenizer, device):
    encodings = lm_tokenizer(f'{lm_tokenizer.bos_token + text}', return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    outputs = lm_model(input_ids, labels=target_ids)
    log_likelihood = outputs[0]
    ll = log_likelihood.item()

    return ll

class Predictor(cog.Predictor):
    def setup(self):
        self.args = get_args()
        self.args.reset_context_delta = True
        self.text_generator = CLIPTextGenerator(**vars(self.args))

    @cog.input(
        "image1",
        type=Path,
        help="Final result will be: image1 + (image2 - image3)"
    )
    @cog.input(
        "image2",
        type=Path,
        help="Final result will be: image1 + (image2 - image3)"
    )
    @cog.input(
        "image3",
        type=Path,
        help="Final result will be: image1 + (image2 - image3)"
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
        default=3, min=1, max=10,
        help="Number of beams to use",
    )
    @cog.input(
        "end_factors",
        type=float,
        default=1.06, min=1.0, max=1.10,
        help="Higher value for shorter captions",
    )
    @cog.input(
        "max_seq_lengths",
        type=int,
        default=3, min=1, max=20,
        help="Maximum number of tokens to generate",
    )
    @cog.input(
        "ce_loss_scale",
        type=float,
        default=0.2, min=0.0, max=0.6,
        help="Scale of cross-entropy loss with un-shifted language model",
    )
    def predict(self, image1, image2, image3, cond_text, beam_size, end_factors, max_seq_lengths, ce_loss_scale):
        self.args.cond_text = cond_text
        self.text_generator.end_factor = end_factors
        self.text_generator.target_seq_length = max_seq_lengths
        self.text_generator.ce_scale = ce_loss_scale
        self.text_generator.fusion_factor = 0.95
        self.text_generator.grad_norm_factor = 0.95

        image_features = self.text_generator.get_combined_feature([str(image1), str(image2), str(image3)], [], [1, 1, -1], None)
        captions = self.text_generator.run(image_features, self.args.cond_text, beam_size=beam_size)

        # CLIP SCORE
        encoded_captions = [self.text_generator.clip.encode_text(clip.tokenize(c).to(self.text_generator.device))
                            for c in captions]
        encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
        best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

        # Perplexity SCORE
        ppl_scores = [perplexity_score(x, self.text_generator.lm_model, self.text_generator.lm_tokenizer, self.text_generator.device) for x in captions]
        best_ppl_index = torch.tensor(ppl_scores).argmin().item()

        best_clip_caption = self.args.cond_text + captions[best_clip_idx]
        best_mixed = self.args.cond_text + captions[0]
        best_PPL = self.args.cond_text + captions[best_ppl_index]

        final = f'Best CLIP: {best_clip_caption} \nBest fluency: {best_PPL} \nBest mixed: {best_mixed}'

        return final
        # return self.args.cond_text + captions[best_clip_idx]


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
    parser.add_argument("--grad_norm_factor", type=float, default=0.95)
    parser.add_argument("--fusion_factor", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    args = parser.parse_args('')
    return args
