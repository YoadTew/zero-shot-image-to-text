# Pytorch Implementation of Zero-Shot Image-to-Text Generation for Visual-Semantic Arithmetic
[[Paper]](https://arxiv.org/abs/2111.14447) [[Notebook]](https://www.kaggle.com/yoavstau/zero-shot-image-to-text/notebook) <a href="https://replicate.com/yoadtew/zero-shot-image-to-text"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=blue"></a> 

⭐ ***New:*** Run captioning configuration it in the [browser](https://replicate.com/yoadtew/zero-shot-image-to-text) using replicate.ai UI.

## Approach
![](git_images/Architecture.jpg)

## Example of capabilities
![](git_images/teaser.jpg)

## Example of Visual-Semantic Arithmetic
![](git_images/relations.jpg)

## Usage

### To run captioning on a single image:

```bash
$ python run.py 
--reset_context_delta
--caption_img_path "example_images/captions/COCO_val2014_000000097017.jpg"
```

### To run model on visual arithmetic:

```bash
$ python run.py 
--reset_context_delta
--end_factor 1.06
--fusion_factor 0.95
--grad_norm_factor 0.95
--run_type arithmetics
--arithmetics_imgs "example_images/arithmetics/woman2.jpg" "example_images/arithmetics/king2.jpg" "example_images/arithmetics/man2.jpg"
--arithmetics_weights 1 1 -1
```

### To run model on real world knowledge:

```bash
$ python run.py
--reset_context_delta --cond_text "Image of" 
--end_factor 1.04 
--caption_img_path "example_images/real_world/simpsons.jpg"
```

### To run model on OCR:

```bash
$ python run.py
--reset_context_delta --cond_text "Image of text that says" 
--end_factor 1.04 
--caption_img_path "example_images/OCR/welcome_sign.jpg"
```

## Citation
Please cite our work if you use it in your research:
```
@article{tewel2021zero,
  title={Zero-Shot Image-to-Text Generation for Visual-Semantic Arithmetic},
  author={Tewel, Yoad and Shalev, Yoav and Schwartz, Idan and Wolf, Lior},
  journal={arXiv preprint arXiv:2111.14447},
  year={2021}
}
```
