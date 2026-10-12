# VRMDiff: Text-Guided Video Referring Matting Generation of Diffusion

[![arXiv](https://img.shields.io/badge/arXiv-2503.10678-b31b1b.svg)](https://arxiv.org/abs/2503.10678)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VRM--10K-yellow)](https://huggingface.co/datasets/xsourse/VRM-10K)

**Paper:** [arXiv:2503.10678](https://arxiv.org/abs/2503.10678)  |  **Dataset:** [VRM-10K on Hugging Face](https://huggingface.co/datasets/xsourse/VRM-10K)

## News

- **[2026-10]** The VRM-10K dataset is released on [Hugging Face](https://huggingface.co/datasets/xsourse/VRM-10K).
- **[2025-03]** The paper is available on [arXiv](https://arxiv.org/abs/2503.10678).

## Overview

We propose **video referring matting**: given a video and a text caption, predict the alpha matte of the instance the caption refers to. VRMDiff treats this as a video generation problem and uses the text-to-video alignment of video diffusion models. We also introduce a Latent-Constructive loss to better separate instances, and release **VRM-10K**, a dataset of 10,000 videos with captions and instance-level alpha mattes.

## VRM-10K Dataset

Each sample is a video with one or more people. Every person has a text caption that refers to them and a per-frame alpha matte.

| Split | Videos |
|-------|--------|
| train | 9,000  |
| test  | 1,000  |

- Each video has 120 frames at 30 fps.
- Alpha mattes are 8-bit grayscale PNGs at the video's resolution, one per frame per person.
- Total size is about 179 GB.

See the [dataset card](https://huggingface.co/datasets/xsourse/VRM-10K) for the full field list.

### Usage

```python
from datasets import load_dataset, Video

ds = load_dataset("xsourse/VRM-10K", split="test", streaming=True)
ds = ds.cast_column("video", Video(decode=False))  # raw MP4 bytes; drop this line to decode with torchcodec

sample = next(iter(ds))
mp4_bytes = sample["video"]["bytes"]
for obj in sample["objects"]:
    caption = obj["caption"]
    mattes = obj["mattes"]  # list of 120 PIL images
```

To download the whole dataset locally:

```bash
hf download xsourse/VRM-10K --repo-type dataset --local-dir VRM-10K
```

### Source data and license

Foreground videos and alpha mattes come from [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/) (footage provided by Storyblocks). Background videos come from [DVM](https://github.com/nowsyn/DVM). Captions were generated with [Tarsier-34B](https://github.com/bytedance/tarsier). Please also cite these works when you use the dataset.

VRM-10K is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.

## Citation

If you find this work or the dataset useful, please cite:

```bibtex
@article{yang2025vrmdiff,
  title   = {VRMDiff: Text-Guided Video Referring Matting Generation of Diffusion},
  author  = {Yang, Lehan and Song, Jincen and Wang, Tianlong and Qi, Daiqing and Shi, Weili and Liu, Yuheng and Li, Sheng},
  journal = {arXiv preprint arXiv:2503.10678},
  year    = {2025}
}
```
