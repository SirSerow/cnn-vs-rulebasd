# Third-party data notices

## Cubes on conveyor belt (colors)

- Dataset: [Cubes on conveyor belt (colors)](https://docs.edgeimpulse.com/datasets/image/cubes-on-conveyor-belt-colors)
- Creator and provider: Edge Impulse
- Download mirror: [Hugging Face dataset repository](https://huggingface.co/datasets/edgeimpulse/cubes-on-conveyor-belt)
- Pinned mirror revision: `e3d1c8b0c4872b70fcd77d86dec3bde7875e6054`
- Source dataset license: [BSD 3-Clause Clear](https://spdx.org/licenses/BSD-3-Clause-Clear.html)
- Mirror metadata license: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

The complete dataset is downloaded during deployment and is not redistributed
in this repository. One representative training image is included under
`docs/images/` so the selected input conditions can be reviewed directly from
the README. Preserve this notice and the applicable license terms when copying
or redistributing the example or dataset.

Suggested attribution:

> “Cubes on conveyor belt (colors)” dataset by Edge Impulse, downloaded from
> the Edge Impulse Hugging Face mirror at revision `e3d1c8b0c4872b70fcd77d86dec3bde7875e6054`.

## UA-DETRAC vehicle example

- Source benchmark: [UA-DETRAC](https://arxiv.org/abs/1511.04136)
- Example archive provider: [PaddleX](https://github.com/PaddlePaddle/PaddleX)
- Archive: `vehicle_coco_examples.tar`
- Use: public research benchmark; verify the upstream dataset terms before
  commercial redistribution.

The downloader fetches a 600-frame COCO-format example derived from UA-DETRAC.
The 100 annotated validation frames are used by this experiment. Dataset bytes,
model weights, and generated review videos are not committed to the repository.

Suggested attribution:

> L. Wen et al., “UA-DETRAC: A New Benchmark and Protocol for Multi-Object
> Detection and Tracking,” arXiv:1511.04136.
