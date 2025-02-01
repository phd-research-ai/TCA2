# Transferable Adversarial Face Attack with Text Controlled Attribute
Code for [Transferable Adversarial Face Attack with Text Controlled Attribute](https://arxiv.org/abs/2412.11735). 

Wenyun Li<sup>1,2</sup>, Zheng Zhang<sup>†1,2</sup>, Xiangyuan Lan<sup>†2.3</sup>, Dongmei Jiang<sup>2</sup>

<sup>1</sup>Harbin Institute of Technology, <sup>2</sup>Pengcheng Laboratory , <sup>3</sup>Pazhou Laboratory (Huangpu)


## Abstract
Traditional adversarial attacks typically produce adversarial examples under norm-constrained conditions, whereas unrestricted adversarial examples are free-form with semantically meaningful perturbations. Current unrestricted adversarial impersonation attacks exhibit limited control over adversarial face attributes and often suffer from low transferability. In this paper, we propose a novel Text Controlled Attribute Attack (TCA<sup>2</sup>) to generate photorealistic adversarial impersonation faces guided by natural language. Specifically, the category-level personal softmax vector is employed to precisely guide the impersonation attacks. Additionally, we propose both data and model augmentation strategies to achieve transferable attacks on unknown target models. Finally, a generative model, i.e, Style-GAN, is utilized to synthesize impersonated faces with desired attributes. Extensive experiments on two high-resolution face recognition datasets validate that our TCA<sup>2</sup> method can generate natural text-guided adversarial impersonation faces with high transferability. We also evaluate our method on real-world face recognition systems, i.e, Face++ and Aliyun, further demonstrating the practical potential of our approach. 

## Install



- Download checkpoints

  Pretrained LDM can be found [here](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.ckpt).

  We use IR152, IRSE50, FaceNet and MobileFace model checkpoints that provided by [AMT-GAN]([CGCL-codes/AMT-GAN: The official implementation of our CVPR 2022 paper "Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer". (github.com)](https://github.com/CGCL-codes/AMT-GAN)). The google drive link they provide is [here]([assets.zip - Google 云端硬盘](https://drive.google.com/file/d/1Vuek5-YTZlYGoeoqyM5DlvnaXMeii4O8/view)).

 



- Download datasets

  In our experiment we use KID-F and CelebA-HQ datasets for evaluation. Because we do not own the datasets,  you need to download them yourself. And you can refer to [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) for CelebA-HQ download.

  

  

## Usage

```bash
python targetedAttack.py
```

## Citation

```
@misc{li2024transferableadversarialfaceattack,
      title={Transferable Adversarial Face Attack with Text Controlled Attribute}, 
      author={Wenyun Li and Zheng Zhang and Xiangyuan Lan and Dongmei Jiang},
      year={2024},
      eprint={2412.11735},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.11735}, 
}
```

Meta-learning module will update later.

If you have any questions, please contact lee_wlving [AT] naver [DOT] com.