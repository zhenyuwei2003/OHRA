# One Hand to Rule Them All

Official Code Repository for **One Hand to Rule Them All: Canonical Representations for Unified Dexterous Manipulation**.

[Zhenyu Wei](https://zhenyuwei2003.github.io/), Yunchao Yao, [Mingyu Ding](https://dingmyu.github.io/)

University of North Carolina at Chapel Hill

<p align="center">
    <a href='#'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://zhenyuwei2003.github.io/OHRA/'>
      <img src='https://img.shields.io/badge/Project-Page-66C0FF?style=plastic&logo=Google%20chrome&logoColor=66C0FF' alt='Project Page'>
    </a>
</p>
<div align="center">
  <img src="Teaser.png" alt="Teaser" width="95%">
</div>

We introduce a canonical hand representation that unifies diverse dexterous hands into a shared parameter space and canonical URDF format, serving as a condition for cross-embodiment policy learning. It enables dexterous grasping and zero-shot generalization to novel hand morphologies, highlighting its potential for a wide range of dexterous manipulation tasks.

----------------

## Update (2026-03-22):
The first stage of the code release is now available!
This release includes the canonical hand representation, along with URDF parsing and rendering scripts.
We will continue to release additional components in the coming weeks, including training pipelines and simulation environments — stay tuned!

## TODO:
- [ ] Extended canonical hand parameters related assets and codes.
- [ ] Training pipeline for multiple experiments.
- [ ] Isaac Gym simulation environment scripts.
- [ ] Additional utility and visualization tools.
- [ ] Comprehensive documentation and tutorials.

## Prerequisites:
Although the current core has minimal environment requirements, we recommend using Python 3.8, as Isaac Gym is only compatible with Python ≤ 3.8.

## Get Started:
```bash
conda create -n ohra python=3.8 -y
conda activate ohra
pip install torch scipy pytorch_kinematics fpsample
```
⚠️ Note:
The environment setup has not been tested yet, but dependency issues are expected to be minimal.
A more detailed and robust setup guide will be provided in future updates.

## Usage:

### Create the new canonical hand:
1. Add the hand URDF file to `assets/robot_urdf/`.
2. Create the hand meta information json file to `assets/meta_infos/` (refer to other files).
3. Run `utils/urdf_parser.py` to get the canonical hand parameters. Though many cases are considered, due to the various URDF format and design, some manual adjustments may be needed for certain hand models.
4. Run `utils/urdf_render.py` to generate the canonical URDF file.
5. Run `visualization/vis_compare.py` to visualize the difference between the original hand model and the canonical hand model. If you find there exists discrepancies, back to Step 3 to manually adjust the parameters and repeat the process until you get a satisfactory result.

⚠️ Note:
More detailed explainations and tutorials will be provided in future updates, including the meaning and design details of the canonical hand parameters, as well as the manual adjustment guidance.

## Repository Structure

```bash
OHRA
├── assets/
│   ├── canonical/  # Canonical hand template, parameters and URDFs
│   ├── canonical_extended/  # TODO: only limited content for now, more to be added in the future
│   ├── meta_infos/  # Hand meta information for usage
│   └── robot_urdf/  # Original URDFs of various robot hands
├── utils  # Various utility scripts, including hand model, URDF parsing, and rendering
└── visualization  # Visualization scripts
```

## Citation
If you find our codes or models useful in your work, please cite [our paper](https://arxiv.org/abs/2602.16712):

```
@article{wei2026one,
    title={One Hand to Rule Them All: Canonical Representations for Unified Dexterous Manipulation},
    author={Wei, Zhenyu and Yao, Yunchao and Ding, Mingyu},
    journal={arXiv preprint arXiv:2602.16712},
    year={2026}
}
```

## Contact

If you have any questions, feel free to contact me through email ([wzhenyu@cs.unc.edu](mailto:wzhenyu@cs.unc.edu))!
