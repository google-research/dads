# Dynamics-Aware Discovery of Skills (DADS)
This repository is the open-source implementation of Dynamics-Aware Unsupervised Discovery of Skills ([project page][website], [arXiv][paper]). We propose an skill-discovery method which can learn skills for different agents without any rewards, while simultaneously learning dynamics model for the skills which can be leveraged for model-based control on the downstream task. This work was published in International Conference of Learning Representations ([ICLR][iclr]), 2020.

We have also included an improved off-policy version of DADS, coined off-DADS. The details have been released in [Emergent Real-World Robotic Skills via Unsupervised Off-Policy Reinforcement Learning](rss_arxiv).

In case of problems, contact Archit Sharma.

## Table of Contents

* [Setup](#setup)
* [Usage](#usage)
* [Citation](#citation)
* [Disclaimer](#disclaimer)

## Setup

#### (1) Setup MuJoCo
Download and setup [mujoco][mujoco] in `~/.mujoco`. Set the `LD_LIBRARY_PATH` in your `~/.bashrc`:
```
LD_LIBRARY_PATH='~/.mujoco/mjpro150/bin':$LD_LIBRARY_PATH
```

#### (2) Setup environment
Clone the repository and setup up the [conda][conda] environment to run DADS code:
```
cd <path_to_dads>
conda env create -f env.yml
conda activate dads-env
```

## Usage
We give a high-level explanation of how to use the code. More details pertaining to hyperparameters can be found in the the `configs/template_config.txt`, `dads_off.py` and the Appendix A of [paper][paper].

Every training run will require an experimental logging directory and a configuration file, which can be created started from the `configs/template_config.txt`. There are two phases: (a) Training where the new skills are learnt along with their skill-dynamics models and (b) evaluation where the learnt skills are evaluated on the task associated with the environment.

For training, ensure `--run_train=1` is set in the configuration file. For on-policy optimization, set `--clear_buffer_every_iter=1` and ensure the replay buffer size is bigger than the number of steps collected in every iteration. For off-policy optimization (details yet to be released), set `--clear_buffer_every_iter=0`. Set the environment name (ensure the environment is listed in `get_environment()` in `dads_off.py`). To change the observation for skill-dynamics (for example to learn in x-y space), set `--reduced_observation` and correspondingly configure `process_observation()` in `dads_off.py`. The skill space can be configured to be discrete or continuous. The optimization parameters can be tweaked, and some basic values have been set in (more details in the [paper][paper]). 

For evaluation, ensure `--run_eval=1` and the experimental directory points to the same directory in which the training happened. Set `--num_evals` if you want to record videos of randomly sampled skills from the prior distribution. After that, the script will use the learned models to execute MPC on the latent space to optimize for the task-reward. By default, the code will call `get_environment()` to load `FLAGS.environment + '_goal'`, and will go through the list of goal-coordinates specified in the eval section of the script.

We have provided the configuration files in `configs/` to reproduce results from the experiments in the [paper][paper]. Goal evaluation is currently only setup for MuJoCo Ant environement. The goal distribution can be changed in `dads_off.py` in evaluation part of the script.

```
cd <path_to_dads>
python unsupervised_skill_learning/dads_off.py --logdir=<path_for_experiment_logs> --flagfile=configs/<config_name>.txt
```

The specified experimental log directory will contain the tensorboard files, the saved checkpoints and the skill-evaluation videos.

## Citation
To cite [Dynamics-Aware Unsupervised Discovery of Skills](paper):
```
@article{sharma2019dynamics,
  title={Dynamics-aware unsupervised discovery of skills},
  author={Sharma, Archit and Gu, Shixiang and Levine, Sergey and Kumar, Vikash and Hausman, Karol},
  journal={arXiv preprint arXiv:1907.01657},
  year={2019}
}
```
To cite off-DADS and [Emergent Real-World Robotic Skills via Unsupervised Off-Policy Reinforcement Learning](rss_arxiv):
```
@article{sharma2020emergent,
    title={Emergent Real-World Robotic Skills via Unsupervised Off-Policy Reinforcement Learning},
    author={Sharma, Archit and Ahn, Michael and Levine, Sergey and Kumar, Vikash and Hausman, Karol and Gu, Shixiang},
    journal={arXiv preprint arXiv:2004.12974},
    year={2020}
}
```
## Disclaimer
This is not an officially supported Google product.

[website]: https://sites.google.com/corp/view/dads-skill 
[paper]: https://arxiv.org/abs/1907.01657
[iclr]: https://openreview.net/forum?id=HJgLZR4KvH
[mujoco]: http://www.mujoco.org/
[conda]: https://docs.conda.io/en/latest/miniconda.html
[rss_arxiv]: https://arxiv.org/abs/2004.12974
