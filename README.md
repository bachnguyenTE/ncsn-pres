# NCSN Reproduction and Extension - Evaluation

This repository contains the code for the evaluation of the "Generative Modeling by Estimating Gradients of the Data Distribution" paper. It is based on the [official implementation](https://github.com/ermongroup/ncsn).

## Contributions

The main focus of this project was to reproduce the toy experiments and extend them to a more complex 2D distribution (Ring of Gaussians) to verify the claims about the effectiveness of Annealed Langevin Dynamics.

### Key Changes
- **Refactoring**: The `gaussians_mixture_example.ipynb` notebook was refactored to use reusable functions for training (`train_langevin_model`, `train_annealed_langevin_model`) and visualization (`visualize_langevin_model`, `visualize_annealed_langevin_model`). This makes the code cleaner and allows for easy experimentation with new datasets.
- **New Dataset**: Implemented a "Ring" dataset in `GMMDist` and `GMMDistAnneal` classes. This dataset consists of 8 Gaussians arranged in a circle, providing a harder challenge for mode mixing compared to the simple 2-Gaussian mixture.
- **Visualization**: Added GIF generation to visualize the score fields across different noise scales.

## How to Run Experiments

The primary experiments are contained in the Jupyter Notebook `gaussians_mixture_example.ipynb`.

### Prerequisites
Ensure you have the following dependencies installed:
- PyTorch
- Matplotlib
- Seaborn
- NumPy
- ImageIO
- TQDM

### Steps
1. Open `gaussians_mixture_example.ipynb` in VS Code or Jupyter Lab.
2. Run the cells sequentially.
3. The notebook will:
    - Train a standard Score Matching model on the 2-Gaussian dataset.
    - Visualize the results (density, samples, vector fields).
    - Train an NCSN (Annealed) model on the 2-Gaussian dataset.
    - Visualize the results and generate `annealed_scores_2gauss.gif`.
    - Repeat the process for the **Ring Dataset**, generating `annealed_scores_ring.gif`.

## Code Evaluation

### Copied vs. Contributed
- **Copied**: The core project structure (`main.py`, `runners/`, `models/`, `configs/`) is preserved from the original repository to maintain compatibility and context. The base logic for score matching loss (`dsm_score_estimation`) and Langevin dynamics (`langevin_dynamics`) was adapted from the original toy examples.
- **Contributed**:
    - The `GMMDist` and `GMMDistAnneal` classes were significantly updated to support multiple distribution types (`2gauss`, `ring`) dynamically.
    - The training loops were encapsulated into functions to support multiple experiments in a single notebook.
    - The Ring dataset experiment is a new addition.
    - GIF generation code was added.

### Reproduction of Results
- **2 Gaussians**: We successfully reproduced the results showing that standard Langevin dynamics can handle simple multimodal distributions reasonably well, but Annealed Langevin dynamics provides better coverage and mixing.
- **Ring Dataset**: We demonstrated that standard Langevin dynamics fails to mix between the 8 modes of the ring, often getting stuck in a subset of modes. In contrast, Annealed Langevin dynamics successfully samples from all modes, confirming the paper's claims about the necessity of noise annealing for complex multimodal distributions.

## Discussion

### Coding Experience
The experience involved understanding the core principles of Score-Based Generative Modeling. Refactoring the code into modular functions was crucial for running comparative experiments efficiently. Working with the score function vector fields provided intuitive insights into how the model learns to push samples towards high-density regions.

### Challenges
- **Mode Collapse**: One of the main challenges was observing mode collapse in the standard Langevin dynamics on the Ring dataset. This confirmed the theoretical limitations discussed in the paper.
- **Hyperparameter Tuning**: Getting the annealing schedule and step sizes right for the Ring dataset required some experimentation to ensure the model could bridge the gaps between modes.

---

# Generative Modeling by Estimating Gradients of the Data Distribution

This repo contains the official implementation for the NeurIPS 2019 paper 
[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600), 

by __Yang Song__ and __Stefano Ermon__. Stanford AI Lab.

**Note**: **The method has been greatly stabilized by the subsequent work
[Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011) ([code](https://github.com/ermongroup/ncsnv2)) and more recently extended by [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) ([code](https://github.com/yang-song/score_sde)). This codebase is therefore not recommended for new projects anymore.**

-------------------------------------------------------------------------------------
We describe a new method of generative modeling based on estimating the derivative of the log density 
function (_a.k.a._, Stein score) of the data distribution. We first perturb our training data by different Gaussian noise with progressively smaller variances. Next, we estimate the score function for each perturbed data distribution, by training a shared neural network named the _Noise Conditional Score Network (NCSN)_ using _score matching_. We can directly produce samples from our NSCN with _annealed Langevin dynamics_.


## Dependencies

* PyTorch

* PyYAML

* tqdm

* pillow

* tensorboardX

* seaborn


## Running Experiments

### Project Structure

`main.py` is the common gateway to all experiments. Type `python main.py --help` to get its usage description.

```bash
usage: main.py [-h] [--runner RUNNER] [--config CONFIG] [--seed SEED]
               [--run RUN] [--doc DOC] [--comment COMMENT] [--verbose VERBOSE]
               [--test] [--resume_training] [-o IMAGE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --runner RUNNER       The runner to execute
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --run RUN             Path for saving running related data.
  --doc DOC             A string for documentation purpose
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  --test                Whether to test the model
  --resume_training     Whether to resume training
  -o IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The directory of image outputs
```

There are four runner classes.

* `AnnealRunner` The main runner class for experiments related to NCSN and annealed Langevin dynamics.
* `BaselineRunner` Compared to `AnnealRunner`, this one does not anneal the noise. Instead, it uses a single fixed noise variance.
* `ScoreNetRunner` This is the runner class for reproducing the experiment of Figure 1 (Middle, Right)
* `ToyRunner` This is the runner class for reproducing the experiment of Figure 2 and Figure 3.

Configuration files are stored in  `configs/`. For example, the configuration file of `AnnealRunner` is `configs/anneal.yml`. Log files are commonly stored in `run/logs/doc_name`, and tensorboard files are in `run/tensorboard/doc_name`. Here `doc_name` is the value fed to option `--doc`.

### Training

The usage of `main.py` is quite self-evident. For example, we can train an NCSN by running

```bash
python main.py --runner AnnealRunner --config anneal.yml --doc cifar10
```

Then the model will be trained according to the configuration files in `configs/anneal.yml`. The log files will be stored in `run/logs/cifar10`, and the tensorboard logs are in `run/tensorboard/cifar10`.

### Sampling

Suppose the log files are stored in `run/logs/cifar10`. We can produce samples to folder `samples` by running

```bash
python main.py --runner AnnealRunner --test -o samples
```

### Checkpoints

We provide pretrained checkpoints [run.zip](https://drive.google.com/file/d/1BF2mwFv5IRCGaQbEWTbLlAOWEkNzMe5O/view?usp=sharing). Extract the file to the root folder. You should be able to produce samples like the following using this checkpoint.

| Dataset | Sampling procedure |
| :------------ | :-------------------------: |
| MNIST |  ![MNIST](assets/mnist_large.gif)|
| CelebA |  ![Celeba](assets/celeba_large.gif)|
|CIFAR-10 |  ![CIFAR10](assets/cifar10_large.gif)|

### Evaluation
Please refer to Appendix B.2 of our paper for details on hyperparameters and model selection. When computing inception and FID scores, we first generate images from our model, and use the [official code from OpenAI](https://github.com/openai/improved-gan/tree/master/inception_score) and the [original code from TTUR authors](https://github.com/bioinf-jku/TTUR) to obtain the scores.


## References

Large parts of the code are derived from [this Github repo](https://github.com/ermongroup/sliced_score_matching) (the official implementation of the [sliced score matching paper](https://arxiv.org/abs/1905.07088))

If you find the code / idea inspiring for your research, please consider citing the following

```bib
@inproceedings{song2019generative,
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11895--11907},
  year={2019}
}
```

and / or

```bib
@inproceedings{song2019sliced,
  author    = {Yang Song and
               Sahaj Garg and
               Jiaxin Shi and
               Stefano Ermon},
  title     = {Sliced Score Matching: {A} Scalable Approach to Density and Score
               Estimation},
  booktitle = {Proceedings of the Thirty-Fifth Conference on Uncertainty in Artificial
               Intelligence, {UAI} 2019, Tel Aviv, Israel, July 22-25, 2019},
  pages     = {204},
  year      = {2019},
  url       = {http://auai.org/uai2019/proceedings/papers/204.pdf},
}
```

