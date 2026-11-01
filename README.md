# Hands-on Simulation-Based Inference with BayesFlow

These tutorials introduce amortized simulation-based inference (SBI) with [BayesFlow 2.1](https://bayesflow.org/) through a diffusion model for conflict tasks and a biophysical neural-circuit model.

## SBI in brief

A simulator maps parameters $\theta$ to observations $x$. This is the **forward problem**. SBI addresses the **inverse problem**: determining which parameters could have produced an observation.

![Forward and inverse problems](https://bayesops.com/images/index/forward-inverse.svg)

Neural SBI trains on simulated $(\theta, x)$ pairs. Once trained, an amortized model can estimate $p(\theta \mid x)$ for many observations without retraining.

## Tutorials

| Tutorial | Topic |
| --- | --- |
| [Diffusion Model for Conflict Tasks](dmc/dmc-bayesflow.ipynb) | Infer cognitive-model parameters from choices and response times. |
| [Human Neocortical Neurosolver](hnn/HNN_Integration.ipynb) | Infer parameters of a biophysical neural circuit. |

Both tutorials load bundled raw training simulations. The HNN posterior-predictive cells require macOS, Linux, or WSL because NEURON is unavailable on native Windows.

## Setup

### 1. Install Visual Studio Code

Install [Visual Studio Code](https://code.visualstudio.com/) and its [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions.

### 2. Install uv and Python

Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/) or one of these commands.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS or Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal and install Python 3.12:

```sh
uv python install 3.12
```

### 3. Get the workshop files

Download and extract the repository ZIP from GitHub (**Code > Download ZIP**) or clone it using the URL shown in the **Code** menu. Open a terminal in the repository folder.

### 4. Install the dependencies

```sh
uv sync --locked
```

This creates a local `.venv` containing the workshop packages.

### 5. Open the tutorials in VS Code

Open the repository folder with **File > Open Folder**, then open either notebook:

- `dmc/dmc-bayesflow.ipynb`
- `hnn/HNN_Integration.ipynb`

Choose **Select Kernel > Python Environments** in the notebook's top-right corner, select this repository's `.venv`, and run the cells from top to bottom.

## Further reading

- Arruda et al. (2026), *Diffusion Models in Simulation-Based Inference: A Tutorial Review*: [paper](https://openreview.net/forum?id=86Q261z005) and [tutorial webpage](https://bayesflow-org.github.io/diffusion-experiments/).
- Li et al. (2026), *Amortized Bayesian Workflow*: [paper](https://openreview.net/forum?id=osV7adJlKD) and [tutorial webpage](https://pipme.github.io/amortized-Bayesian-workflow/).
- *BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python*: [software paper](https://arxiv.org/abs/2602.07098).
