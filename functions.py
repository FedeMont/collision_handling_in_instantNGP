# Types
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, TypeVar

# Math
import math

# Numpy
import numpy as np

# Scipy
from scipy.stats import norm

# Scikit-learn
from sklearn.model_selection import train_test_split

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Einops
from einops import rearrange, reduce, repeat

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision import io

# datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from datetime import datetime

# other
import traceback
import inspect
import os
import random
import wandb
from tqdm import tqdm
import yaml
from copy import deepcopy


# --- Cuda --- #
print("Cuda avilable:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Available device {i}:", torch.cuda.get_device_name(i))

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Current device {torch.cuda.current_device()}:", torch.cuda.get_device_name(torch.cuda.current_device()))

# torch.set_default_device(device)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# ------------ #

plt.style.use("ggplot")


# --- Random seed --- #
def set_random_seed(seed: int | None = None) -> None:
    random_seed = (
        np.random.randint(0, (2**16 - 1)) 
        if seed is None 
        else 
        seed
    )
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    print("Random seed:", random_seed)
# ------------------- #


# --- Debug functions --- #
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


decoded_functions = {
    # Dataset
    "__init__": 0,
    "__getitem__": 1,
    # "_create_sliding_windows": 2,
    # "_create_sliding_windows_in_batches": 3,
    # "_compose_images_from_windows": 4,
    "_create_input": 2,
    "_get_levels": 3,
    "_scale_to_grid": 4,
    "_create_kl_div_targets": 5,
    
    # general
    "__init__": 0,
    "forward": 1,
    "train_loop": 1,
    "test_loop": 1,
    "plot_images": 2,
    "differentiable_round": 3,
    "differentiable_histogram": 4,
    "differentiable_indexing": 5,
    
    # LearnableHashingFucntion
    "_learnable_hashing_function": 2,
    # "_multiresolution_hash": 3,
    "_static_hash": 4,
    # "_scale_to_grid": 3,
    # "_calc_bilinear_coefficients": 4,
    # "_calc_dummies": 5,
    # "_calc_hash_collisions": 6,
    # "_calc_uniques": 7,
    # "_hist_collisions": 8,
    
    # GeneralNeuralGaugeFields
    "_calc_bilinear_coefficients": 2,
    "_look_up_features": 3,
    "_bilinear_interpolation": 4,
    
    # Loss
    "_calc_hist_pdf": 2,
    "_plot_histograms": 3,
    
    # Test loop
    "create_indices_mapping": 1,

    # wandb
    "wandb_init": 1,
    "wandb_log": 2,
}


def log(texts, allowed: List | bool, color: bcolors = bcolors.OKCYAN) -> None:
    calling_func = inspect.stack()[1][0].f_code.co_name
    should_log = (
        (type(allowed) == bool and allowed)
        or 
        (type(allowed) == list and decoded_functions[calling_func] in allowed)
    )

    if should_log:
        stack = traceback.extract_stack()
        calling_frame = stack[-2]
        calling_line = calling_frame.line
        print(color, "Function:", calling_func, ", Line:", calling_line, bcolors.ENDC)

        try:
            print(*texts)
        except:
            print(texts)

        # print_allocated_memory(True)
        
        print(color, "-"*20, bcolors.ENDC)


def print_allocated_memory(log: bool = True):
    if log:
        stack = traceback.extract_stack()
        calling_frame = stack[-2]
        calling_line = calling_frame.line
        print(bcolors.HEADER, "Line: ", calling_line, ", from: ", inspect.stack()[1][0].f_code.co_name, bcolors.ENDC)

        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to gigabytes
        print(f"Allocated Memory: {allocated_memory:.2f} GB")

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to gigabytes
        print(f"Peak Allocated Memory: {peak_memory:.2f} GB")

        print(bcolors.OKCYAN, "-"*20, bcolors.ENDC)


def plot_images(outs: np.ndarray, targets: np.ndarray, allowed: List = [], is_test: bool = False) -> None:
    if allowed: # if allowed is not empty then check else do nothing
        if decoded_functions[inspect.stack()[0][0].f_code.co_name] in allowed:
            rows = outs.shape[0]
            cols = 2

            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axs = axs.flatten()
            for i in range(0, (rows * cols), cols):
                axs[i + 0].imshow(outs[i//cols])
                axs[i + 0].set_title("Prediction" if not is_test else "Output")
                axs[i + 1].imshow(targets[i//cols])
                axs[i + 1].set_title("Target")
            plt.show()
# ----------------------- #


# --- Backward Pass Differentiable Approximation --- #
# https://github.com/kitayama1234/Pytorch-BPDA
def differentiable_round(
    x: torch.Tensor, 
    round_function: Optional[callable] = torch.round
) -> torch.Tensor:
    
    forward_value = round_function(x)
    out = x.clone()
    out.data = forward_value.data

    def backward(grad_output):
        return grad_output

    out.register_hook(backward)

    return out
# -------------------------------------------------- #


# --- Differentiable Hisogram --- #
# https://github.com/hyk1996/pytorch-differentiable-histogram
def differentiable_histogram(
    x: torch.Tensor, 
    bins: Optional[int] = 255, 
    min: Optional[float] = 0.0, 
    max: Optional[float] = 1.0, 
    should_log: Optional[bool] = False
) -> torch.Tensor:

    if len(x.shape) == 4:
        n_samples, n_chns, _, _ = x.shape
    elif len(x.shape) == 2:
        n_samples, n_chns = 1, 1
    else:
        raise AssertionError('The dimension of input tensor should be 2 or 4.')

    hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
    log(("hist_torch:", hist_torch.shape), should_log)
    delta = (max - min) / bins

    BIN_Table = torch.arange(start=0, end=bins, step=1) * delta
    log(("BIN_Table:", BIN_Table.shape), should_log)

    for dim in range(1, bins-1, 1):
        h_r = BIN_Table[dim].item()             # h_r
        h_r_sub_1 = BIN_Table[dim - 1].item()   # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
        hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)

        del mask_sub
        del mask_plus
        del h_r
        del h_r_sub_1
        del h_r_plus_1

    log(("hist_torch:", hist_torch.shape), should_log)

    del BIN_Table

    return hist_torch / delta
# ------------------------------- #


# --- Differentiable Indexing --- #
class DifferentiableIndexing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, indexing_fn: callable):
        should_log: bool = False

        ctx.save_for_backward(input, indices)
        ctx.indexing_fn = indexing_fn

        try:
            log(("input:", input.shape, input.requires_grad, input.is_leaf), should_log, color=bcolors.HEADER)
        except:
            log(("input:", input.weight.shape, input.weight.requires_grad, input.weight.is_leaf), should_log, color=bcolors.HEADER)
        log(("indices:", indices.shape, indices.requires_grad, indices.is_leaf), should_log, color=bcolors.HEADER)

        # TODO - dividere in più parti, salvarsi il gradiente rispetto aglli indici e non fare backword su .long()
        output = indexing_fn.forward(input, indices.long())

        # TODO - vedere se con o senza backward i gradienti sono uguali?

        log(("output:", output.shape, output.requires_grad, output.is_leaf), should_log, color=bcolors.HEADER)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        should_log: bool = False

        input, indices = ctx.saved_tensors
        indexing_fn = ctx.indexing_fn

        log(("grad_output:", grad_output.shape), should_log, color=bcolors.HEADER)

        # grad_input = grad_output.clone()
        grad_input = indexing_fn.backward_input(input, grad_output, indices.long())
        
        log(("grad_input:", grad_input.shape), should_log, color=bcolors.HEADER)

        # Calculate gradient with respect to learnable indices (using STE)
        grad_indices = None
        if indices.requires_grad:
            grad_indices = indexing_fn.backward_indices(grad_output, indices.long())
            log(("grad_indices:", grad_indices.shape), should_log, color=bcolors.HEADER)

        return grad_input, grad_indices, None


# Abstract class
class Indexing():
    def __init__(self, ) -> None:
        pass

    def forward(self, x, indices) -> torch.Tensor:
        pass

    def backward_input(self, x, indices) -> torch.Tensor:
        pass

    def backward_indices(self, x, indices) -> torch.Tensor:
        pass


class Custom_Indexing(Indexing):
    def __init__(self, ) -> None:
        super(Custom_Indexing, self).__init__()

    def forward(self, x, indices) -> torch.Tensor:
        return x[indices]
    
    def backward_input(self, x, indices) -> torch.Tensor:
        return x[indices]
    
    def backward_indices(self, x, indices) -> torch.Tensor:
        return x[indices, 0]


class Hash_Tables_Indexing(Indexing):
    def __init__(self, ) -> None:
        super(Hash_Tables_Indexing, self).__init__()

    def forward(self, x, indices) -> torch.Tensor:
        out = x[indices]
        return out
    
    def backward_input(self, input, grad, indices) -> torch.Tensor:
        grad_input = torch.zeros_like(input)
        grad_input[indices] = grad

        return grad_input

    def backward_indices(self, grad, indices) -> torch.Tensor:
        return grad

# https://ai.stackexchange.com/questions/41264/how-to-do-backpropagation-with-argmax
def score_gradient_estimator(
    features: torch.Tensor, 
    indices: torch.Tensor
) -> torch.Tensor:
    
    cat = torch.distributions.Categorical(logits=indices)
    log_probs = cat.log_prob(indices.long()[..., indices.shape[-1]//2])
    return features[indices.long()], -log_probs,

# ------------------------------- #

# --- Metrics --- #
def calc_accuracy(predicted: np.ndarray, target: np.ndarray, size: int) -> float:
    return (np.equal(predicted, target).sum() / size) * 100


def calc_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.square(pred - target).mean()
    return 20 * np.log10(np.max(target)) - 10 * np.log10(mse) # psne
# --------------- #


# --- Wandb --- #
def wandb_init(
    config: Dict[str, Any],
    save_code: Optional[bool] = False
) -> None:
    
    to_log_config = {}
    to_log_config["time"] = config["time"]
    to_log_config.update(config["flags"])
    to_log_config.update(config["train_params"])

    log((to_log_config, ), config["log_flags"]["wandb"])
    log((to_log_config.keys(), ), config["log_flags"]["wandb"])

    if config["flags"]["should_wandb"]:
        wandb.init(
            entity=config["wandb"]["entity"],
            project=config["wandb"]["project"],
            name=config["wandb"]["name"],
            config=to_log_config,
            save_code=save_code
        ) 


def wandb_log(
    e: int,
    batch_size: int,
    results: Dict[str, Any],
    lr: Optional[float | None] = None,
    should_log: Optional[bool] = False,
    should_wandb: Optional[bool] = False
) -> None:
    """
    
    Parameters
    ----------
    e : int
        Current epoch.
    results : Dict[str, Any]
        Results object.
    lr : float | None, optional (default is None)
        Learning rate.
    should_log : bool, optional (default is False)
        Whether to log or not.
    should_wandb : bool, optional (default is False)
        Whether to log to wandb or not.
    
    Returns
    -------
    None
    """
    
    to_log = {}

    to_log["epoch"] = e

    if lr is not None:
        to_log["lr"] = lr

    for res_key, res_values in results.items():
        if res_values is not None:
            for key, values in res_values.items():
                if values is not None:
                    if isinstance(values, np.ndarray) or isinstance(values, List):
                        for i, value in enumerate(values):
                            # TODO: check, not sure if this is the best way to do it
                            num = ("level_" if (len(values) > batch_size) else "") + str(i) 

                            if isinstance(value, plt.Figure) or isinstance(value, np.ndarray):
                                to_log[f"{res_key}/media/{key}_{num}"] = wandb.Image(value)
                            else:
                                to_log[f"{res_key}/{key}_{num}"] = value
                    else:
                        to_log[f"{res_key}/{key}"] = values

    log(("log:", to_log), should_log)
    if should_wandb:
        wandb.log(to_log)

    del to_log


def wandb_finish(
    should_wand: Optional[bool] = False
) -> None:
    if should_wand:
        wandb.finish()
# ------------- #


# --- Save checkpoint --- #
def save_checkpoint(
    epoch: int,
    run_name: str,
    model: nn.Module,
    config: Dict[str, Any],
    best_loss: float,
    best_kl_div_loss: float,
    best_psnr: float,
    loss: float,
    kl_div_loss: float,
    psnr: float | None,
    NeRF_optimizer: Optional[torch.optim.Optimizer | None] = None,
    learnable_hash_optimizer: Optional[torch.optim.Optimizer | None] = None,
    save_weights_rate: Optional[int | None] = None,
    save_weights_path: Optional[str] = "./weights",
    should_early_stop: Optional[bool] = False,
    should_log: Optional[bool] = False,
) -> Tuple[float, float, float]:
    
    checkpoint = {
        "epoch": epoch,
        "run_name": run_name,
        "configuration": config,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "kl_div_loss": kl_div_loss,
        "psnr": psnr,
    }
    if NeRF_optimizer:
        checkpoint["optimizer_NeRF_state_dict"] = NeRF_optimizer.state_dict()
    if learnable_hash_optimizer:
        checkpoint["optimizer_learnable_hash_state_dict"] = learnable_hash_optimizer.state_dict()

    should_checkpoint_end = (
        (epoch == config["train_params"]["epochs"] - 1) 
        or should_early_stop
    )

    save_directory = os.path.join(save_weights_path, run_name)

    if (loss < best_loss): # only if the loss is better than the previous one save the model
        best_loss = loss
        weights_name = run_name + "_loss"

        if (save_weights_rate is not None and (epoch % save_weights_rate == 0)):
            os.makedirs(save_directory, exist_ok=True)
            torch.save(
                checkpoint, 
                os.path.join(save_directory, f"{weights_name}_checkpoint.pth")
            )

    if (kl_div_loss < best_kl_div_loss): # only if the kl_divloss is better than the previous one save the model
        best_kl_div_loss = kl_div_loss
        weights_name = run_name + "_kl_div_loss"

        if (save_weights_rate is not None and (epoch % save_weights_rate == 0)):
            os.makedirs(save_directory, exist_ok=True)
            torch.save(
                checkpoint, 
                os.path.join(save_directory, f"{weights_name}_checkpoint.pth")
            )
    
    should_checkpoint_psnr = (psnr > best_psnr) if psnr else False
    if should_checkpoint_psnr:
        best_psnr = psnr
        weights_name = run_name + "_psnr"

        if (save_weights_rate is not None and (epoch % save_weights_rate == 0)):
            os.makedirs(save_directory, exist_ok=True)
            torch.save(
                checkpoint, 
                os.path.join(save_directory, f"{weights_name}_checkpoint.pth")
            )

    if should_checkpoint_end:
        weights_name = run_name + "_last"

        if (save_weights_rate is not None and (epoch % save_weights_rate == 0)):
            os.makedirs(save_directory, exist_ok=True)
            torch.save(
                checkpoint, 
                os.path.join(save_directory, f"{weights_name}_checkpoint.pth")
            )

    log(("Checkpoint saved:", checkpoint), should_log, color=bcolors.OKGREEN)
    
    del checkpoint

    return best_loss, best_kl_div_loss, best_psnr
# ----------------------- #


# --- Training loop --- #
def train_loop(
    unique_grids_per_level: List[torch.Tensor],
    min_possible_collisions_per_level: torch.Tensor,
    scaled_coords: torch.Tensor,
    grid_coords: torch.Tensor,
    kl_div_targets_per_level: List[torch.Tensor],
    images_target: torch.Tensor,
    images_dimensions: torch.Tensor,
    multires_levels: torch.Tensor,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    gradient_clip: Optional[float | None] = None,
    should_draw_hists: Optional[bool] = False,
    should_log: Optional[List[int]| bool] = False,
) -> Dict[str, Any]:
    """
    Trains the model.
    
    Parameters
    ----------
    unique_grids_per_level : List[torch.Tensor]
        Unique images grids coordinates per level.
    min_possible_collisions_per_level : torch.Tensor
        Minimum possible collisions per level.
    scaled_coords : torch.Tensor
        Scaled images coordinates.
    grid_coords : torch.Tensor
        Grid images coordinates.
    kl_div_targets_per_level : torch.Tensor
        KL divergence targets per level.
    images_target : torch.Tensor
        Target images.
    images_dimensions : torch.Tensor
        Images dimensions.
    multires_levels : torch.Tensor
        Multiresolution levels.
    model : nn.Module
        Model to train.
    loss_fn : nn.Module
        Loss function.
    optimizers : Dict[str, torch.optim.Optimizer]
        Dictionary of optimizers.
    gradient_clip : float | None, optional (default is None)
        Gradient clipping value. If None then no clipping is applied.
    should_draw_hists : bool, optional (default is False)
        Whether to draw matplotlib histograms or not.
    should_log : List[int] | bool, optional (default is False)
        - 1: Log
        - 2: Plot.
        - 3: Log grads

    Returns
    -------
    Dict[str, Any]
        Dictionary with the train results.
    """
    log(("Train loop", ), should_log != False, color=bcolors.WARNING)

    model.train()

    for key, optimizer in optimizers.items():
        optimizer.zero_grad()

    images_pred, indices_log_probs, indices, sigmas, collisions, hists = model(
        unique_grids_per_level=unique_grids_per_level,
        min_possible_collisions_per_level=min_possible_collisions_per_level,
        scaled_coords=scaled_coords,
        grid_coords=grid_coords,
        multires_levels=multires_levels,
        should_draw_hists=should_draw_hists,
        should_show_hists=((2 in should_log) if should_log else should_log)
    )

    losses_results = loss_fn(
        indices_log_probs=indices_log_probs,
        indices=(
            indices 
            if isinstance(indices, list) 
            else 
            rearrange(indices, "batch pixels levels verts 1 -> batch levels pixels (verts 1)")[0] # 0 because images are all of same dimensions
        ),
        sigmas=(
            sigmas 
            if isinstance(sigmas, list) 
            else 
            rearrange(sigmas, "batch pixels levels verts 1 -> batch levels pixels (verts 1)")[0] # 0 because images are all of same dimensions
        ),
        min_possible_collisions_per_level=min_possible_collisions_per_level,
        kl_div_targets_per_level=kl_div_targets_per_level,
        images_pred=images_pred,
        images_target=images_target,
        model_named_parameters=model.named_parameters(),
        should_draw_hists=should_draw_hists,
    )

    if (3 in should_log) if should_log else should_log:
        indices_log_probs.retain_grad()

        if images_pred is not None:
            images_pred.retain_grad()
        
        if isinstance(indices, list):
            for l in range(len(indices)):
                indices[l].retain_grad()
        else:
            indices.retain_grad()

        if isinstance(sigmas, list):
            for l in range(len(sigmas)):
                sigmas[l].retain_grad()
        else:
            sigmas.retain_grad()

    losses_results["loss"].backward()

    if gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.learnable_hash_function_model.parameters(), gradient_clip, norm_type=2)

    if (3 in should_log) if should_log else should_log:
        log(
            (
                "indices_log_probs gradient: ",
                indices_log_probs.grad,
                indices_log_probs.grad_fn,
                indices_log_probs.shape,
                indices_log_probs.grad.sum()
            ),
            True,
            color=bcolors.OKGREEN
        )

        if images_pred is not None:
            log(
                (
                    "Images pred gradient: ", 
                    images_pred.grad, 
                    images_pred.grad_fn, 
                    images_pred.shape, 
                    images_pred.grad.sum()
                ), 
                True, 
                color=bcolors.OKGREEN
            )
        
        if isinstance(indices, list):
            log(
                (
                    "Indices gradient: ", 
                    [
                        (indices[l].grad, indices[l].grad_fn, indices[l].shape, indices[l].grad.sum())
                        for l in range(len(indices))
                    ]
                ), 
                True, 
                color=bcolors.OKGREEN
            )
        else:
            log(
                (
                    "Indices gradient: ", 
                    indices.grad, 
                    indices.grad_fn, 
                    indices.shape, 
                    indices.grad.sum()
                ), 
                True, 
                color=bcolors.OKGREEN
            )
        
        if isinstance(sigmas, list):
            log(
                (
                    "Sigmas gradient: ", 
                    [
                        (sigmas[l].grad, sigmas[l].grad_fn, sigmas[l].shape, sigmas[l].grad.sum())
                        for l in range(len(sigmas))
                    ]
                ), 
                True, 
                color=bcolors.OKGREEN
            )
        else:
            log(("Sigmas gradient: ", sigmas.grad, sigmas.grad_fn, sigmas.shape, sigmas.grad.sum()), True, color=bcolors.OKGREEN)

        for name, param in model.named_parameters():
            if "learnable_hash_function_model" in name:
                log(("Grad", name, param.grad, param.grad.sum()), True, color=bcolors.OKGREEN)
                log(("Params", name, param), True, color=bcolors.FAIL)

    for key, optimizer in optimizers.items():
        # TODO HOW?
        ######
        # Lastly, we skip Adam steps for hash table entries whose gradient is exactly 0. 
        # This saves ∼10% performance when gradients are sparse, which is a common occurrence with 𝑇 ≫ BatchSize. 
        # Even though this heuristic violates some of the assumptions behind Adam, we observe no degradation in convergence.
        ######
        optimizer.step()

    pred_images, target_images, images_psnr = None, None, None
    if images_pred is not None:
        pred_images = (images_pred * 255).reshape(-1, *images_dimensions, 3).to(int).detach().cpu().numpy()
        target_images = (images_target * 255).reshape(-1, *images_dimensions, 3).to(int).detach().cpu().numpy()

        plot_images(pred_images, target_images, should_log)

        images_psnr = [calc_psnr(pred_images[i], target_images[i]) for i in range(pred_images.shape[0])]

    results = {
        "pred_images": pred_images,
        "target_images": target_images,
        "images_psnr": images_psnr,
        
        "min_possible_collisions_per_level": min_possible_collisions_per_level.detach().cpu().numpy(),
        "collisions": collisions.detach().cpu().numpy() if collisions is not None else None,
        "histograms": losses_results["hists"],

        "kl_div_losses": losses_results["kl_div_losses"].detach().cpu().numpy(),
        "sigmas_losses": losses_results["sigmas_losses"].detach().cpu().numpy(),
        "reg_loss": losses_results["reg_loss"],
        "indices_log_probs_loss": losses_results["indices_log_probs_loss"],
        "images_losses": losses_results["images_losses"].detach().cpu().numpy() if losses_results["images_losses"] is not None else None,

        "loss": losses_results["loss"].item(),
    }

    return results
# --------------------- #


# --- Test loop --- #
def create_indices_mapping(
    pred: np.ndarray,
    h: int,
    w: int,
    min_possible_collisions: np.ndarray,
    hashed: np.ndarray,
    hash_table_size: int,
    save_dir: str | None = None,
    should_use_all_levels: bool = False,
    should_show: bool = False,
    should_log: List[int] = [],
) -> List[plt.Figure]:
    """
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted images.
    h : int
        Images height.
    w : int
        Images width.
    min_possible_collisions: np.ndarray
        Minimum possible collisions per level
    hashed : np.ndarray
        Hashed coordinates.
    hash_table_size : int
        Hash table size.
    save_dir : str | None, optional (default is None)
        Directory to save the plots.
    should_use_all_levels : bool, optional (default is False)
        Whether to use all levels or not.
    should_show : bool, optional (default is False)
        Whether to show the plots or not.
    should_log : List[int], optional (default is [])
        - 1: Log

    Returns
    -------
    List[plt.Figure]
        List of plots, one for each level, for each batch.
    """
    
    batch, pixels, levels, verts, xyz = hashed.shape
    original_vertex = 0

    maps = []
    colors = plt.cm.viridis(np.linspace(0, 1, hash_table_size))
    cmap = ListedColormap(colors)

    for b in range(batch):
        pred_b = pred[b, :, :].reshape(h, w, 3)

        for l in range(levels):
            if min_possible_collisions[l] <= 0 and not should_use_all_levels:
                maps.append(None)
                continue

            fig, ax = plt.subplots()
            log(("Batch:", b, ", level:", l), should_log, color=bcolors.OKGREEN)

            hashed_bl = hashed[b, :, l, original_vertex, :].squeeze(-1).reshape(h, w)

            plt.imshow(pred_b)
            plt.imshow(hashed_bl, cmap=cmap, alpha=0.75, vmin=0, vmax=hash_table_size)

            ticks = [n for n in range(0, hash_table_size + 10, 10)]
            plt.colorbar(ticks=ticks, label="Hashed indices", orientation="vertical", alpha=1.0)
            
            plt.title(f"Image batch {b}, level {l}")

            if should_show:
                plt.show()

            if save_dir is not None:
                plt.title(f"Image {save_dir.split('/')[-1]} batch {b}, level {l}")
                fig.savefig(f"{save_dir}/{save_dir.split('/')[-1]}_batch_{b}_level_{l}_image.png", dpi=fig.dpi)
            plt.close()

            maps.append(fig)

    return maps


def test_loop(
    min_possible_collisions_per_level: torch.Tensor,
    scaled_coords: torch.Tensor,
    grid_coords: torch.Tensor,
    kl_div_targets_per_level: torch.Tensor,
    images_target: torch.Tensor,
    images_dimensions: torch.Tensor,
    multires_levels: torch.Tensor,
    model: nn.Module,
    loss_fn: nn.Module,
    should_use_all_levels: Optional[bool] = False,
    should_draw: Optional[bool] = False,
    should_log: Optional[List[int]| bool] = False,
) -> Dict[str, Any]:
    """
    Trains the model.
    
    Parameters
    ----------
    min_possible_collisions_per_level : torch.Tensor
        Minimum possible collisions per level.
    scaled_coords : torch.Tensor
        Scaled images coordinates.
    grid_coords : torch.Tensor
        Grid images coordinates.
    kl_div_targets_per_level : torch.Tensor
        KL divergence targets per level.
    images_target : torch.Tensor
        Target images.
    images_dimensions : torch.Tensor
        Images dimensions.
    multires_levels : torch.Tensor
        Multiresolution levels.
    model : nn.Module
        Model to train.
    loss_fn : nn.Module
        Loss function.
    should_use_all_levels : bool, optional (default is False)
        Whether to use all levels or not.
    should_draw : bool, optional (default is False)
        Whether to draw matplotlib histograms and indices mappings or not.
    should_log : List[int] | bool, optional (default is False)
        - 1: Log
        - 2: Plot.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the test results.
    """
    log(("Test loop", ), should_log != False, color=bcolors.WARNING)

    model.eval()

    images_pred, indices_log_probs, indices, sigmas, collisions, hists = model(
        unique_grids_per_level=None,
        min_possible_collisions_per_level=min_possible_collisions_per_level,
        scaled_coords=scaled_coords,
        grid_coords=grid_coords,
        multires_levels=multires_levels,
        should_draw_hists=should_draw,
        should_show_hists=((2 in should_log) if should_log else should_log)
    )

    with torch.no_grad():
        losses_results = loss_fn(
            indices_log_probs=indices_log_probs,
            indices=rearrange(indices, "batch pixels levels verts 1 -> batch levels pixels (verts 1)")[0], # 0 because images are all of same dimensions
            sigmas=rearrange(sigmas, "batch pixels levels verts 1 -> batch levels pixels (verts 1)")[0], # 0 because images are all of same dimensions
            min_possible_collisions_per_level=min_possible_collisions_per_level,
            kl_div_targets_per_level=kl_div_targets_per_level,
            images_pred=images_pred,
            images_target=images_target,
            model_named_parameters=model.named_parameters(),
            should_draw_hists=should_draw,
        )

    pred_images, target_images, images_psnr = None, None, None
    indices_mappings = None
    if images_pred is not None:
        pred_images = (images_pred * 255).reshape(-1, *images_dimensions, 3).to(int).detach().cpu().numpy()
        target_images = (images_target * 255).reshape(-1, *images_dimensions, 3).to(int).detach().cpu().numpy()

        plot_images(pred_images, target_images, should_log)

        images_psnr = [calc_psnr(pred_images[i], target_images[i]) for i in range(pred_images.shape[0])]

        if should_draw:
            indices_mappings = create_indices_mapping(
                pred=pred_images, 
                h=images_dimensions[0], 
                w=images_dimensions[1], 
                min_possible_collisions=min_possible_collisions_per_level.detach().cpu().numpy(), 
                hashed=indices.detach().cpu().numpy(), 
                hash_table_size=kl_div_targets_per_level[-1].shape[0],
                should_use_all_levels=should_use_all_levels,
                should_show=False, 
                should_log=should_log
            )

    results = {
        "pred_images": pred_images,
        "target_images": target_images,
        "images_psnr": images_psnr,
        
        "min_possible_collisions_per_level": min_possible_collisions_per_level.detach().cpu().numpy(),
        "collisions": collisions.detach().cpu().numpy() if collisions is not None else None,
        "histograms": losses_results["hists"],

        "kl_div_losses": losses_results["kl_div_losses"].detach().cpu().numpy(),
        "sigmas_losses": losses_results["sigmas_losses"].detach().cpu().numpy(),
        "reg_loss": losses_results["reg_loss"],
        "indices_log_probs_loss": losses_results["indices_log_probs_loss"],
        "images_losses": losses_results["images_losses"].detach().cpu().numpy() if losses_results["images_losses"] is not None else None,

        "indices_mappings": indices_mappings,

        "loss": losses_results["loss"].item(),
    }

    return results
# ----------------- #


# --- Run --- #
def run(
    data: Dict[str, Any],
    model: nn.Module,
    loss_fn: nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    gradient_clip: Optional[float | None] = None,
    should_use_all_levels: Optional[bool] = False,
    should_draw: Optional[bool] = False,
    should_log: Optional[List | bool] = False,
    device: Optional[torch.device | str] = "cuda:0",
) -> Dict[str, Any]:
        
    results = {}

    train_results = train_loop(
        unique_grids_per_level=data["unique_grids_per_level"],
        min_possible_collisions_per_level=data["min_possible_collisions_per_level"],
        kl_div_targets_per_level=data["kl_div_targets_per_level"],
        scaled_coords=data["scaled_X"],
        grid_coords=data["grid_X"],
        images_target=data["Y"],
        images_dimensions=data["dimensions"],
        multires_levels=data["multires_levels"],
        model=model,
        loss_fn=loss_fn,
        optimizers=optimizers,
        gradient_clip=gradient_clip,
        should_draw_hists=should_draw,
        should_log=should_log["train"],
    )
    if train_results is not None:
        results["train"] = train_results

    test_results = test_loop(
        min_possible_collisions_per_level=data["min_possible_collisions_per_level"],
        kl_div_targets_per_level=data["kl_div_targets_per_level"],
        scaled_coords=data["scaled_X"],
        grid_coords=data["grid_X"],
        images_target=data["Y"],
        images_dimensions=data["dimensions"],
        multires_levels=data["multires_levels"],
        model=model,
        loss_fn=loss_fn,
        should_use_all_levels=should_use_all_levels,
        should_draw=should_draw,
        should_log=should_log["test"],
    )
    if test_results is not None:
        results["test"] = test_results
    
    return results
# ----------- #