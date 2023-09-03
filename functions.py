from typing import Tuple, List

import numpy as np
from math import atan2
import cv2
from PIL import Image
import os
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FixedLocator
import matplotlib.pyplot as plt

import collections, functools, operator
from collections import Counter
import itertools

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from torchinfo import summary

from einops import rearrange, reduce, repeat

import traceback
from pprint import pprint

import wandb
from tqdm import tqdm

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
    
from datetime import datetime

from params import *


random_seed = None
if should_random_seed:
    random_seed = 2**16 - 1
    torch.manual_seed(random_seed)
    print("Random seed:", random_seed)

device = torch.device('cuda')
print("Device:", device)

torch.set_default_device(device)

plt.style.use("ggplot")


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


def print2(texts, log: bool = False) -> None:
    if log:
        stack = traceback.extract_stack()
        calling_frame = stack[-2]
        calling_line = calling_frame.line
        print(bcolors.OKCYAN, "Line: ", calling_line, bcolors.ENDC)
        for text in texts:
            print(text)
        print(bcolors.OKCYAN, "-"*20, bcolors.ENDC)


def print_allocated_memory(log: bool = should_log_allocated_memory):
    if log:
        stack = traceback.extract_stack()
        calling_frame = stack[-2]
        calling_line = calling_frame.line
        print(bcolors.HEADER, "Line: ", calling_line, bcolors.ENDC)

        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to gigabytes
        print(f"Allocated Memory: {allocated_memory:.2f} GB")

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to gigabytes
        print(f"Peak Allocated Memory: {peak_memory:.2f} GB")

        print(bcolors.OKCYAN, "-"*20, bcolors.ENDC)


def get_optimizer(
    net: torch.nn.Module,
    encoding_lr: float,
    HPD_lr: float,
    MLP_lr: float,
    encoding_weight_decay: float,
    HPD_weight_decay: float,
    MLP_weight_decay: float,
    betas: tuple = (0.9, 0.99),
    eps: float = 1e-15
):

    if should_use_hash_function:
        optimizer = torch.optim.Adam(
            [
                {"params": net.encoding.parameters(), "lr": encoding_lr, "weight_decay": encoding_weight_decay},
                {"params": net.mlp.parameters(), "lr": MLP_lr, "weight_decay": MLP_weight_decay}
            ],
            betas=betas,
            eps=eps,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": net.encoding.parameters(), "lr": encoding_lr, "weight_decay": encoding_weight_decay},
                {"params": net.HPD.parameters(), "lr": HPD_lr, "weight_decay": HPD_weight_decay},
                {"params": net.mlp.parameters(), "lr": MLP_lr, "weight_decay": MLP_weight_decay}
            ],
            betas=betas,
            eps=eps,
        )
    return optimizer


def calc_accuracy(predicted: np.ndarray, target: np.ndarray, size: int) -> float:
    return (np.equal(predicted, target).sum() / size) * 100


def calc_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.square(pred - target).mean()
    return 20 * np.log10(np.max(target)) - 10 * np.log10(mse) # psne


def train_step(
    net: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim,
    x: torch.Tensor,
    target: torch.Tensor,
    w: int,
    h: int,
    hash_table_size: int,
    topk_k: int,
    l_mse: float,
    l_js_kl: float,
    l_collisions: float,
    batch_percentage: float = 1,
    num_levels: int = 4,
    should_bw: bool = False,
    should_calc_counts: bool = False,
    should_shuffle: bool = True,
    shuffled_indices: any = None,
    reordered_indices: any = None,
    previous_collisions: torch.Tensor = torch.tensor([]),
    previous_min_possible_collisions: torch.Tensor = torch.tensor([])
):
    net.train()

    shape = w * h

    num_batches = int(np.ceil(shape / (shape * batch_percentage)))

    batch_mse_losses = np.zeros((num_batches, ))
    batch_kl_div_losses = np.zeros((num_batches, num_levels))
    batch_collisions_losses = np.zeros((num_batches, num_levels))
    batch_total_losses = np.zeros((num_batches, ))
    batch_counts_per_level = np.zeros((num_batches, num_levels), dtype=object)

    batch_outputs = torch.empty((shape, 3 if not should_bw else 1))

    if should_use_hash_function:
        batch_indices_topk = torch.empty((w*h, num_levels, 2**x.shape[1]))
    else:
        batch_indices_topk = torch.empty((w*h, num_levels, 2**x.shape[1], int(topk_k * 1/batch_size)))

    print_allocated_memory() # After initializaion of batch_indices_topk

    for batch_id in range(num_batches):

        # Divide input in batches
        start_idx = batch_id * int(batch_percentage * shape)
        stop_idx = (batch_id + 1) * int(batch_percentage * shape) if batch_id != num_batches else shape

        if should_shuffle:
            batch_x = x[shuffled_indices[start_idx : stop_idx]]
            batch_target = target[shuffled_indices[start_idx : stop_idx]]
        else:
            batch_x = x[start_idx : stop_idx]
            batch_target = target[start_idx : stop_idx]
        # end

        print_allocated_memory() # After initializaion of batch_x and batch_target

        # print("batch_x:", batch_x, batch_x.shape)

        optimizer.zero_grad()

        output, probs, indices_topk, counts_per_level = net(batch_x, batch_percentage, should_calc_counts=should_calc_counts)
        del batch_x

        print_allocated_memory() # After net()

        batch_counts_per_level[batch_id] = np.array(counts_per_level) if should_calc_counts else np.zeros((num_levels, ))
        del counts_per_level

        batch_outputs[start_idx : stop_idx, :] = output

        if should_use_hash_function:
            batch_indices_topk[start_idx : stop_idx] = indices_topk
        else:
            batch_indices_topk[start_idx : stop_idx, ..., (batch_id * topk_k):((batch_id + 1) * topk_k)] = indices_topk

        del indices_topk

        print_allocated_memory() # After assigning results to batches

        # print(counts_per_level)
        # print("OUTPUT:", output, output.shape)
        # print("TARGET:", batch_target, batch_target.shape)

        # print("output:", output, "batch_target:", batch_target, "hash_table_size:", hash_table_size, "probs:", probs, probs.shape)
        mse_loss, kl_div_losses, collisions_losses = loss_fn(
            output,
            batch_target,
            probs.shape[-1] if not should_use_hash_function else None,
            probs,
            previous_collisions if not should_use_hash_function else None,
            previous_min_possible_collisions if not should_use_hash_function else None
            # (previous_collisions - previous_min_possible_collisions) if not should_use_hash_function else None
        )
        del output
        del batch_target
        del probs
        # print("mse_loss:", mse_loss, "collisions_losses:", collisions_losses.sum(0))

        print_allocated_memory() # After loss_fn()

        loss = l_mse * mse_loss
        if not should_use_hash_function:
            loss = loss + ((l_js_kl * kl_div_losses) + (l_collisions * collisions_losses if collisions_losses.nelement() != 0 else 1)).sum(0)

        batch_mse_losses[batch_id] = mse_loss.item()
        del mse_loss

        batch_kl_div_losses[batch_id] = (
            kl_div_losses.detach().cpu().numpy()
            if not should_use_hash_function else
            None
        )
        del kl_div_losses

        batch_collisions_losses[batch_id] = (
            (
                collisions_losses.detach().cpu().numpy()
                if collisions_losses.nelement() != 0 else np.ones((num_levels, ))
            )
            if not should_use_hash_function else
            None
        )
        del collisions_losses

        batch_total_losses[batch_id] = loss.item()
        # print("Loss:", loss.item())

        print_allocated_memory() # After assigning losses to batches

        loss.backward()
        del loss

        print_allocated_memory() # After loss.backward()

        # for name, param in net.named_parameters():
        #     print(name, param.requires_grad, param.grad, param.grad_fn)
        # print("-"*8)

        optimizer.step()

        print_allocated_memory() # After optimizer.step()

    del x
    del target

    loss_item = np.mean(batch_total_losses)
    del batch_total_losses

    mse_loss = np.mean(batch_mse_losses)
    del batch_mse_losses

    kl_div_losses = (
        np.mean(batch_kl_div_losses, axis=0)
        if not should_use_hash_function else
        None
    )
    del batch_kl_div_losses

    collisions_losses = (
        np.mean(batch_collisions_losses, axis=0)
        if not should_use_hash_function else
        None
    )
    del batch_collisions_losses

    output = batch_outputs[reordered_indices] if should_shuffle else batch_outputs
    del batch_outputs

    # print(output, output.shape)

    indices_topk = batch_indices_topk[reordered_indices] if should_shuffle else batch_indices_topk
    del batch_indices_topk

    print_allocated_memory() # After meaning batches

    if should_calc_counts:
        indices_per_level = [
            dict(zip(*np.unique(level, return_counts=True)))
            for level in rearrange(indices_topk, "p l v k -> l (p v k)" if not should_use_hash_function else "p l v -> l (p v)").detach().cpu().numpy()
        ]
        print2((indices_per_level[0], len(indices_per_level), len(indices_per_level[0].keys())), False)
    else:
        indices_per_level = []

    collisions, min_possible_collisions = net.calc_hash_collisions(indices_topk)
    del indices_topk

    print_allocated_memory() # After net.encoding.calc_hash_collisions

    to_show_img = (output*255).reshape((h, w, 3) if not should_bw else (h, w))
    del output
    to_show_img = to_show_img.int()
    to_show_img = to_show_img.detach().cpu().numpy()

    print_allocated_memory() # After creating to_show_img

    counts_per_level = [
        dict(
            functools.reduce(
                operator.add,
                map(collections.Counter, batch_counts_per_level[:, i])
            )
        )
        for i in range(num_levels)
        if should_calc_counts
    ]
    del batch_counts_per_level

    # print("counts_per_level:", counts_per_level)

    print_allocated_memory() # After reducing counts_per_level

    return loss_item, to_show_img, collisions, min_possible_collisions, counts_per_level, mse_loss, kl_div_losses, collisions_losses, indices_per_level


def counts_per_level_histograms(
    counts_per_level: List[dict],
    hash_table_size: int,
    should_draw: bool = False,
    is_test_only: bool = False
) -> List[matplotlib.figure.Figure]:
    """

    Parameters
    ----------
    counts_per_level: List[dict]
        A list of dictionaries containing the number of times an index has been obtained for each level
    hash_table_size: int
        The number of indices
    # step: int (optional)
    #     The step size of the xticks ( default 50 )
    should_draw: bool (optional)
        Wether or not to plt.show the histograms ( default False )

    Returns
    ----------
    List[matplotlib.figure.Figure]
        A list containing the counts per level histogram figures for each level

    """

    to_return = []

    dec_number = hash_table_size * 0.1
    if dec_number < 100 or dec_number % 100 < 10:
        if dec_number % 10 < 5:
            step = (dec_number // 5 * 5)
        else:
            step = (dec_number // 10 * 10)
    else:
        step = (dec_number // 100 * 100)

    if is_test_only and len(counts_per_level) > 1:
        n_levels = len(counts_per_level)

        fig, axs = plt.subplots(((n_levels // 2) + (n_levels % 2)), 2, figsize=(20, 10))

        axs = axs.flatten()
        for level, counts in enumerate(counts_per_level):
            ax = axs[level]
            counts_data = {i: counts.get(i, 0) for i in range(hash_table_size)}

            ax.bar(range(hash_table_size), counts_data.values(), width=1, align="center", edgecolor="grey")

            ax.set_xlim(-1, hash_table_size)
            ax.xaxis.set_major_locator(MultipleLocator(int(step)))
            ax.xaxis.set_minor_locator(MultipleLocator(int(step * 0.1)))

            y_max = max(counts.values())
            ax.set_ylim(bottom=0, top=(y_max + y_max*0.05))
            ax.set_title(f"Level {level} ({hash_table_size})")
            ax.set_xlabel("Hashed indices")  # viene tagliato perchè i ticks sono ruotati
            ax.set_ylabel("Counts")

        to_return.append(fig)
        if should_draw:
            plt.tight_layout()
            plt.show()

        del counts_data
        plt.close()

    else:
        for level, counts in enumerate(counts_per_level):
            counts_data = {i: counts.get(i, 0) for i in range(hash_table_size)}

            fig, ax = plt.subplots(figsize=(15, 5))
            ax.bar(range(hash_table_size), counts_data.values(), width=1, align="center", edgecolor="grey")

            ax.set_xlim(-1, hash_table_size)
            ax.xaxis.set_major_locator(MultipleLocator(int(step)))
            ax.xaxis.set_minor_locator(MultipleLocator(int(step * 0.1)))

            y_max = max(counts.values())
            plt.ylim(bottom=0, top=(y_max + y_max*0.05))
            plt.title(f"Level {level} ({hash_table_size})")
            plt.xlabel("Hashed indices")  # viene tagliato perchè i ticks sono ruotati
            plt.ylabel("Counts")

            to_return.append(fig)
            if should_draw:
                plt.show()

            del counts_data
            plt.close()

    return to_return


def get_grid_search_configs(configs: dict) -> list:
    grid_search = [dict(zip(configs.keys(), cc)) for cc in itertools.product(*configs.values())]

    prev_seen = set()
    def filter_config(obj):
        if obj["should_sum_js_kl_div"]:
            obj["should_js_div"] = False
        else:
            obj["loss_gamma"] = 0

        if tuple(obj.items()) in prev_seen:
            return False
        prev_seen.add(tuple(obj.items()))
        return True

    filtered_grid_search = list(
        filter(
            lambda obj: filter_config(obj),
            grid_search
        )
    )

    return filtered_grid_search


def grid_search_loop(
    filtered_grid_search: list,
    x: torch.Tensor,
    y: torch.Tensor,
    w: int,
    h: int,
    image_name: str,
    og_image: np.ndarray,
    shuffled_indices: torch.Tensor,
    reordered_indices: torch.Tensor,
    GeneralNeuralGaugeFields: nn.Module,
    Loss: nn.Module,
    EarlyStopping: nn.Module,
    should_bw: bool = False,
    start_id_param: int = 0,
    end_id_param: int = None,
    is_test_only: bool = False,
    wandb_entity: str = "dl_project_bussola-fasoli-montagna",
    wandb_project: str = "cv_project_final_grid_search",
    wandb_name: str = None,
    drive_folder: str = "./"
):
    end_id_param = end_id_param if end_id_param != None else len(filtered_grid_search)
    
    for id_param, params in enumerate(filtered_grid_search[start_id_param:end_id_param]):
        id_param += start_id_param

        time = (datetime.now(ZoneInfo("Europe/Rome"))).strftime("%Y%m%d%H%M%S") if wandb_name == None else wandb_name
        print("RUN:", time)

        print(bcolors.HEADER, f"Grid search params: {id_param}", bcolors.ENDC)
        print(params)

        should_shuffle_pixels = params["should_shuffle_pixels"]
        should_keep_topk_only = params["should_keep_topk_only"]

        should_sum_js_kl_div = params["should_sum_js_kl_div"]
        loss_gamma = params["loss_gamma"]

        should_js_div = params["should_js_div"]

        l_mse = params["l_mse"]
        l_js_kl = params["l_js_kl"]
        l_collisions = params["l_collisions"]
        
        MLP_lr = params["MLP_lr"]
        HPD_lr = params["HPD_lr"]

        topk_k = params["topk_k"]

        # ------------------------------ #
        #       TRAINING PARAMS          #

        loss_gamma = loss_gamma if should_sum_js_kl_div else -1

        loss_epsilon = 1                                        #@param {type: "number"}
        loss_epsilon = loss_epsilon if should_sum_js_kl_div else (0 if should_js_div else 1)

        # ------------------------------ #
        #       MODEL DEFINITION         #

        GGNF_model = GeneralNeuralGaugeFields(
            input_dim=x.shape[1],
            hash_table_size=hash_table_size,
            num_levels=num_levels,
            n_min=n_min,
            n_max=n_max,
            MLP_hidden_layers_widths=MLP_hidden_layers_widths,
            HPD_hidden_layers_widths=HPD_hidden_layers_widths,
            HPD_out_features=HPD_out_features,
            feature_dim=feature_dim,
            topk_k=topk_k,
            should_keep_topk_only=should_keep_topk_only,
            should_bw=should_bw,
            should_log=False                                   #@param {type: "boolean"}
        )

        # print(summary(GGNF_model, list(x.shape)))
        # print(GGNF_model)

        loss_fn = Loss(
            delta=1,
            gamma=loss_gamma,
            epsilon=loss_epsilon,
            should_log=False                                   #@param {type: "boolean"}
        )

        optimizer = get_optimizer(
            net=GGNF_model,
            encoding_lr=encoding_lr,
            HPD_lr=HPD_lr,
            MLP_lr=MLP_lr,
            encoding_weight_decay=encoding_weight_decay,
            HPD_weight_decay=HPD_weight_decay,
            MLP_weight_decay=MLP_weight_decay,
        )

        early_stopper = EarlyStopping(
            tolerance=tolerance,
            min_delta=min_delta
        )

        if not is_test_only:
            # ------------------------------ #
            #          WANDB INIT            #
            # start a new wandb run to track this script
            wandb.init(
                entity = wandb_entity,
                # set the wandb project where this run will be logged
                project = wandb_project,

                group = image_name,

                name = time,

                # track hyperparameters and run metadata
                config = {
                    "id_grid_search_params":        id_param,
                    "grid_search_params":           params,
                    "random_seed":                  random_seed,
                    "HPD_learning_rate":            HPD_lr,
                    "encoding_learning_rate":       encoding_lr,
                    "MLP_learning_rate":            MLP_lr,
                    "encoding_weight_decay":        encoding_weight_decay,
                    "HPD_weight_decay":             HPD_weight_decay,
                    "MLP_weight_decay":             MLP_weight_decay,
                    "batch_size%":                  batch_size,
                    "shuffled_pixels":              should_shuffle_pixels,
                    "normalized_data":              True if not should_batchnorm_data else "BatchNorm1d",
                    "architecture":                 "GeneralNeuralGaugeFields",
                    "dataset":                      image_name,
                    "epochs":                       epochs,
                    "color":                        'RGB' if not should_bw else 'BW',
                    "hash_table_size":              hash_table_size,
                    "num_levels":                   num_levels,
                    "n_min":                        n_min,
                    "n_max":                        n_max,
                    "MLP_hidden_layers_widths":     str(MLP_hidden_layers_widths),
                    "HPD_hidden_layers_widths":     str(HPD_hidden_layers_widths),
                    "HPD_out_features":             HPD_out_features,
                    "feature_dim":                  feature_dim,
                    "topk_k":                       topk_k,
                    "loss_type":                    "JS+KLDiv" if should_sum_js_kl_div else ("KLDiv" if not should_js_div else "JSDiv"),
                    "loss_lambda_MSE":              l_mse,
                    "loss_lambda_JS_KL":            l_js_kl,
                    "loss_lambda_collisions":       l_collisions,
                    "loss_gamma":                   loss_gamma,
                    "loss_epsilon":                 loss_epsilon,
                    "inplace_scatter":              should_inplace_scatter,
                    "MLP_activations":              "LeakyReLU" if should_leaky_relu else "ReLU",
                    "collisions_loss_probs":        "topk_only" if should_keep_topk_only else "hash_table_size",
                    "avg_topk_features":            "softmax_avg" if should_softmax_topk_features else ("weighted_avg" if should_softmax_topk_features != None else None),
                    "hash_type":                    "HPD" if not should_use_hash_function else "hash_function"
                }
            )

        # ------------------------------ #

        previous_collisions: torch.Tensor = torch.tensor([])
        previous_min_possible_collisions: torch.Tensor = torch.tensor([])

        should_show_counts = False
        if not is_test_only:
            plt.ioff()

        print_allocated_memory() # Start

        pbar = tqdm(range(0, epochs))

        best_psnr = 0

        for e in pbar:
            train_loss, train_img, collisions, min_possible_collisions, counts_per_level, mse_loss, js_kl_div_losses, collisions_losses, indices_per_level = train_step(
                net=GGNF_model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                x=x.clone(),
                target=y.clone(),
                w=w,
                h=h,
                hash_table_size=hash_table_size,
                topk_k=topk_k,
                l_mse=l_mse,
                l_js_kl=l_js_kl,
                l_collisions=l_collisions,
                batch_percentage=batch_size,
                num_levels=num_levels,
                should_bw=should_bw,
                should_calc_counts=(e == epochs - 1) or (should_show_counts) or (e % histograms_rate == 0),
                should_shuffle=should_shuffle_pixels,
                shuffled_indices=shuffled_indices,
                reordered_indices=reordered_indices,
                previous_collisions=previous_collisions,
                previous_min_possible_collisions=previous_min_possible_collisions
            )

            previous_collisions = collisions
            previous_min_possible_collisions = min_possible_collisions

            train_accuracy = calc_accuracy(train_img, og_image, w*h)

            train_psnr = calc_psnr(train_img, og_image)

            pbar.set_description(f"Training_psnr: {train_psnr}")
            
            if is_test_only:
                # print(f"Epoch: {e}")
                # print(f"\tKL_div Losses: {js_kl_div_losses}, Collisions Losses: {collisions_losses}")
                # print(f"\tMSE loss: {mse_loss}, KL_Collisions Loss: {np.sum(l_js_kl * js_kl_div_losses + (l_collisions * collisions_losses if len(collisions_losses) != 0 else 1), axis=0)}, Training_loss: {train_loss}")
                # print(f"\tTraining_accuracy: {train_accuracy}, Training_psnr: {train_psnr}")
                # print("\tCollisions:", collisions)
                # print("\tMin Possible Collisions:", min_possible_collisions)
                # print("-"*80)

                _, axs = plt.subplots(1, 2, figsize=(12, 12))
                axs = axs.flatten()
                for img, ax in zip([["og_image", og_image], ["output", train_img]], axs):
                    if not should_bw:
                        ax.imshow(img[1])
                    else:
                        ax.imshow(img[1], cmap="gray")
                    ax.set_title(img[0])
                plt.show()

                counts_per_level_histograms(indices_per_level, hash_table_size, (e == epochs - 1) or (should_show_counts) or (e % histograms_rate == 0), is_test_only)
            else:

                wandb_image = wandb.Image(
                    train_img,
                    caption=f"Train Image, epoch:{e}"
                )

                log = {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_psnr": train_psnr,
                    "train_image": wandb_image,
                    "mse_loss": mse_loss
                }
                del wandb_image

                if not should_use_hash_function:
                    for level, loss in enumerate(js_kl_div_losses):
                        log[f"kl_div_loss_level{level}"] = loss

                if not should_use_hash_function:
                    for level, loss in enumerate(collisions_losses):
                        log[f"collisions_loss_level{level}"] = loss

                if not should_use_hash_function:
                    for level in range(num_levels):
                    # for level, kl_div_loss, collisions_loss in enumerate(zip(kl_div_losses, collisions_losses)):
                        log[f"kl_collisions_loss_level{level}"] = np.sum(l_js_kl * js_kl_div_losses[level] + l_collisions * collisions_losses[level], axis=0)

                for level, n_coll in enumerate(collisions):
                    log[f"collisions_level{level}"] = n_coll
                    log[f"min_possible_collisions_level{level}"] = min_possible_collisions[level]

                if (e == epochs - 1) or (should_show_counts) or (e % histograms_rate == 0):
                    histograms = counts_per_level_histograms(indices_per_level, hash_table_size, False, is_test_only)
                    for level, fig in enumerate(histograms):
                        log[f"hist_counts_level{level}"] = wandb.Image(
                            fig,
                            caption=f"Hashed indices counts at level {level} at epoch {e}"
                        )
                    del histograms

                wandb.log(log)
                del log

            if train_psnr >= best_psnr and should_save_params:
                best_psnr = train_psnr

                os.makedirs(os.path.join(os.path.join(drive_folder, 'weights')), exist_ok=True)

                print("Saving model")
                torch.save(GGNF_model.state_dict(), os.path.join(os.path.join(drive_folder, os.path.join('weights', f"{id_param}_{time}")), f"model_{e}.pt"))
                torch.save(optimizer.state_dict(), os.path.join(os.path.join(drive_folder, os.path.join('weights', f"{id_param}_{time}")), f'opt_{e}.pt'))
                print("="*80)

            if early_stopper.early_stop:
                break

            if e != 0:
                early_stopper(train_loss)
                if early_stopper.early_stop:
                    should_show_counts = True
                    print("!!! Stopping at epoch:", e, "!!!")
                #     break

            del train_img
            del train_loss
            del train_accuracy
            del train_psnr
            del collisions
            del counts_per_level
            del js_kl_div_losses
            del collisions_losses
            del mse_loss
            del indices_per_level

            print_allocated_memory() # End epoch

        wandb.finish()
        del pbar
        torch.cuda.empty_cache()