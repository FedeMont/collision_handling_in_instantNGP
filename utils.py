from functions import *
from models import *

# --- Apply rules to configuration --- #
def config_apply_rules(config: Dict[str, Any]) -> Dict[str, Any]:
    log(("Applying rules to config...", ), True, color=bcolors.HEADER)
    config_copy = deepcopy(config)

    config_copy["time"] = eval(config["time"], {"datetime": datetime, "ZoneInfo": ZoneInfo})

    if config["flags"]["should_fast_hash"] is True: # should_fat_hash
        config_copy["flags"]["should_learn_images"] = True

        for key, value in config["train_params"]["learnable_hash_model"].items():
            if key not in ["prime_numbers", "should_calc_collisions", "should_use_all_levels", "should_normalize_grid_coords"]:
                del config_copy["train_params"]["learnable_hash_model"][key]

        config_copy["train_params"]["learnable_hash_model"]["should_use_all_levels"] = True
        config_copy["train_params"]["learnable_hash_model"]["should_normalize_grid_coords"] = False
    
        config_copy["train_params"]["gngf_model"]["topk"] = 1

        config_copy["train_params"]["loss"]["lambda_kl_div"] = 0
        config_copy["train_params"]["loss"]["lambda_sigmas"] = 0
        config_copy["train_params"]["loss"]["lambda_reg"] = 0
        config_copy["train_params"]["loss"]["should_give_different_level_importance"] = False
        
    else: # not should_fat_hash
        del config_copy["train_params"]["learnable_hash_model"]["prime_numbers"]

        if config["train_params"]["learnable_hash_model"]["scheduler"]["step_size"] == -1:
            config_copy["train_params"]["learnable_hash_model"]["scheduler"]["step_size"] = config["train_params"]["epochs"]

        if config["train_params"]["learnable_hash_model"]["scheduler"]["stop_epoch"] == -1:
            config_copy["train_params"]["learnable_hash_model"]["scheduler"]["stop_epoch"] = config["train_params"]["epochs"]

    if config["train_params"]["epochs"] < 100:
        config_copy["flags"]["should_wandb"] = False
    if config["train_params"]["dataset"]["should_normalize_input"] is True:
        config_copy["train_params"]["dataset"]["should_normalize_grid_coords"] = False

    if config["flags"]["should_learn_images"] is False: # should_learn_images
        for key, value in config["train_params"]["gngf_model"].items():
            del config_copy["train_params"]["gngf_model"][key]
        
        config_copy["train_params"]["loss"]["lambda_images"] = 0

    if config["train_params"]["early_stopper"]["tolerance"] == -1:
        config_copy["train_params"]["early_stopper"]["tolerance"] = config["train_params"]["epochs"]
    
    if config["wandb"]["name"] is None:
        config_copy["wandb"]["name"] = config_copy["time"]

    if config["flags"]["should_continue_training"] is True:
        assert (
            config["train_params"]["load_weights_path"] is not None
        ), "To continue the training 'load_weights_path' must be provided."

    if config["flags"]["should_use_pretrained"] is True:
        assert (
            config["train_params"]["load_weights_path"] is not None
        ), "To use pretrained weights 'load_weights_path' must be provided."
    
    return config_copy
# ----------------------- #


# --- Loss --- #
class Loss(nn.Module):
    def __init__(
        self,
        hash_table_size: Optional[int] = 2**14,
        should_use_all_levels: Optional[bool] = False,
        should_log: Optional[List | bool] = False,
        kl_div_reduction: Optional[str] = "batchmean",
        excluded_params_from_reg_loss: Optional[List[str]] = ["_prime_numbers", "hash_tables", "mlp"],
        norm_regularization_order: Optional[int] = 2,
        lambda_kl_div: Optional[float] = 1,
        lambda_sigmas: Optional[float] = 1,
        lambda_images: Optional[float] = 1,
        lambda_reg: Optional[float] = 1,
        should_exp_normalize_kl_div: Optional[bool] = False,
        should_give_different_level_importances: Optional[bool] = False,
        device: Optional[torch.device | str] = "cpu",
        **kwargs
    ) -> None:
        """
        
        Parameters
        ----------
        hash_table_size : int, optional (default is 2**14)
            Hash table size.
        should_use_all_levels : bool, optional (default is False)
            Whether to use all levels or only the ones with collisions.
        should_log : List | bool, optional (default is False)
            List of decoded functions to log or False to not log.
        kl_div_reduction : str, optional (default is "batchmean")
            KL divergence reduction.
        excluded_params_from_reg_loss : List[str], optional (default is ["_prime_numbers", "hash_tables", "mlp"])
            List of parameters to exclude from the regularization loss.
        norm_regularization_order : int, optional (default is 2)
            Order of the norm for the regularization loss.
        lambda_kl_div : float, optional (default is 1)
            Weight for the KL divergence loss.
        lambda_sigmas : float, optional (default is 1)
            Weight for the sigmas loss.
        lambda_images : float, optional (default is 1)
            Weight for the images loss.
        lambda_reg : float, optional (default is 1)
            Weight for the regularization loss.
        should_exp_normalize_kl_div : bool, optional (default is False)
            Whether to normalize the KL divergence or not.
        should_give_different_level_importances : bool, optional (default is False)
            Whether to give different importance to different levels or not.
        device : torch.device | str, optional (default is "cpu")
            Device to use for Tensors.
        **kwargs

        Returns
        -------
        None
        """
        super(Loss, self).__init__()

        self._hash_table_size: int = kwargs.get("hash_table_size", hash_table_size)
        self._should_use_all_levels: bool = kwargs.get("should_use_all_levels", should_use_all_levels)
        self._should_log: List | bool = kwargs.get("should_log", should_log)

        self._hist_sanitization_eps: float = 1e-10
        self._excluded_params_from_reg_loss: List[str] = kwargs.get("excluded_params_from_reg_loss", excluded_params_from_reg_loss)
        self._norm_regularization_order: int = kwargs.get("norm_regularization_order", norm_regularization_order)
        
        self._lambda_kl_div: float = eval(kwargs.get("lambda_kl_div", str(lambda_kl_div)))
        self._lambda_sigmas: float = eval(kwargs.get("lambda_sigmas", str(lambda_sigmas)))
        self._lambda_images: float = eval(kwargs.get("lambda_images", str(lambda_images)))
        self._lambda_reg: float = eval(kwargs.get("lambda_reg", str(lambda_reg)))

        self._should_exp_normalize_kl_div: bool = kwargs.get("should_exp_normalize_kl_div", should_exp_normalize_kl_div)
        self._should_give_different_level_importances: bool = kwargs.get("should_give_different_level_importances", should_give_different_level_importances)

        self._device: torch.device = device

        self._KLDiv: nn.KLDivLoss = nn.KLDivLoss(reduction=kwargs.get("kl_div_reduction", kl_div_reduction))
        self._MSE: nn.MSELoss = nn.MSELoss(reduction="mean")

    def forward(
        self,
        indices: torch.Tensor | List[torch.Tensor],
        sigmas: torch.Tensor | List[torch.Tensor],
        min_possible_collisions_per_level: torch.Tensor,
        kl_div_targets_per_level: torch.Tensor, 
        images_pred: torch.Tensor, 
        images_target: torch.Tensor,
        model_named_parameters: Dict[str, torch.Tensor | None],
        should_draw_hists: bool = False,
    ) -> Dict[str, torch.Tensor | List[plt.Figure] | None]:
        
        levels: int = min_possible_collisions_per_level.shape[0]
        
        kl_div_losses: torch.Tensor = torch.zeros(levels, device=self._device)
        histograms: List[plt.Figure] = [None for _ in range(levels)]
        for l in range(levels):
            p = self._calc_hist_pdf(indices[l])

            kl_div_level_loss = self._KLDiv(
                p.log(), 
                kl_div_targets_per_level[l]
            )
            
            if should_draw_hists:
                fig = self._plot_histograms(
                    p=p.detach().cpu().numpy(), 
                    q=kl_div_targets_per_level[l].detach().cpu().numpy(),
                    level=l,
                    should_show=(3 in self._should_log) if self._should_log else self._should_log
                )

                histograms[l] = fig

            # not using all levels and level has no collisions -> skip to next level
            if not self._should_use_all_levels and min_possible_collisions_per_level[l] <= 0:
                del kl_div_level_loss
                continue

            kl_div_losses[l] = kl_div_level_loss if not self._should_exp_normalize_kl_div else (1 - torch.exp(-kl_div_level_loss))
        log(("kl_div_losses:", kl_div_losses, kl_div_losses.shape, kl_div_losses.requires_grad, kl_div_losses.is_leaf), self._should_log)
        
        sigmas_losses: torch.Tensor = torch.stack([
            self._MSE(sigmas[l], torch.zeros_like(sigmas[l]))
            for l in range(levels)
        ])
        log(("sigmas_losses:", sigmas_losses, sigmas_losses.shape, sigmas_losses.requires_grad, sigmas_losses.is_leaf), self._should_log)
        
        images_losses: torch.Tensor | None = None
        if images_pred is not None:
            images_losses: torch.Tensor = torch.stack([
                self._MSE(images_pred[b], images_target[b])
                for b in range(images_pred.shape[0])
            ])
            log(("images_losses:", images_losses, images_losses.shape, images_losses.requires_grad, images_losses.is_leaf), self._should_log)

        reg_loss: torch.Tensor = torch.tensor(0.0, device=self._device)
        for name, param in model_named_parameters:
            if not any([excluded_param in name for excluded_param in self._excluded_params_from_reg_loss]):
                reg_loss += torch.linalg.norm(param, ord=self._norm_regularization_order)

        level_importances: torch.Tensor = torch.ones(levels, device=self._device)
        if self._should_give_different_level_importances:
            level_importances = torch.exp(torch.arange(levels, 0, step=-1, device=self._device))

        loss: torch.Tensor = (
            # (self._lambda_kl_div * torch.sum(level_importances * kl_div_losses)) +
            # (self._lambda_sigmas * torch.sum(level_importances * sigmas_losses)) +
            (
                self._lambda_images * torch.sum(images_losses)
                if images_losses is not None
                else 0.0
            ) +
            (self._lambda_reg * reg_loss)
        )

        return {
            "kl_div_losses": kl_div_losses,
            "sigmas_losses": sigmas_losses,
            "images_losses": images_losses,
            "reg_loss": reg_loss,
            "loss": loss,
            "hists": histograms
        }

    def _calc_hist_pdf(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the histogram pdf.

        Parameters
        ----------
        indices : torch.Tensor
            Hashed coordinates.

        Returns
        -------
        torch.Tensor
            Histogram pdf.
        """

        hist_p = differentiable_histogram(
            indices, 
            bins=self._hash_table_size, 
            min=0, 
            max=self._hash_table_size, 
            should_log=self._should_log
        ).squeeze(0).squeeze(0)
        log(
            (
                "hist_p_diff:", hist_p.long(), 
                "shape:", hist_p.shape, 
                "sum:", torch.sum(hist_p), 
                "requires_grad:", hist_p.requires_grad, 
            ), 
                self._should_log
        )

        # Real implemenation of histogram pdf, but not differentiable
        # hist_p_nondiff = torch.histc(indices, bins=self._hash_table_size, min=0, max=self._hash_table_size) # ? maybe max=(self._hash_table_size - 1)
        # log((f"hist_p_nondiff: {hist_p_nondiff}, shape: {hist_p_nondiff.shape}, sum: {torch.sum(hist_p_nondiff)}, requires_grad: {hist_p_nondiff.requires_grad}", ), self._should_log, color=bcolors.WARNING)
        # log((f"hist_p_diff - hist_p_nondiff: {hist_p - hist_p_nondiff}", ), self._should_log, color=bcolors.WARNING)
        del indices

        hist_p = hist_p + (torch.sum(hist_p) * self._hist_sanitization_eps)

        p = hist_p / torch.sum(hist_p)
        log(("p:", p, p.shape, p.requires_grad, p.is_leaf, p.sum().item()), self._should_log)

        del hist_p

        return p

    def _plot_histograms(self, p: np.ndarray, q: np.ndarray, level: int, should_show: bool = False) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(np.arange(len(p)), p, alpha=1, label="p")
        ax.bar(np.arange(len(q)), q, alpha=0.5, label="q")

        ax.set_xlim(-1, self._hash_table_size)
        ax.xaxis.set_ticks(np.arange(0, self._hash_table_size, 10))

        plt.title(f"Level {level}")
        plt.xlabel("Hashed coordinates")
        plt.ylabel("Normalized collisions histogram")
        plt.legend()

        # start, end = ax.get_ylim()
        # step = int(end * 0.1)
        # ax.yaxis.set_ticks(np.arange(0, end, step if step > 0 else 1))

        if should_show:
            plt.show()
        plt.close()

        return fig
# ----------------------- #


# --- Optimizer --- #
def get_optimizer(
    models_parameters: Dict[str, Any],
    optimizers: List[torch.optim.Optimizer] = [torch.optim.Adam, torch.optim.AdamW],
) -> Dict[str, torch.optim.Optimizer]:
    
    optims = {}
    
    for model, optimizer in zip(models_parameters.items(), optimizers):
        models_params = []

        for m in model[1]["each"]:
            param, lr, weight_decay= m.values()

            models_params.append({
                "params": param, "lr": lr, "weight_decay": weight_decay
            })
        
        betas = model[1]["betas"]
        eps = model[1]["eps"]

        optims[model[0]] = optimizer(
            models_params,
            betas=betas,
            eps=eps,
        )
    
    return optims
# ----------------------- #


# --- Scheduler --- #
# TODO try to find a way to let use all schedulers
def get_scheduler(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    optimizer: torch.optim.Optimizer,
    step_size: int,
    gamma_eta_min: float
) -> torch.optim.lr_scheduler._LRScheduler:
    
    if scheduler == "CosineAnnealingLR":
        to_return_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=step_size, 
            eta_min=gamma_eta_min
        )
    elif scheduler == "CosineAnnealingWarmRestarts":
        to_return_scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=step_size,
            T_mult=1, 
            eta_min=gamma_eta_min
        )
    elif scheduler == "StepLR":
        to_return_scheduler = StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma_eta_min
        )

    return to_return_scheduler
# ----------------------- #


# --- Early stopper --- #
class EarlyStopper:
    def __init__(self, tolerance: int = 5, min_delta: int = 0, should_reset: bool = True):
        self.tolerance: int = tolerance
        self.min_delta: int = min_delta
        self.best_loss: float = np.inf
        self.counter: int = 0
        self.early_stop: bool = False
        self._should_reset: bool = should_reset

    def __call__(self, loss):
        if abs(self.best_loss - loss) < self.min_delta and (loss < self.best_loss):
            self.counter += 1
        elif abs(self.best_loss - loss) > self.min_delta and (loss > self.best_loss):
            self.counter += 1
        else:
            if not self._should_reset:
                if self.counter <= 0:
                    self.counter = 0
                else:
                    self.counter -= 1
            else:
                self.counter = 0
                self.best_loss = loss

        if self.counter >= self.tolerance:
            self.early_stop = True
# ----------------------- #


# --- Load checkpoint --- #
def load_checkpoint(
    config: Optional[Dict[str, Any]],
    model: GNGFModel,
    optimizers: Dict[str, torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    weights_path: str | None = None,
    should_continue_training: Optional[bool] = False,
    should_use_pretrained: Optional[bool] = False,
    should_log: Optional[bool] = False,
) -> Tuple[
        Dict[str, Any] | None, # loaded configuration
        GNGFModel | None, # loaded model
        Dict[str, torch.optim.Optimizer] | None, # loaded optimizers
        torch.optim.lr_scheduler._LRScheduler | None, # loaded scheduler
    ]:

    loaded_config = config
    loaded_model = model
    loaded_optimizers = optimizers
    loaded_scheduler = scheduler
    if weights_path is not None:
        checkpoint = torch.load(weights_path)
        
        # loaded_config = checkpoint["configuration"]

        loaded_config["wandb"]["name"] = checkpoint["run_name"] + (
            "_continued" 
            if should_continue_training 
            else 
            (
                "_pretrained"
                if should_use_pretrained
                else ""
            )
        )

        if should_continue_training:
            loaded_config["train_params"]["start_epoch"] = checkpoint["epoch"] + 1
            loaded_config["train_params"]["epochs"] = (checkpoint["epoch"] + config["train_params"]["epochs"] + 1) 

        if not (
            loaded_config["flags"]["should_fast_hash"] 
            and 
            loaded_config["flags"]["should_learn_images"]
        ):
            hashModel_state_dict = {}
            for key, value in checkpoint["model_state_dict"].items():
                hashModel_state_dict[key.replace("learnable_hash_function_model.", "")] = value

            model.learnable_hash_function_model.load_state_dict(hashModel_state_dict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model = model

        if should_continue_training:
            if "optimizer_NeRF_state_dict" in checkpoint.keys():
                optimizers["NeRF"].load_state_dict(checkpoint["optimizer_NeRF_state_dict"])
            if "optimizer_learnable_hash_state_dict" in checkpoint.keys():
                optimizers["learnable_hash"].load_state_dict(checkpoint["optimizer_learnable_hash_state_dict"])
            loaded_optimizers = optimizers

        if (
            "scheduler" in loaded_config["train_params"]["learnable_hash_model"].keys()
            and (
                checkpoint["configuration"]["train_params"]["learnable_hash_model"]["scheduler"]["name"]
                if should_continue_training and not should_use_pretrained
                else loaded_config["train_params"]["learnable_hash_model"]["scheduler"]["name"]
            ) is not None
        ):
            loaded_scheduler = get_scheduler(
                scheduler=(
                    checkpoint["configuration"]
                    if should_continue_training and not should_use_pretrained
                    else loaded_config
                )["train_params"]["learnable_hash_model"]["scheduler"]["name"],
                optimizer=optimizers["learnable_hash"],
                step_size=(
                    checkpoint["configuration"]
                    if should_continue_training and not should_use_pretrained
                    else loaded_config
                )["train_params"]["learnable_hash_model"]["scheduler"]["step_size"],
                gamma_eta_min=eval((
                    checkpoint["configuration"]
                    if should_continue_training and not should_use_pretrained
                    else loaded_config
                )["train_params"]["learnable_hash_model"]["scheduler"]["gamma_eta_min"]),
            )
            if should_continue_training:
                loaded_scheduler.last_epoch = checkpoint["epoch"] # this states the last epoch of the scheduler
            # this states how many epochs the scheduler should have done, do not overwrite it
            # loaded_config["train_params"]["learnable_hash_model"]["scheduler"]["last_epoch"] = checkpoint["epoch"] 

        # loaded_config["flags"]["should_wandb"] = config["train_params"]["epochs"] >= 100
        loaded_config["flags"]["should_continue_training"] = should_continue_training
        loaded_config["flags"]["should_use_pretrained"] = should_use_pretrained

        log(("Weights loaded", checkpoint), should_log)
    
    return loaded_config, loaded_model, loaded_optimizers, loaded_scheduler
# ----------------------- #
