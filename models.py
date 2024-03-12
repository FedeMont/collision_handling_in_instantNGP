from functions import *

# --- Learnable Hash Function Model --- #
class LearnableHashFunctionModel(nn.Module):
    def __init__(
        self,
        num_levels: Optional[int] = 4,
        input_size: Optional[int] = 2,
        hash_table_size: Optional[int] = 2**14,
        should_calc_collisions: Optional[bool] = False,
        should_static_hash: Optional[bool] = False,
        should_use_all_levels: Optional[bool] = False,
        should_log: Optional[List | bool] = False,
        output_size: Optional[int] = 2,
        sigmas_scale: Optional[float] = 1.0,
        hidden_layers_widths: Optional[List[int]] = [16],
        dropout_rate: Optional[float | None] = None,
        hidden_layers_activation: Optional[nn.Module] = nn.Tanh(),
        weights_initialization: Optional[nn.Module | None] = None,
        prime_numbers: Optional[List] = [1, 2654435761, 805459861],
        device: Optional[torch.device | str] = "cpu",
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        input_size : int, optional (default is 2)
            Input size.
        hash_table_size : int, optional (default is 2**14)
            Hash table size.
        should_calc_collisions : bool, optional (default is False)
            Whether to calculate collisions or not.
        should_static_hash : bool, optional (default is False)
            Whether to use fast hash instead of HashFunction or not.
        should_use_all_levels : bool, optional (default is False)
            Whether to use all levels or only the ones with collisions.
        should_log : List | bool, optional (default is False)
            List of decoded functions to log or False to not log.
        output_size : int, optional (default is 2)
            Output size.
        sigmas_scale : float, optional (default is 1.0)
            Sigmas scale.
        hidden_layers_widths : List[int]
            List of hidden layers widths.
        dropout_rate : float | None, optional (default is None)
            Dropout rate or None for no dropout.
        hidden_layers_activation : nn.Module, optional (default is nn.Tanh())
            Activation function for hidden layers.
        weights_initialization : nn.Module | None, optional (default is None)
            Weights initialization.
        prime_numbers : List, optional (default is [1, 2654435761, 805459861])
            Prime numbers to use when static hash.
        device : torch.device | str, optional (default is "cpu")
            Device to use for Tensors.
        **kwargs

        Returns
        -------
        None
        """
        super(LearnableHashFunctionModel, self).__init__()

        self._num_levels: int = kwargs.get("num_levels", num_levels)
        self._input_size: int = kwargs.get("input_size", input_size)
        self._hash_table_size: int = kwargs.get("hash_table_size", hash_table_size)

        self._should_calc_collisions: bool = kwargs.get("should_calc_collisions", should_calc_collisions)
        self._should_static_hash: bool = kwargs.get("should_static_hash", should_static_hash)
        self._should_log: List | bool = kwargs.get("should_log", should_log)

        self._device: torch.device | str = device

        self._sigmas_eps: float = 1e-10

        if not self._should_static_hash:
            self._output_size: int = kwargs.get("output_size", output_size)
            self._sigmas_scale: float = kwargs.get("sigmas_scale", sigmas_scale)
            dropout_rate: float | None = kwargs.get("dropout_rate", dropout_rate)
            self._dropout: nn.Module | None = nn.Dropout(dropout_rate) if dropout_rate is not None else None

            # TODO Fix this eval in case of using non-kwargs weigths initialization
            self._weights_initialization: nn.Module = eval(kwargs.get("weights_initialization", weights_initialization), {"torch": torch})
            
            self._should_use_all_levels: bool = kwargs.get("should_use_all_levels", should_use_all_levels)

            layers_widths = [self._input_size, *kwargs.get("hidden_layers_widths", hidden_layers_widths), self._output_size]
            self._module_list_per_level: nn.ModuleList = nn.ModuleList([ # one MLP for each level
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(
                            in_features=layers_widths[i], 
                            out_features=layers_widths[i + 1],
                            device=self._device
                        ),
                        (
                            eval(kwargs.get("hidden_layers_activation", hidden_layers_activation), {"torch": torch}) 
                            if (i < len(layers_widths) - 2) 
                            else nn.Sigmoid()
                        )
                    )
                    for i in range(len(layers_widths) - 1)
                ])
                for _ in range(self._num_levels)
            ])

            if self._weights_initialization is not None:
                self._init_weights()
            
        else:
            self._prime_numbers = torch.nn.Parameter(
                torch.from_numpy(
                    np.array(kwargs.get("prime_numbers", prime_numbers)),
                ).to(self._device),
                False
            )

    def forward(
        self, 
        x: torch.Tensor | List[torch.Tensor],
        min_possible_collisions_per_level: torch.Tensor,
        multires_levels: torch.Tensor,
        should_draw_hists: Optional[bool] = False,
        should_show_hists: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hashes the coordinates.

        Parameters
        ----------
        x : torch.Tensor | List[torch.Tensor]
            Images' grid coordinates or list of images' unique grid coordinates.
        min_possible_collisions_per_level : torch.Tensor
            Minimum possible hash collisions for each level.
        multires_levels : torch.Tensor
            Multiresolution levels.
        should_draw_hists : bool, optional (default is False)
            Whether to draw matplotlib histograms or not.
        should_show_hists : bool, optional (default is False)
            Whether to show histograms or not.
        
        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor] | None, List[plt.Figure] | None]
            The hashed coordinates, their uncertainty, the number of collisions and the collisions' histograms divided by level.
        """

        indices, sigmas = (
            self._learnable_hashing_function(x, min_possible_collisions_per_level, multires_levels)
            if not self._should_static_hash
            else self._static_hash(x)
        )

        collisions = None if self._should_calc_collisions else None
        hists = None if should_draw_hists else None
        
        return indices, sigmas, collisions, hists
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                # Applying Xavier initialization to linear and convolutional layers
                self._weights_initialization(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _learnable_hashing_function(
        self, 
        x: torch.Tensor | List[torch.Tensor],
        min_possible_collisions_per_level: torch.Tensor,
        multires_levels: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Hashes the coordinates using learnable hash function.

        Parameters
        ----------
        x : torch.Tensor | List[torch.Tensor]
            Images' grid coordinates or list of images' unique grid coordinates.
        min_possible_collisions_per_level : torch.Tensor
            Minimum possible hash collisions for each level.
        multires_levels : torch.Tensor
            Multiresolution levels.
        
        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor]]
            The hashed coordinates and their uncertainty divided by level.
        """

        if not isinstance(x, list):
            x = [x[:, :, l, :, :] for l in range(self._num_levels)]
        log(("x:", x, len(x), [(x[l].shape, x[l].requires_grad, x[l].is_leaf) for l in range(self._num_levels)]), self._should_log)

        indices, sigmas = [], []
        for l in range(self._num_levels):
            if not self._should_use_all_levels and min_possible_collisions_per_level[l] <= 0:
                # levels with no collisions are directly mapped to the output

                # should be fixed -> Fix in case of dataset->should_normalize_input
                hashed_non_collisions_level = (
                    (multires_levels[l] + 1) *  x[l][..., 0]
                ) + x[l][..., 1]

                indices.append(hashed_non_collisions_level.unsqueeze(-1))
                log((f"Level {l}, indices:", indices[l], indices[l].shape, indices[l].requires_grad, indices[l].is_leaf), self._should_log)

                sigmas.append(torch.zeros((*x[l].shape[:-1], 1), device=self._device) + self._sigmas_eps)
                log((f"Level {l}, sigmas:", sigmas[l], sigmas[l].shape, sigmas[l].requires_grad, sigmas[l].is_leaf), self._should_log)

                continue

            for i, layer in enumerate(self._module_list_per_level[l]):
                x[l] = layer(x[l])
                # log((f"Level {l}, after layer {i}:", x[l], x[l].shape, x[l].requires_grad, x[l].is_leaf), self._should_log)
            
            x[l] = x[l].unsqueeze(-1)
            log((f"Level {l}, after unsqueeze:", x[l], x[l].shape, x[l].requires_grad, x[l].is_leaf), self._should_log)
            
            indices.append(
                differentiable_round(x[l][..., 0, :] * (self._hash_table_size - 1))
                # x[l][..., 0, :] * (self._hash_table_size - 1)
            )
            log((f"Level {l}, indices:", indices[l], indices[l].shape, indices[l].requires_grad, indices[l].is_leaf), self._should_log)
            
            sigmas.append(
                x[l][..., 1, :] * self._sigmas_scale
            )
            log((f"Level {l}, sigmas:", sigmas[l], sigmas[l].shape, sigmas[l].requires_grad, sigmas[l].is_leaf), self._should_log)
        del x

        return indices, sigmas

    @torch.no_grad()
    def _static_hash(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the static hash function proposed by NVIDIA.

        Parameters
        ----------
        x : torch.Tensor
            Grid coordinates to hash of shape (batch, pixels, levels, 2^input_dim, input_dim)
            This tensor should contain the vertices of the hyper cube for each level.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Hashed coordinates and their uncertainty.
        """
        tmp = torch.zeros(
            (x.shape[0], x.shape[1], self._num_levels, 2**self._input_size),
            device=self._device
        ).to(int)

        for i in range(self._input_size):
            tmp = torch.bitwise_xor(
                (x[:, :, :, :, i].to(int) * self._prime_numbers[i]),
                tmp
            )

        hash = torch.remainder(tmp, self._hash_table_size).unsqueeze(-1).float() # TODO: check if self._hash_table_size - 1 is correct
        del tmp

        sigmas: torch.Tensor = torch.zeros_like(
            hash, 
            dtype=torch.float32, 
            device=self._device
        ) + self._sigmas_eps # to prevent division by zero

        return hash, sigmas
# ----------------------- #
 
# --- General Neural Gauge Fields Model --- #
class GNGFModel(nn.Module):
    def __init__(
        self,
        learnable_hash_function_model: LearnableHashFunctionModel,
        batch_size: int,
        feature_size: Optional[int] = 2,
        hidden_layers_widths: List[int] = [64, 64],
        topk: Optional[int] = 1,
        num_levels: Optional[int | None] = None,
        hash_table_size: Optional[int | None] = None,
        should_circular_topk: Optional[bool] = True,
        should_learn_images: Optional[bool] = False,
        should_log: Optional[List | bool] = False,
        device: Optional[torch.device | str] = "cpu",
        **kwargs
    ) -> None:
        """
        
        Parameters
        ----------
        learnable_hash_function_model : LearnableHashFunctionModel
            Learnable hash function model.
        batch_size : int
            Batch size.
        feature_size : int, optional (default is 2)
            Feature size.
        hidden_layers_widths : List[int], optional (default is [64, 64])
            List of hidden layers widths.
        topk : int, optional (default is 1)
            Top k. If -1 then all the hashed coordinates are used.
        num_levels : int | None, optional (default is None)
            Number of levels. If None then MultiresolutionModel's num_levels is used.
        hash_table_size : int | None, optional (default is None)
            Hash table size. If None then MultiresolutionModel's hash_table_size is used.
        should_circular_topk : bool, optional (default is True)
            Whether to use circular topk or not.
        should_learn_images : bool, optional (default is False)
            Whether to learn the images or not.
        should_log : List | bool, optional (default is False)
            List of decoded functions to log or False to not log.
        device : torch.device | str, optional (default is "cpu")
            Device to use for Tensors.
        **kwargs
        
        Returns
        -------
        None
        """
        super(GNGFModel, self).__init__()

        self.learnable_hash_function_model: LearnableHashFunctionModel = learnable_hash_function_model
        
        self._should_learn_images: bool = kwargs.get("should_learn_images", should_learn_images)
        self._should_log: bool = kwargs.get("should_log", should_log)

        self._device: torch.device | str = device
        
        if self._should_learn_images:
            self._batch_size: int = kwargs.get("batch_size", batch_size)
            self._feature_size: int = kwargs.get("feature_size", feature_size)
            self._topk: int = kwargs.get("topk", topk)
            self._num_levels: int = kwargs.get("num_levels", num_levels)
            self._hash_table_size: int = kwargs.get("hash_table_size", hash_table_size)

            self._should_circular_topk: bool = kwargs.get("should_circular_topk", should_circular_topk)

            self.hash_tables: torch.nn.ModuleList = torch.nn.ModuleList([
                torch.nn.ModuleList([
                    torch.nn.Embedding(self._hash_table_size, self._feature_size, device=self._device)
                    for _ in range(self._num_levels)
                ])
                for _ in range(self._batch_size)
            ])
            log(
                (
                    "Hash table image 0 level 0:", 
                    self.hash_tables[0][0].weight, 
                    self.hash_tables[0][0].weight.shape
                ), 
                self._should_log
            )

            self._apply_init(torch.nn.init.uniform_, -1.0, 1.0)
            log(
                (
                    "Initialized hash table image 0 level 0:", 
                    self.hash_tables[0][0].weight, 
                    self.hash_tables[0][0].weight.shape
                ), 
                self._should_log
            )

            self._mlp = None
            layers_widths = [
                (self._num_levels * self._feature_size), 
                *kwargs.get("hidden_layers_widths", hidden_layers_widths), 
                3
            ]
            self.mlp: nn.ModuleList = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(
                        in_features=layers_widths[i], 
                        out_features=layers_widths[i + 1],
                        device=self._device
                    ),
                    nn.ReLU() if (i < (len(layers_widths) - 2)) else nn.Sigmoid()
                )
                for i in range(len(layers_widths) - 1)
            ])

    def forward(
        self, 
        unique_grids_per_level: List[torch.Tensor] | None,
        min_possible_collisions_per_level: torch.Tensor,
        scaled_coords: torch.Tensor,
        grid_coords: torch.Tensor,
        multires_levels: torch.Tensor,
        should_draw_hists: bool = False,
        should_show_hists: bool = False,
    ) -> Tuple[
        torch.Tensor | None, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor | None, 
        List[plt.Figure] | None
    ]:
        
        indices, sigmas, collisions, hists = self.learnable_hash_function_model(
            (
                unique_grids_per_level 
                if (not self._should_learn_images and unique_grids_per_level is not None)
                else grid_coords
            ),
            min_possible_collisions_per_level,
            multires_levels,
            should_draw_hists, 
            should_show_hists
        )

        if (self._should_learn_images or unique_grids_per_level is None):
            if isinstance(indices, list):
                indices = torch.stack(indices, dim=2)
                log(("indices:", indices, indices.shape, indices.requires_grad, indices.is_leaf), self._should_log)
            
            if isinstance(sigmas, list):
                sigmas = torch.stack(sigmas, dim=2)
                log(("sigmas:", sigmas, sigmas.shape, sigmas.requires_grad, sigmas.is_leaf), self._should_log)

        out = None
        if self._should_learn_images:  
            coeffs = self._calc_bilinear_coefficients(scaled_coords, grid_coords)

            out = self._look_up_features(indices, sigmas)

            out = self._bilinear_interpolation(out, coeffs)
            del coeffs

            for i, layer in enumerate(self.mlp):
                out = layer(out)
                log((f"After layer {i}:", out, out.shape, out.requires_grad, out.is_leaf), self._should_log)

        return out, indices, sigmas, collisions, hists
    
    @torch.no_grad()
    def _calc_bilinear_coefficients(self, scaled_coords: torch.Tensor, grid_coords: torch.Tensor) -> torch.Tensor:
        """
        
        Parameters
        ----------
        scaled_coords : torch.Tensor
            Scaled coordinates.
        grid_coords : torch.Tensor
            Grid coordinates.
        
        Returns
        -------
        torch.Tensor
            Coefficients for bilinear interpolation.
        """

        log(("SCALED COORDS:", scaled_coords[0, :, :, 0, :], scaled_coords.shape), self._should_log)
        log(("GRID COORDS:", grid_coords[0, :, 0, :, :], grid_coords.shape), self._should_log)

        _as: torch.Tensor = grid_coords[:, :, :, 0, :].unsqueeze(-2)  # bottom-right vertices of cells
        _ds: torch.Tensor = grid_coords[:, :, :, -1, :].unsqueeze(-2)  # top-left vertices of cells

        log(("_as:", _as, _as.shape), self._should_log)
        log(("_ds:", _ds, _ds.shape), self._should_log)

        coeffs: torch.Tensor = torch.stack([
            (_ds[:, :, :, :, 0] - scaled_coords[:, :, 0, :, :]) * (_ds[:, :, :, :, 1] - scaled_coords[:, :, 1, :, :]),  # (xd - x) * (yd - y)
            (scaled_coords[:, :, 0, :, :] - _as[:, :, :, :, 0]) * (_ds[:, :, :, :, 1] - scaled_coords[:, :, 1, :, :]),  # (x - xa) * (yd - y)
            (_ds[:, :, :, :, 0] - scaled_coords[:, :, 0, :, :]) * (scaled_coords[:, :, 1, :, :] - _as[:, :, :, :, 1]),  # (xd - x) * (y - ya)
            (scaled_coords[:, :, 0, :, :] - _as[:, :, :, :, 0]) * (scaled_coords[:, :, 1, :, :] - _as[:, :, :, :, 1]),  # (x - xa) * (y - ya)
        ], dim=-1).squeeze(-2).unsqueeze(2)
        
        del _as
        del _ds
        del scaled_coords
        del grid_coords

        log(("COEFFS:", coeffs, coeffs.shape), self._should_log)

        return coeffs

    def _look_up_features(self, indices: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Looks up features from the hash tables.

        Parameters
        ----------
        indices : torch.Tensor
            Hashed coordinates.
        sigmas : torch.Tensor
            Hashed coordinates' uncertainty.

        Returns
        -------
        torch.Tensor
            Looked up features.
        """
        log(("indices:", indices, indices.shape, indices.requires_grad, indices.is_leaf), self._should_log)
        log(("sigmas:", sigmas, sigmas.shape, sigmas.requires_grad, sigmas.is_leaf), self._should_log)

        if self._topk == -1: # use all hashed coordinates
            topks = torch.arange(
                self._hash_table_size,
                dtype=torch.float32,
                device=self._device
            )
        else:
            topks = torch.arange(
                -(self._topk // 2), (self._topk // 2) + 1,
                dtype=torch.float32,
                device=self._device
            )
        log(("topks:", topks, topks.shape, topks.requires_grad, topks.is_leaf), self._should_log)

        # indices = indices + topks
        # if self._should_circular_topk: # CIRCULAR IMPLEMENTATION
        #     indices = torch.remainder(indices, self._hash_table_size)
        # else: # LINEAR IMPLEMENTATION
        #     indices[indices < 0] = 0
        #     indices[indices >= self._hash_table_size] = self._hash_table_size - 1
        # log(("new_indices:", indices, indices.shape, indices.requires_grad, indices.is_leaf), self._should_log)
        
        # problema con self.hash_tables embeddings: 
        # RuntimeError: function DifferentiableIndexingBackward returned a gradient different than None at position 1, but the corresponding forward input was not a Variable
        # looked_up: torch.Tensor = DifferentiableIndexing.apply(self.hash_tables, indices, Hash_Tables_Indexing())
        
        # looked_up: torch.Tensor = rearrange(
        #     torch.stack([
        #         torch.stack([
        #             self.hash_tables[b][l](
        #                 x[:, l, :, :].int() # Problema molto più probabilemente qua, deve essere int per fare indexing ma facendo int si stacca il gradiente
        #             )
        #             for l in range(self._num_levels)
        #         ])
        #         for b, x in enumerate(indices)
        #     ]),
        #     "batch levels pixels verts k features -> batch pixels levels features verts k"
        # )

        looked_up: torch.Tensor = rearrange(
            torch.stack([
                torch.stack([
                    DifferentiableIndexing.apply(self.hash_tables[b][l].weight, indices[b, :, l, :, :], Hash_Tables_Indexing())
                    for l in range(self._num_levels)
                ])
                for b in range(indices.shape[0])
            ]),
            "batch levels pixels verts k features -> batch pixels levels features verts k"
        )
        
        log(("looked_up:", looked_up, looked_up.shape, looked_up.requires_grad, looked_up.is_leaf), self._should_log)
        del indices

        # Calculate the Gaussian probabilities
        gaussian_probs = (
            torch.exp(-(1/2) * ((topks - 0) / (sigmas))**2) 
            / 
            ((sigmas) * torch.sqrt(2 * torch.tensor(np.pi)))
        ).unsqueeze(3)
        log(("gaussian_probs:", gaussian_probs, gaussian_probs.shape, gaussian_probs.requires_grad, gaussian_probs.is_leaf), self._should_log, color=bcolors.WARNING)
        del topks

        # (weighted avg) sum(looked_up * topk)/sum(topk)
        looked_up = rearrange(
            torch.sum(looked_up * gaussian_probs, dim=-1) 
            / 
            torch.sum(gaussian_probs, dim=-1),
            "batch pixels levels features verts -> batch pixels features levels verts"
        )
        log(("Weighted avg looked_up:", looked_up, looked_up.shape, looked_up.requires_grad, looked_up.is_leaf), self._should_log)
        del gaussian_probs

        return looked_up

    def _bilinear_interpolation(self, features: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Bilinear interpolate features with coefficients.

        Parameters
        ----------
        features : torch.Tensor
            Features to interpolate.
        coeffs : torch.Tensor
            Coefficients for bilinear interpolation.

        Returns
        -------
        torch.Tensor
            Interpolated features.
        """
        log(("coeffs:", coeffs, coeffs.shape), self._should_log)
        log(("features:", features, features.shape), self._should_log)

        weighted_features: torch.Tensor = features * coeffs
        del features
        del coeffs
        log(("weighted_features:", weighted_features, weighted_features.shape), self._should_log)

        weighted_summed_features: torch.Tensor = torch.sum(weighted_features, dim=-1)
        del weighted_features
        log(("weighted_summed_features:", weighted_summed_features, weighted_summed_features.shape), self._should_log)
        
        stack: torch.Tensor = rearrange(weighted_summed_features, "batch pixels features levels -> batch pixels (levels features)")#.to(device)
        del weighted_summed_features
        log(("stacked:", stack, stack.shape), self._should_log)

        return stack
    
    def _apply_init(self, init_func, *args) -> None:
        """
        Initializes the hash tables weights with a random uniform function
        """
        for b in range(self._batch_size):
            for i in range(self._num_levels):
                init_func(self.hash_tables[b][i].weight, *args)
# ----------------------------- #

