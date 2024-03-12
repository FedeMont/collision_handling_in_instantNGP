from functions import *

# --- Image Dataset --- #
class ImageDataset(Dataset):
    def __init__(
        self,
        images_paths: List[str],
        n_min: int,
        n_max: int,
        num_levels: int,
        hash_table_size: Optional[int] = 2**14,
        input_indexing: Optional[str] = "ij",
        should_randomize_input: Optional[bool] = False,
        should_normalize_input: Optional[bool] = False,
        should_use_all_levels: Optional[bool] = False,
        should_log: Optional[List | bool] = False,
        device: Optional[torch.device | str] = "cpu",
        **kwargs
    ):
        """
        
        Parameters
        ----------
        images_paths : List[str]
            List of paths to images.
        n_min : int
            Minimum number of levels.
        n_max : int
            Maximum number of levels.
        num_levels : int
            Number of levels.
        hash_table_size : int, optional (default is 2**14)
            Size of the hash table.
        input_indexing : str, optional (default is "ij")
            Indexing of the input.
        should_randomize_input : bool, optional (default is False)
            Should randomize the images.
        should_normalize_input : bool, optional (default is False)
            Should normalize the images' coordinates between [-1, 1].
        should_use_all_levels : bool, optional (default is False)
            Whether to use all levels or not.
        should_log : List[int], optional (default is [])
            List of decoded functions to log.,
        device : torch.device | str, optional (default is "cpu")
            Device to use for Tensors.
        **kwargs
        
        Returns
        -------
        None
        """
        super(ImageDataset, self).__init__()

        self._images_paths: List[str] = kwargs.get("images_paths", images_paths)
        self._n_min: int = kwargs.get("n_min", n_min)
        self._n_max: int = kwargs.get("n_max", n_max)
        self._num_levels: int = kwargs.get("num_levels", num_levels)
        self._hash_table_size: int = eval(kwargs.get("hash_table_size", str(hash_table_size)))

        self._should_randomize_input: bool = kwargs.get("should_randomize_input", should_randomize_input)
        self._should_normalize_input: bool = kwargs.get("should_normalize_input", should_normalize_input)
        self._should_use_all_levels: bool = kwargs.get("should_use_all_levels", should_use_all_levels)
        self._should_log: bool | List = kwargs.get("should_log", should_log)

        self._device: torch.device | str = device

        # --- Read images --- #
        # TODO pad images if necessary
        self._images: torch.Tensor = torch.stack(
            [
                io.read_image(path).to(self._device) # rgb h w
                for path in self._images_paths
            ]
        )
        log(("Images:", self._images.shape), self._should_log)

        self._dimensions = torch.tensor(self._images.shape[2:], device=self._device) # first two dimensions are batch and channels
        log(("Dimensions:", self._dimensions), self._should_log)
        # ------------------- #

        # --- Ground Truth --- #
        self._Y: torch.Tensor = (
            rearrange(
                self._images.unsqueeze(-1) if len(self._images.shape) <= 4 else self._images, 
                "b c h w d -> b (h w d) c"
            ).float() / 255.0
        ).to(self._device)
        log(("Y:", self._Y.shape), self._should_log)
        # -------------------- #

        # --- Prepare Levels --- #
        self._levels, self._voxels_helper_hypercube = self._get_levels(
            n_min=self._n_min, 
            n_max=self._n_max, 
            num_levels=self._num_levels, 
            input_size=len(self._dimensions), 
        )

        # --- Create input --- #
        x, self._min_possible_collisions_per_level, max_input_values_per_level = self._create_input(
            dimensions=self._dimensions,
            input_indexing=kwargs.get("input_indexing", input_indexing),
        )

        # --- Scale to Grid --- #
        self._scaled_X, self._grid_X = self._scale_to_grid(
            x=repeat(x, "pixels dim num_levels 1 -> batch pixels dim num_levels 1", batch=len(self)),
            levels=self._levels,
            voxels_helper_hypercube=self._voxels_helper_hypercube,
        )

        self._unique_grids_per_level: List[torch.Tensor] = [
            torch.unique(
                rearrange(
                    self._grid_X[0, :, l, :, :], # 0 because the images have the same dimensions
                    "pixels verts dim -> (pixels verts) dim"
                ), 
                dim=0
            ) for l in range(self._num_levels)
        ]
        log(("unique_grids_per_level:", self._unique_grids_per_level, len(self._unique_grids_per_level), [self._unique_grids_per_level[l].shape for l in range(len(self._unique_grids_per_level))]), self._should_log)
        # -------------------- #

        # --- Create KL Div target --- #
        self._kl_div_targets: torch.Tensor = self._create_kl_div_targets(
            max_input_values_per_level
        )
        # ---------------------------- #

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx == -1:
            idx = torch.arange(len(self), device=self._device)
        
        if not torch.is_tensor(idx):
            idx = torch.tensor([idx], device=self._device)

        return {
            "unique_grids_per_level": deepcopy(self._unique_grids_per_level), # TODO WHY do we need this?
            "min_possible_collisions_per_level": self._min_possible_collisions_per_level,
            "kl_div_targets_per_level": self._kl_div_targets, #[idx],
            "scaled_X": self._scaled_X[idx],
            "grid_X": self._grid_X[idx],
            "Y": self._Y[idx],
            "multires_levels": self._levels.squeeze(),
            "dimensions": self._dimensions,
        }

    def __len__(self) -> int:
        return len(self._images_paths)
    
    def _create_input(
        self,
        dimensions: torch.Tensor,
        input_indexing: str = "ij",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Parameters
        ----------
        dimensions : torch.Tensor
            Dimensions of the input.
        input_indexing : str, optional (default is "ij")
            Indexing of the input.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Input pixel's coordinates and minimum possible collisions per level.
        """
        
        x: torch.Tensor = torch.tensor(
            np.stack(
                np.meshgrid(
                    *[range(dim) for dim in dimensions], 
                    indexing=input_indexing
                ), 
                axis=-1
            ).reshape(-1, 2),
            device=self._device
        )
        log(("x:", x, x.shape), self._should_log)

        add_offset = torch.zeros_like(dimensions)
        if torch.unique(dimensions).shape[0] > 1:
            add_offset = add_offset.scatter_(0, torch.argmin(dimensions, dim=-1), 1)
        
        max_values_per_level = torch.floor(
            x[-1] / max(dimensions - 1) * self._levels.squeeze().unsqueeze(-1)
        ) + add_offset

        min_possible_collisions_per_level = torch.prod(max_values_per_level + 1, dim=-1) - self._hash_table_size
        min_possible_collisions_per_level[min_possible_collisions_per_level < 0] = 0

        x = repeat(x, "pixels dim -> pixels dim num_levels", num_levels=self._num_levels)#.unsqueeze(-1)
        log(("x:", x, x.shape), self._should_log)

        center_normalization = torch.zeros(self._num_levels, dimensions.shape[0], device=self._device)
        scale_normalization = torch.ones(self._num_levels, device=self._device)
        for l in range(self._num_levels):
            scale_normalization[l] = scale_normalization[l] * max(dimensions)

            if not self._should_use_all_levels and min_possible_collisions_per_level[l] <= 0:
                continue

            if self._should_normalize_input:
                center_normalization[l] = (dimensions - 1) / 2
                scale_normalization[l] = scale_normalization[l] / 2

        log(("center_normalization:", center_normalization, center_normalization.shape), self._should_log)
        log(("scale_normalization:", scale_normalization, scale_normalization.shape), self._should_log)

        center_normalization = rearrange(center_normalization, "num_levels dim -> dim num_levels")
        x = ((x - center_normalization) / scale_normalization).unsqueeze(-1)
        log(("x:", x, x.shape), self._should_log)

        return x, min_possible_collisions_per_level, max_values_per_level
    
    def _get_levels(
        self, 
        n_min: int, 
        n_max: int, 
        num_levels: int,
        input_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        n_min : int
            Minimum number of levels.
        n_max : int
            Maximum number of levels.
        num_levels : int
            Number of levels.
        input_size : int
            Input size.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Levels and voxels helper hypercube.
        """

        b: torch.Tensor = torch.tensor(
            np.exp((np.log(n_max) - np.log(n_min)) / (num_levels - 1)),
            device=self._device
        ).float()
        if b > 2 or b <= 1:
            print(
                f"The between level scale is recommended to be <= 2 and needs to be > 1 but was {b:.4f}."
            )
        
        levels: torch.Tensor = torch.stack([
            torch.floor(n_min * (b ** l)) for l in range(self._num_levels)
        ]).reshape(1, 1, -1, 1)
        log(("Levels:", levels, levels.shape, levels.requires_grad, ), self._should_log)

        voxels_helper_hypercube: torch.Tensor = rearrange(
            torch.tensor(
                np.stack(
                    np.meshgrid(
                        range(2), range(2), range(input_size - 1), 
                        indexing="ij"
                    ), 
                    axis=-1
                ),
                device=self._device
            ),
            "cols rows depths verts -> (depths rows cols) verts"
        ).T[:input_size, :].unsqueeze(0).unsqueeze(2)
        log(
            (
                "voxels_helper_hypercube:", 
                voxels_helper_hypercube, 
                voxels_helper_hypercube.shape, 
                voxels_helper_hypercube.requires_grad
            ), 
            self._should_log
        )
        
        return levels, voxels_helper_hypercube

    def _scale_to_grid(
        self,
        x: torch.Tensor,
        levels: torch.Tensor,
        voxels_helper_hypercube: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Parameters
        ----------
        x : torch.Tensor
            Input.
        levels : torch.Tensor
            Levels.
        voxels_helper_hypercube : torch.Tensor
            Voxel helper hypercube.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Scaled coordinates and scaled grid coordinates.
        """

        scaled_coords: torch.Tensor = x * levels # batch pixels dim num_levels 1
        log(("scaled_coords:", scaled_coords, scaled_coords.shape, scaled_coords.requires_grad), self._should_log)

        grid_coords: torch.Tensor = rearrange(
            torch.add(
                torch.floor(scaled_coords),
                voxels_helper_hypercube
            ),
            "batch pixels dim num_levels verts -> batch pixels num_levels verts dim"
        )
        log(("grid_coords:", grid_coords, grid_coords.shape, grid_coords.requires_grad), self._should_log)

        return scaled_coords, grid_coords

    def _create_kl_div_targets(
        self, 
        max_input_values_per_level: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        
        Parameters
        ----------
        max_input_values_per_level : torch.Tensor
            Maximum input values per level.
        
        Returns
        -------
        List[torch.Tensor]
            KL Div target.
        """
        
        kl_div_targets: List[torch.Tensor] = []
        for l in range(self._num_levels):
            
            if self._min_possible_collisions_per_level[l] <= 0:
                size = int(torch.prod(max_input_values_per_level[l] + 1).item())
                start_end = np.log(1e-11)
                end_start = np.log(1/size/100)
                
                kl_div_targets.append(
                    torch.cat([
                        # torch.exp(torch.arange(
                        #     start_end, 
                        #     end_start, 
                        #     step=(end_start - start_end)/np.floor((self._hash_table_size - size)/2),
                        #     device=self._device
                        # )),
                        # torch.zeros(
                        #     math.floor((self._hash_table_size - size)/2),
                        #     device=self._device
                        # ),
                        (
                            torch.ones(
                                size,
                                device=self._device
                            ) / size
                        ),
                        # torch.zeros(
                        #     math.ceil((self._hash_table_size - size)/2),
                        #     device=self._device
                        # )
                        # torch.exp(torch.arange(
                        #     end_start, 
                        #     start_end, 
                        #     step=(start_end - end_start)/np.ceil((self._hash_table_size - size)/2),
                        #     device=self._device
                        # )),
                        torch.exp(torch.arange(
                            end_start, 
                            start_end, 
                            step=(start_end - end_start)/np.round((self._hash_table_size - size)),
                            device=self._device
                        )),
                    ])
                )
            else:
                kl_div_targets.append(
                    torch.ones(
                        self._hash_table_size,
                        device=self._device
                    ) / self._hash_table_size
                )
        log(
            (
                "KL Div target:", 
                [
                    (
                        kl_div_targets[l], 
                        kl_div_targets[l].shape, 
                        kl_div_targets[l].requires_grad, 
                        kl_div_targets[l].is_leaf
                    ) for l in range(self._num_levels)
                ]
            ), 
            self._should_log
        )
        
        return kl_div_targets

    def get_num_dimensions(self) -> int:
        return len(self._dimensions)
# --------------------- #
