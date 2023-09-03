from functions import *
from params import *


class DifferentiableTopk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, k: int, dim: int):
        # ctx.k = k
        ctx.dim = dim

        values, indices = torch.topk(input, k=k, dim=dim, largest=True, sorted=True)

        print_allocated_memory() # After torch.topk

        ctx.save_for_backward(input, indices)

        print_allocated_memory() # After save_for_backward

        return values, indices

    @staticmethod
    def backward(ctx, grad_values: torch.Tensor, grad_indices: torch.Tensor):
        input, indices = ctx.saved_tensors
        del grad_indices
        # k = ctx.k

        grad_input = torch.zeros_like(input, requires_grad=True)
        del input

        if should_inplace_scatter is None:
            grad_input.scatter(ctx.dim, indices, grad_values)
        elif should_inplace_scatter:
            grad_input.scatter_(ctx.dim, indices, grad_values)
        else:
            grad_input = grad_input.scatter(ctx.dim, indices, grad_values) # out of place operation should help with memory problems

        del grad_values
        del indices

        # del k

        return grad_input, None, None
    

class HashProbDistribution(nn.Module):
    def __init__(
        self,
        hidden_layers_widths: list,
        in_features: int = 2,
        out_features: int = 2**14,
        k: int = 1,
        topk_dim: int = -1,
        should_log: bool = False
    ):
        """ Compute a sort of hash function of the input

        Parameters
        ----------
        hidden_layers_widths : list
            The list conaining the width of each hidden layer (must be of length +1 of the desired one)
            first: output of input layer, input of first hidden layer
            ...
            last: output of last hidden layer, input of output layer
        in_features : int, optional
            The number of input features (default is 2)
        out_features : int, optional
            The number of output features (default is 2^14)
        k : int, optional
            The top k value for choosing hashed indices (default is 1)
        topk_dim: int, optional
            The dimension on which apply the topk operation (default is -1)
        """
        super(HashProbDistribution, self).__init__()

        self.out_features = out_features
        self._k = k
        self._topk_dim = topk_dim
        self._should_log = should_log

        layers_widths = [in_features, *hidden_layers_widths, out_features]

        self.module_list = nn.ModuleList([
            torch.nn.Sequential(
                nn.Linear(in_features=layers_widths[i], out_features=layers_widths[i + 1]),
                nn.ReLU() if i < len(layers_widths) - 2 else nn.Softmax(dim=-1) #nn.LogSoftmax(dim=-1)
            )
            for i in range(len(layers_widths) - 1)
        ])

    def forward(self, x: torch.Tensor) -> Tuple:
        """ Returns the probability distribution over the possible indices of the hash table

        Parameters
        ----------
        x : torch.Tensor
            The input

        Returns
        ----------
        Tuple
            The total probability distribution, the probability distribution of the topk and the indices of the topk
        """
        print2(("INPUT:", x.requires_grad, x.grad, x.grad_fn), self._should_log)

        for layer in self.module_list:
            x = layer(x)
            print2((layer, x.requires_grad, x.grad, x.grad_fn), self._should_log)

        print_allocated_memory() # After module_list

        x = torch.nan_to_num(x.squeeze(-1)) # Sanitize nan to 0.0
        print2(("PROBS grad:", x.requires_grad, x.grad, x.grad_fn), self._should_log)

        print_allocated_memory() # After nan_to_num

        topk_probs, topk_indices = DifferentiableTopk.apply(x, self._k, self._topk_dim)
        if self._k != 1:
            topk_probs = topk_probs.squeeze(-1)
            topk_indices = topk_indices.squeeze(-1)

        print_allocated_memory() # After DifferentiableTopk

        return x, topk_probs, topk_indices
    

class MultiResHashEncoding(nn.Module):
    def __init__(
        self,
        hash_table_size: int,
        num_levels: int,
        feature_dim: int = 2,
        topk_k: int = 4,
        should_log: bool = False
    ) -> None:
        """

        Parameters
        ----------
        hash_table_size : int
            The size of the hasing table [2**14, 2**24]
        num_levels: int
            The number of levels
        feature_dim : int, optional
            The dimension of the feature (default is 2)
        topk_k : int, optional
            The topk value for choosing hashed indices (default is 4)
        should_log : bool, optional
            Wether or not to log the outputs (default False)
        """
        super(MultiResHashEncoding, self).__init__()

        self._hash_table_size: int = hash_table_size
        self._num_levels: int = num_levels
        self._feature_dim: int = feature_dim
        self._topk_k: int = topk_k
        self._should_log: bool = should_log

        # Create hash tables one for each level
        self._hash_tables: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self._hash_table_size, self._feature_dim, device=device)
                for _ in range(self._num_levels)
            ]
        )
        print2(("HASH TABLE level 0:", self._hash_tables[0].weight, self._hash_tables[0].weight.shape), self._should_log)
        # end

        # initialize weights of hash tables
        self._apply_init(torch.nn.init.uniform_, -10.0**(-4), 10.0**(-4))
        print2(("INITIALIZED HASH TABLE level 0:", self._hash_tables[0].weight, self._hash_tables[0].weight.shape), self._should_log)
        # end

    def forward(
        self,
        hashed_indices: torch.Tensor,
        hashed_probs_topk: torch.Tensor,
        should_calc_counts: bool = False
    ) -> Tuple[torch.Tensor]:

        # Look up hashed indices features
        if should_use_hash_function:
            looked_up = torch.stack(
                [
                    self._hash_tables[j](
                        hashed_indices[:, j].int()
                    ).permute(0, 2, 1)
                    for j in range(self._num_levels)
                ],
                dim=2
            )
            print2(("looked_up level 0:", looked_up[0], looked_up[0].shape), self._should_log)

        else:
            looked_up: torch.Tensor = rearrange(
                torch.stack(
                    [
                        torch.stack([
                            self._hash_tables[j](
                                hashed_indices[:, j, :, k].int()
                            ).permute(0, 2, 1)
                            for j in range(self._num_levels)
                        ])
                        for k in range(self._topk_k)
                    ],
                    dim=2
                ),
                "l p k f v -> p l f v k"
            )

            print2(("looked_up level 0:", looked_up[0], looked_up.shape), self._should_log)

            if should_softmax_topk_features == None: # (wrong) sum(looked_up * topk)
                looked_up = torch.sum(looked_up * hashed_probs_topk.unsqueeze(2), dim=-1)
            elif should_softmax_topk_features: # (avg) sum(looked_up * softmax(topk))
                looked_up = torch.sum(looked_up * F.softmax(hashed_probs_topk.unsqueeze(2), dim=-1), dim=-1)
            else: # (weighted avg) sum(looked_up * topk)/sum(topk)
                looked_up = torch.sum(looked_up * hashed_probs_topk.unsqueeze(2), dim=-1) / torch.sum(hashed_probs_topk.unsqueeze(2), dim=-1)

            looked_up = rearrange(
                looked_up,
                "p l f v -> p f l v"
            )

            print2(("Wieghted avg looked_up level 0:", looked_up[0], looked_up.shape), self._should_log)

        print_allocated_memory() # After looked_up
        # end

        return looked_up

    def _apply_init(self, init_func, *args):
        """
        Initializes the hash tables weights with a random uniform function
        """
        for i in range(self._num_levels):
            init_func(self._hash_tables[i].weight, *args)


class GeneralNeuralGaugeFields(nn.Module):
    def __init__(
        self,
        input_dim: list,
        hash_table_size: int,
        num_levels: int,
        n_min: int,
        n_max: int,
        MLP_hidden_layers_widths: list,
        HPD_hidden_layers_widths: list,
        HPD_out_features: int = 1,
        feature_dim: int = 2,
        topk_k: int = 4,
        should_keep_topk_only: bool = False,
        should_bw: bool = False,
        should_log: bool = False,
    ):
        """

        Parameters
        ----------
        input_dim: list
            The shape of the input
        hash_table_size : int
            The size of the hasing table [2**14, 2**24]
        num_levels: int
            The number of levels
        n_min: int
            Coarsest resolution
        n_max: int
            Finest resolution
        MLP_hidden_layers_widths: list
            The list of the witdhs of the hidden layers of the MLP network
        HPD_hidden_layers_widths: list
            The list of the witdhs of the hidden layers of the GGNF network
        HPD_out_features: int
            The number of output features of the GGNF network. Default is 1
        topk_k : int, optional
            The topk value for choosing hashed indices (default is 4)
        should_keep_topk_only : bool, optional
            If True only the topk values will be kept (default is False)
        should_bw : bool, optional
            If image should be black & white the output feature of the last layer will be 1 instead of 3 (default is False)
        should_log : bool, optional
            Wether or not to log the outputs (default False)
        """

        super(GeneralNeuralGaugeFields, self).__init__()

        self._hash_table_size: int = hash_table_size
        self._num_levels: int = num_levels
        self._n_min: int = n_min
        self._n_max: int = n_max
        self._feature_dim: int = feature_dim
        self._input_dim: int = input_dim
        self._topk_k = topk_k
        self._should_log: bool = should_log
        self._should_keep_topk_only: bool = should_keep_topk_only

        # Calc Resolution Layers
        b: float = np.exp((np.log(self._n_max) - np.log(self._n_min)) / (self._num_levels - 1))
        if b > 2 or b <= 1:
            print(
                f"The between level scale is recommended to be <= 2 and needs to be > 1 but was {b:.4f}."
            )

        self._n_ls: torch.Tensor = torch.from_numpy(
            np.array(
                [
                    np.floor(self._n_min * b**l) for l in range(self._num_levels)
                ]
            )
        ).reshape(1, 1, -1, 1).to(device).int()
        print2(("NL:", self._n_ls, self._n_ls.shape), self._should_log)
        # end

        # create Voxel Helper Hypercube
        helper_hypercube: np.ndarray = np.empty((self._input_dim, 2**self._input_dim), dtype=int)
        for i in range(self._input_dim):
            pattern = np.array(
                ([0] * (2**i) + [1] * (2**i)) * (2**(self._input_dim - i - 1)),
                dtype=int
            )
            helper_hypercube[i, :] = pattern
        print2(("helper_hypercube:", helper_hypercube, helper_hypercube.shape), self._should_log)

        self._voxels_helper_hypercube: torch.Tensor = torch.from_numpy(helper_hypercube).unsqueeze(0).unsqueeze(2).to(device).int()
        print2(("voxels_helper_hypercube:", self._voxels_helper_hypercube, self._voxels_helper_hypercube.shape), self._should_log)

        del helper_hypercube
        # end

        # ---------------------- #
        # Layers

        self._batch_norm = nn.BatchNorm1d(self._input_dim)

        if should_use_hash_function:
            # initialize prime numbers
            self._prime_numbers = torch.nn.Parameter(
                torch.from_numpy(
                    np.array([1, 2654435761, 805459861])
                ).to(device),
                False
            )
            # end
        else:
            # initialize HPD model
            self.HPD = HashProbDistribution(
                hidden_layers_widths=HPD_hidden_layers_widths,
                in_features=self._input_dim,
                out_features=HPD_out_features,
                k=self._topk_k,
                topk_dim=-1
            )
            print2(("HPD:", self.HPD), self._should_log)
            print2(("#PARAMETERS:", [(name[0], param.numel()) for name, param in zip(self.HPD.named_parameters(), self.HPD.parameters())]), self._should_log)
            # end

        self.encoding = MultiResHashEncoding(
            hash_table_size=self._hash_table_size,
            num_levels=self._num_levels,
            feature_dim=self._feature_dim,
            topk_k=self._topk_k,
            should_log=self._should_log
        )

        self._MLP_hidden_layers_widths = [self._num_levels * self._feature_dim, *MLP_hidden_layers_widths, (3 if not should_bw else 1)]
        # self._MLP_hidden_layers_widths.append(3 if not should_bw else 1)

        self.mlp = torch.nn.ModuleList([
            torch.nn.Sequential(
                nn.Linear(in_features=self._MLP_hidden_layers_widths[i], out_features=self._MLP_hidden_layers_widths[i+1]),
                (
                    nn.LeakyReLU() if should_leaky_relu else nn.ReLU()
                ) if i < len(self._MLP_hidden_layers_widths) - 2 else nn.Sigmoid()
            )
            for i in range(len(self._MLP_hidden_layers_widths) - 1)
        ])

    def forward(self, x: torch.Tensor, batch_percentage: float, should_calc_counts: bool = False):
        # 0. batch norm
        if should_batchnorm_data:
            x = self._batch_norm(x)

            print_allocated_memory() # After batch_norm()
        # end

        # 1. Scale and get surrounding grid coords
        scaled_coords, grid_coords = self._scale_to_grid(x)
        print2(("x:", x, x.shape), self._should_log)
        print2(("scaled_coords:", scaled_coords, scaled_coords.shape), self._should_log)
        print2(("grid_coords:", grid_coords, grid_coords.shape), self._should_log)

        print_allocated_memory() # After _scale_to_grid()
        # end

        # 2. Hash the grid coords
        if should_use_hash_function:
            hashed_indices = self._fast_hash(grid_coords.int())
            print2(("hashed_indices:", hashed_indices, hashed_indices.shape), self._should_log)
        else:
            rearranged_grid_coords = rearrange(grid_coords, "p xy l v -> p l v xy").requires_grad_()
            
            hashed_probs, hashed_probs_topk, hashed_indices_topk = self.HPD(rearranged_grid_coords)
            del rearranged_grid_coords
            print2(("HASHED PROBS:", hashed_probs, hashed_probs.shape), self._should_log)
            print2(("HASHED INDICES TOPK:", hashed_indices_topk.requires_grad, hashed_indices_topk.grad, hashed_indices_topk.grad_fn), self._should_log)
            print2(("MLP indices TOPK:", hashed_indices_topk, hashed_indices_topk.shape), self._should_log)
            print2(("MLP probs TOPK:", hashed_probs_topk, hashed_probs_topk.shape), self._should_log)

        print_allocated_memory() # After HPD()
        # end

        # 3. Calc hash collisions
        counts_per_level = []

        if should_calc_counts:
            counts_per_level = self._calc_counts_per_level(
                (
                    hashed_indices_topk[..., 0] # takes only best ?
                    if not should_use_hash_function else
                    hashed_indices
                ),
                grid_coords
            )

            print2(("Counts per level:", counts_per_level), self._should_log)

        print_allocated_memory() # After _calc_hash_collisions()
        # end

        # 4. Calc features
        x = self.encoding(
            (
                hashed_indices_topk
                if not should_use_hash_function else
                hashed_indices
            ),
            hashed_probs_topk if not should_use_hash_function else None,
            should_calc_counts=should_calc_counts
        )

        print_allocated_memory() # After encoding()
        # end

        # 5. Bilinear interpolation
        x: torch.Tensor = self._bilinear_interpolate(scaled_coords, grid_coords, x)
        del scaled_coords
        del grid_coords

        print_allocated_memory() # After _bilinear_interpolate()
        # end

        # 6. MLP
        for layer in self.mlp:
            x = layer(x)

        print_allocated_memory() # After mlp()
        # end

        if should_use_hash_function:
            return x, None, hashed_indices, counts_per_level
        else:
            to_return_probs = hashed_probs_topk.clone() if self._should_keep_topk_only else hashed_probs.clone()
            del hashed_probs
            del hashed_probs_topk

            print_allocated_memory() # Before return

            return x, to_return_probs, hashed_indices_topk, counts_per_level

    @torch.no_grad()
    def _scale_to_grid(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scales the inputs into a set of grids and returns the scaled coordinate
        as well as the coordinate of all voxel vertices for the input at all scales.
        """
        scaled_coords = torch.mul(
            x.unsqueeze(-1).unsqueeze(-1),
            self._n_ls
        )

        grid_coords = torch.add(
            torch.floor(scaled_coords), #.int(),
            self._voxels_helper_hypercube
        )

        return scaled_coords, grid_coords

    @torch.no_grad()
    def _fast_hash(self, grid: torch.Tensor) -> torch.Tensor:
        '''
        Implements the hash function proposed by NVIDIA.
        Args:
            grid: A tensor of the shape (batch, input_dim, levels, 2^input_dim).
               This tensor should contain the vertices of the hyper cuber
               for each level.
        Returns:
            A tensor of the shape (batch, levels, 2^input_dim) containing the
            indices into the hash table for all vertices.
        '''
        tmp = torch.zeros(
            (grid.shape[0], self._num_levels, 2**self._input_dim),
            dtype=torch.int64,
            device=device
        )

        for i in range(self._input_dim):
            tmp = torch.bitwise_xor(grid[:, i, :, :] * self._prime_numbers[i], tmp)

        hash = torch.remainder(tmp, self._hash_table_size)
        del tmp

        return hash

    @torch.no_grad()
    def _calc_counts_per_level(self, hash: torch.Tensor, grid: torch.Tensor) -> Tuple:
        print2(("Grid Coordinates:", grid, grid.shape), self._should_log)
        rearranged_grid: torch.Tensor = rearrange(grid, "p xy l v -> l p (v xy)") # p (v xy)
        print2(("Rearranged Grid Coordinates:", rearranged_grid, rearranged_grid.shape), self._should_log)

        print2(("Hash:", hash, hash.shape), self._should_log)

        vertices_per_level: torch.Tensor = rearrange(hash, "p l v -> l (p v)")
        print2(("Hash indices per level:", vertices_per_level, vertices_per_level.shape), self._should_log)

        counts_per_level = []
        for level in range(self._num_levels):

            np_grid = rearranged_grid[level].detach().cpu().numpy()
            # remove duplicated pixels inside same cell
            _, unique_indices = np.unique(np_grid, axis=0, return_index=True)
            del np_grid
            # _, unique_indices = torch.unique(rearranged_grid[level], dim=0, return_inverse=True)

            unique_indices_tensor = torch.tensor(unique_indices)
            del unique_indices

            cells = vertices_per_level[level][unique_indices_tensor]
            del unique_indices_tensor

            counts_per_level.append(
                # dict(Counter(vertices_per_level[level][unique_indices].detach().cpu().numpy().tolist()))
                dict(Counter(cells.detach().cpu().numpy().tolist()))
            )
            del cells
            # del unique_indices

        # del rearranged_grid
        del vertices_per_level

        return counts_per_level

    @torch.no_grad()
    def calc_hash_collisions(self, indices: torch.Tensor) -> Tuple:
        """
        Calculate the number of collisions of the 'hash function' by subtracting the number of uniques the hashed indices from the number of unique vertices at each level
        """

        if should_use_hash_function:
            vertices_per_level: torch.Tensor = rearrange(indices, "p l v -> l (p v)")
            print2(("Hash indices per level:", vertices_per_level, vertices_per_level.shape), self._should_log)

            collisions = torch.from_numpy(
                np.array([
                    (
                        (4 + (self._n_ls[0, 0, i, 0].item() + 1 - 2) * 4 + (self._n_ls[0, 0, i, 0].item() + 1 - 2)**2) - torch.unique(vertices_per_level[i], dim=0).shape[0]
                    ) for i in range(self._num_levels)
                ])
            )
            del vertices_per_level
        else:
            collisions = torch.empty((indices.shape[-1], self._num_levels)).to(device)

            for k in range(indices.shape[-1]):
                hash = indices[..., k]
                print2(("Hash:", hash, hash.shape), self._should_log)

                vertices_per_level: torch.Tensor = rearrange(hash, "p l v -> l (p v)")
                del hash
                print2(("Hash indices per level:", vertices_per_level, vertices_per_level.shape), self._should_log)

                collisions[k] = torch.from_numpy(
                    np.array([
                        (
                            (4 + (self._n_ls[0, 0, i, 0].item() + 1 - 2) * 4 + (self._n_ls[0, 0, i, 0].item() + 1 - 2)**2) - torch.unique(vertices_per_level[i], dim=0).shape[0]
                        ) for i in range(self._num_levels)
                    ])
                )
                del vertices_per_level

            collisions = torch.mean(collisions, dim=0)
            collisions[collisions < 0] = 0

        min_possible_collisions = torch.from_numpy(
            np.array([
                (
                    (4 + (self._n_ls[0, 0, i, 0].item() + 1 - 2) * 4 + (self._n_ls[0, 0, i, 0].item() + 1 - 2)**2) - self._hash_table_size
                ) for i in range(self._num_levels)
            ])
        ).to(device)

        min_possible_collisions[min_possible_collisions < 0] = 0

        return collisions, min_possible_collisions

    def _bilinear_interpolate(self, scaled_coords: torch.Tensor, grid_coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        print2(("SCALED COORDS:", scaled_coords[:, :, 0, :], scaled_coords.shape), self._should_log)
        print2(("GRID COORDS:", grid_coords[:, :, 0, :], grid_coords.shape), self._should_log)
        print2(("FEATURES:", features[0, :, 0, :], features.shape), self._should_log)

        _as: torch.Tensor = grid_coords[:, :, :, 0].unsqueeze(-1)  # bottom-right vertices of cells
        _ds: torch.Tensor = grid_coords[:, :, :, -1].unsqueeze(-1)  # top-left vertices of cells

        print2(("_as:", _as, _as.shape), self._should_log)
        print2(("_ds:", _ds, _ds.shape), self._should_log)

        coeffs: torch.Tensor = torch.stack([
            (_ds[:, 0, :, :] - scaled_coords[:, 0, :, :]) * (_ds[:, 1, :, :] - scaled_coords[:, 1, :, :]),  # (xd - x) * (yd - y)
            (scaled_coords[:, 0, :, :] - _as[:, 0, :, :]) * (_ds[:, 1, :, :] - scaled_coords[:, 1, :, :]),  # (x - xa) * (yd - y)
            (_ds[:, 0, :, :] - scaled_coords[:, 0, :, :]) * (scaled_coords[:, 1, :, :] - _as[:, 1, :, :]),  # (xd - x) * (y - ya)
            (scaled_coords[:, 0, :, :] - _as[:, 0, :, :]) * (scaled_coords[:, 1, :, :] - _as[:, 1, :, :]),  # (x - xa) * (y - ya)
        ], dim=-1).squeeze(-2).unsqueeze(1)#.to(device)
        del _as
        del _ds
        print2(("COEFFS:", coeffs, coeffs.shape), self._should_log)

        weighted_features: torch.Tensor = features * coeffs
        del coeffs
        print2(("W_FEATURES:", weighted_features[0, :, 0, :], weighted_features.shape), self._should_log)

        weighted_summed_features: torch.Tensor = torch.sum(weighted_features, dim=-1)#.to(device)
        del weighted_features
        print2(("W_S_FEATURES:", weighted_summed_features, weighted_summed_features.shape), self._should_log)
        print2(("W_S_FEATURES:", weighted_summed_features[0, :, 0], weighted_summed_features.shape), self._should_log)

        stack: torch.Tensor = rearrange(weighted_summed_features, "p f l -> p (l f)")#.to(device)
        del weighted_summed_features
        print2(("STACK:", stack, stack.shape), self._should_log)

        return stack
    
