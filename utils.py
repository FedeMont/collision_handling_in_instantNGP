from functions import *
from params import *


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, dir_name: str, image_name: str, should_bw: bool = False) -> None:
        """

        Parameters
        ----------
        root : str
            The root directory path
        dir_name : str
            The directory path
        image_name : str
            The name of the image to load
        should_bw: bool
            If the image should be converted to black & white only (default False)
        """

        self._root = root
        self._dir_name = dir_name
        self._image_name = image_name

        self._should_bw = should_bw

        self._image_path = os.path.join(
            os.path.join(self._root, self._dir_name),
            self._image_name
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """

        Parameters
        ----------
        idx: int
            The index of the image to retrieve

        Returns
        ----------
        tuple
            X, Y, height, width
        """

        self._image: np.ndarray = cv2.cvtColor(
            cv2.imread(
                self._image_path
            )[:, :, :3],  # keep only rgb in case of rgba image
            cv2.COLOR_BGR2RGB if not self._should_bw else cv2.COLOR_BGR2GRAY
        )

        height = self._image.shape[0]  # to opencv should be height
        width = self._image.shape[1]  # to opencv should be width

        X = np.array([
            (x, y)
            for x, y in np.stack(np.meshgrid(range(height), range(width), indexing="ij"), axis=-1).reshape(-1, 2)
        ])

        Y = np.array([rgb if not self._should_bw else [rgb] for col in self._image for rgb in col]) / 255

        X = torch.tensor(X).float()
        Y = torch.tensor(Y).float()

        return X, Y, height, width # width & height should always be in this order and as last

    def __len__(self) -> int:
        return 1

    def get_image(self) -> np.ndarray:
        return self._image

    def get_image_name(self) -> str:
        return self._image_name
    

class Loss(torch.nn.Module):
    def __init__(self, delta: float = 1, gamma: float = 1, epsilon: float = 0.5, should_log: bool = False) -> None:
        super(Loss, self).__init__()

        self._delta = delta
        self._should_log = should_log

        self.mse: torch.nn.MSELoss = torch.nn.MSELoss()
        self.kl_div_loss: torch.nn.KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')

        self._gamma = gamma
        self._epsilon = epsilon

    def forward(
        self,
        pred: torch.tensor,
        labels: torch.tensor,
        N: int,
        prob: torch.Tensor,
        collisions: torch.Tensor,
        min_possible_collisions: torch.Tensor
    ) -> float:
        mse_loss: torch.Tensor = self.mse(pred, labels)
        print2(("MSE Loss:", mse_loss, mse_loss.dtype), self._should_log)

        if should_use_hash_function:
            return mse_loss, None, None
        else:
            print2(("prob:", prob, prob.shape), self._should_log)

            collisions_losses: torch.Tensor = collisions / (min_possible_collisions + self._delta)
            print2(("Collisions Losses:", collisions_losses, collisions_losses.dtype), self._should_log)

            kl_divs: torch.Tensor = torch.stack([

                self.js_kl_div(N, prob[:, l, :], (prob.shape[0] * prob.shape[2]))

                for l in range(prob.shape[1])
            ])

            print2(("KL Divergences:", kl_divs, kl_divs.dtype), self._should_log)

            return mse_loss, kl_divs, collisions_losses

    def js_kl_div(self, N: int, p: torch.Tensor, div: int):
        """
        Function that sum JS Div and KL Div Losses:
        """

        return -(self._gamma + self._epsilon) * self.js_div(N, p, div) + self._epsilon * self.kl_div(N, p, div)


    def kl_div(self, N: int, p: torch.Tensor, div: int):
        """
        Function that measures KL divergence between target and output logits:
        """
        q_output = (torch.ones(N) / float(N))

        print2(("p:", p, p.shape), self._should_log)

        p_output = p.sum(0).sum(0) / div

        print2(("p_output", p_output, p_output.shape, p_output.sum(0)), self._should_log)
        print2(("q_output", q_output, q_output.shape, q_output.sum(0)), self._should_log)

        # loss = self.kl_div(p_output, q_output)
        loss = self.kl_div_loss(p_output.log(), q_output)

        del q_output
        del p_output

        return loss

    def js_div(self, N: int, p: torch.Tensor, div: int):
        """
        Function that measures JS divergence between target and output logits:
        """
        q_output = (torch.ones(N) / float(N))

        print2(("p:", p, p.shape), self._should_log)

        p_output = p.sum(0).sum(0) / div

        print2(("p_output", p_output, p_output.shape, p_output.sum(0)), self._should_log)
        print2(("q_output", q_output, q_output.shape, q_output.sum(0)), self._should_log)

        # log_mean_output = ((p_output + q_output) / 2).log()
        # loss = (self.kl_div_loss(log_mean_output, p_output) + self.kl_div_loss(log_mean_output, q_output)) / 2

        log_mean_output = ((p_output + q_output) / 2)
        loss = (self.kl_div_loss(p_output.log(), log_mean_output) + self.kl_div_loss(q_output.log(), log_mean_output)) / 2

        del log_mean_output
        del q_output
        del p_output

        return loss
    

class EarlyStopping:
    def __init__(self, tolerance: int = 5, min_delta: int = 0, should_reset: bool = True):
        self.tolerance: int = tolerance
        self.min_delta: int = min_delta
        self.best_loss: float = np.inf
        self.counter: int = 0
        self.early_stop: bool = False
        self._should_reset: bool = should_reset

    def __call__(self, loss):
        # print(f"best_loss: {self.best_loss}, loss: {loss}, counter: {self.counter}")

        if abs(self.best_loss - loss) < self.min_delta and (loss < self.best_loss):
            # print("Stall")
            self.counter += 1
        elif abs(self.best_loss - loss) > self.min_delta and (loss > self.best_loss):
            # print("Growing")
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
