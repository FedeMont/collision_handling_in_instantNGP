from functions import *
from utils import *
from models import *
from params import *

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run General Gauge Neural Fields.')
    parser.add_argument('-f', '--filename', type=str, default="strawberry.jpeg", help='Name and extension of the image file to be used (image should be in "./images" directory).')
    parser.add_argument('--should_bw', type=bool, default=False, help='Whether the image should be converted to black and white.')
    parser.add_argument('-s', '--start_id_param', type=int, default=0, help='The id of the first parameter to be used in the grid search.')
    parser.add_argument('-e', '--end_id_param', type=int, default=None, help='The id of the last parameter to be used in the grid search.')
    parser.add_argument('-t', '--is_test', type=bool, default=True, help='If False will log data on wandb.')
    parser.add_argument('--wandb_entity', type=str, default="dl_project_bussola-fasoli-montagna", help='The entity to be used in wandb.')
    parser.add_argument('--wandb_project', type=str, default="cv_project_final_grid_search", help='The project where to log wandb.')
    parser.add_argument('--wandb_name', type=str, default=None, help='The name of the wandb run (only works if start_id_param == end_id_param).')

    args = parser.parse_args()

    is_test_only = args.is_test

    filename = args.filename
    should_bw = args.should_bw
    image_name, extension = filename.split('.')
    extension: str = '.' + extension

    start_id_param = args.start_id_param
    end_id_param = args.end_id_param + 1 if args.end_id_param is not None else args.end_id_param

    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name if (args.start_id_param == args.end_id_param) else None

    dataset: MyDataset = MyDataset(
        root=".",
        dir_name="images",
        image_name=image_name + extension,
        should_bw=should_bw
    )
    x, y, h, w = dataset[0]

    if not should_batchnorm_data:
        x = (x / (max(w, h) - 1))
    
    og_img: np.ndarray = dataset.get_image()

    shape = w * h
    shuffled_indices = torch.randperm(shape).int()
    reordered_indices = torch.zeros((shape, )).int()
    reordered_indices[shuffled_indices] = torch.arange(shape).int()

    filtered_grid_search = get_grid_search_configs(configs=grid_search_configs)

    grid_search_loop(
        filtered_grid_search=filtered_grid_search,
        x=x,
        y=y,
        w=w,
        h=h,
        image_name=image_name,
        og_image=og_img,
        shuffled_indices=shuffled_indices,
        reordered_indices=reordered_indices,
        GeneralNeuralGaugeFields=GeneralNeuralGaugeFields,
        Loss=Loss,
        EarlyStopping=EarlyStopping,
        should_bw=should_bw,
        start_id_param = start_id_param,
        end_id_param=end_id_param,
        is_test_only=is_test_only,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name
    )

    del shuffled_indices
    del reordered_indices
