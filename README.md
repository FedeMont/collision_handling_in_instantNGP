# NeRF: Collision handling in Instant Neural Graphics Primitives
#### Federico Montagna (fedemonti00@gmail.com), Luca Santagata (lucasantagatas3@gmail.com)

## Code composition:
- `images/`: directory containing the image used as dataset for the project (add image here if you want to try with another image)
- `models.py`: contains the models used for the project (DifferentiableTopK, HashProbDistribution, MultiResHashEncoding, GeneralNeuralGaugeFields)
- `functions.py`: contains various helperfunctions used for the train loop, the grid search and others
- `utils.py`: contains the Dataset, Loss and EarlyStopping classes used for the project
- `params.py`: contains the default parameters of the project
- `main.py`: contains the main function of the project

## How to run the code:
### (Python)
```python
python3 main.py -f strawberry.jpeg -s 4061 -e 4061
```

> `4061` is the ID of the best parameter we could found during the grid search. With this command you will train the model with the best parameters on the strawberry image.