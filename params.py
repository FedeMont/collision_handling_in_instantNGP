# should save the model's parameters and the optimizer state?
should_save_params = False                             #@param {type: "boolean"}

# should BatchNorm the input coordinates or just divide them by max(w, h)?
should_batchnorm_data = False                           #@param {type: "boolean"}

# should use a random seed or not?
should_random_seed = True                              #@param {type: "boolean"}

# should use out of place scatter or scatter without saving?
should_inplace_scatter = True                          #@param {type: "boolean"}

# should use softmax topk or weighted average when summing looked_up features?
should_softmax_topk_features = True                    #@param {type: "boolean"}

# should ReLU or LeakyReLU in GeneralNeuralGaugeFields MLP?
should_leaky_relu = False                               #@param {type: "boolean"}

# should use hash function or HPD?
should_use_hash_function = True                        #@param {type: "boolean"}

# should log the allocated memory or not?
should_log_allocated_memory = False                     #@param{type: "boolean"}


exp = 8                                                #@param {type: "integer"}
hash_table_size: int = 2**exp
num_levels: int = 4                                    #@param {type: "integer"}
n_min: int = 8                                         #@param {type: "integer"}
n_max: int = 32                                        #@param {type: "integer"}
feature_dim: int = 2                                   #@param {type: "integer"}

MLP_hidden_layers_widths: list = [64, 64]                  #@param {type: "raw"}
HPD_hidden_layers_widths: list = [32, 64, 128]             #@param {type: "raw"}
HPD_out_features: int = hash_table_size

encoding_lr = 1e-4                                      #@param {type: "number"}

encoding_weight_decay = 0                               #@param {type: "number"}
HPD_weight_decay = 1e-6                                 #@param {type: "number"}
MLP_weight_decay = 1e-6                                 #@param {type: "number"}

batch_size = 1/3                                           #@param {type: "raw"}

epochs = 5000

tolerance = 500                                        #@param {type: "integer"}
min_delta = 1e-6                                        #@param {type: "number"}

histograms_rate = 100                                   #@param {type: "number"}
weights_rate = 20                                       #@param {type: "number"}


# ---------------------------------- #
#       Grid Search Parameters       #


grid_search_configs = {
    "should_shuffle_pixels": [True, False],
    "should_keep_topk_only": [False, True],

    "should_sum_js_kl_div": [False, True],
    "loss_gamma": [-2, -3, -0.5, 0],

    "should_js_div": [False, True],

    "l_mse": [1, 1e1, 1e2, 1e3, 5e2],
    "l_js_kl": [1, 1e1, 1e2, 1e3, 5e2],
    "l_collisions": [1, 1e-1, 1e-2, 1e-3],
    
    "MLP_lr": [1e-3, 1e-4],
    "HPD_lr": [1e-3, 1e-4],

    "topk_k": [1, 4, 20, 32, 128],
}
