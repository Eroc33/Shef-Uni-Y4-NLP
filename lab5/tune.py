import ray
from ray import tune
from skopt import Optimizer
from ray.tune.suggest.skopt import SkOptSearch
import lab5

ray.init()

search_space = {
    'embedding_dim':(5,50),
    'hidden_dim':(64,1024),
    'learning_rate':(0.0001,0.01),
}
current_best_params = {
    'embedding_dim':10,
    'hidden_dim':128,
    'learning_rate':0.0001,
}
optimizer = Optimizer(list(search_space.values()))
current_best_params = [list(current_best_params.values())]
algo = SkOptSearch(optimizer,
    list(search_space.keys()),
    max_concurrent=4,
    reward_attr="neg_mean_loss",
    )

all_trials = tune.run(
    lab5.main,
    num_samples=20,
    name="combined_score_metric",
    #bail once we get our dseired score, or hit the limit we wish to train to
    stop={'total_score': 1,'training_iteration':500},
    config={'passes': 500},
    search_alg=algo
)

print("ALL_TRIALS:",all_trials)