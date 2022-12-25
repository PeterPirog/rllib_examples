import ray
from ray.data.preprocessors import StandardScaler
from ray.air.config import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
from ray.train.batch_predictor import BatchPredictor
from ray.train.xgboost import XGBoostPredictor
import pprint
pp = pprint.PrettyPrinter(indent=4)

# DATA PART
# Load data.
dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

# Split data into train and validation.
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

# Create a test dataset by dropping the target column.
test_dataset = valid_dataset.drop_columns(cols=["target"])

print(dataset.show())
print(test_dataset.show())

# CREATE PREPROCESSOR
# Create a preprocessor to scale some columns.
preprocessor = StandardScaler(columns=["mean radius", "mean texture"])


# TRAINING


trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        # Number of workers to use for data parallelism.
        num_workers=2,
        # Whether to use GPU acceleration.
        use_gpu=False,
    ),
    label_column="target",
    num_boost_round=20,
    params={
        # XGBoost specific params
        "objective": "binary:logistic",
        # "tree_method": "gpu_hist",  # uncomment this to use GPUs.
        "eval_metric": ["logloss", "error"],
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
    preprocessor=preprocessor,
)
result = trainer.fit()
pp.pprint(result.metrics)
# Get checkpoint to inference

checkpoint=result.checkpoint
print(dir(checkpoint))

batch_predictor = BatchPredictor.from_checkpoint(checkpoint, XGBoostPredictor)
predicted_probabilities = batch_predictor.predict(test_dataset)
predicted_probabilities.show()

"""

# You can also create a checkpoint from a trained model using
# `XGBoostCheckpoint.from_model`.
# = best_result.checkpoint
checkpoint = result.best_checkpoints

batch_predictor = BatchPredictor.from_checkpoint(checkpoint, XGBoostPredictor)

predicted_probabilities = batch_predictor.predict(test_dataset)
predicted_probabilities.show()

"""

"""
# HYPERPARAMETER TUNING
from ray import tune

param_space = {"params": {"max_depth": tune.randint(1, 9)}}
#metric = "train-logloss"
metric = "valid-logloss"

from ray.tune.tuner import Tuner, TuneConfig
from ray.air.config import RunConfig

tuner = Tuner(
    trainer,
    param_space=param_space,
    tune_config=TuneConfig(num_samples=5, metric=metric, mode="min"),
)
# Execute tuning.
result_grid = tuner.fit()

# Fetch the best result.
best_result = result_grid.get_best_result()
print("Best Result:", best_result)
"""