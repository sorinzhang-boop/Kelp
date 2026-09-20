ACTIVE_MODEL = "qwen3_14b"


MODEL_CONFIGS = {
    "qwen3_8b": {
        "model_name": "Qwen/Qwen3-8B",
        "idx_layer": 20,
    },

    "qwen3_14b": {
        "model_name": "Qwen/Qwen3-14B",
        "idx_layer": 20,
    },
}


TRAIN_CONFIG = {
    "lr": 5e-5,
    "weight_decay": 0.0,
    "warmup_ratio": 0.05,
    "num_train_epochs": 1,

    "batch_size": 1,
    "gradient_acc_steps": 32,

    "max_length": 4096,
    "num_supervised_token": 10,

    "seed": 42,
}