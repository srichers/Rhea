from run_trial import run_training_trial

  run_training_trial({
      "model_tier": "small",
      "lr": 1e-3,
      "batch_size": 64,
      "seed": 42,
  })