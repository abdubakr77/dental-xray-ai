def on_fit_epoch_end(trainer):
  if hasattr(trainer, "metrics") and trainer.metrics:
    print(f"\n--- Epoch {trainer.epoch + 1} Validation Results ---")
    for k, v in trainer.metrics.items():
      print(f"{k}: {v}")
