import yaml

best_fitness = 0.0
wait_count = 0
with open(f'../configs/stage{input('What Stage You Are In Right Now? (1 or 2): ').strip()}.yaml','r') as f: patience_limit = yaml.safe_load(f)['model_args']['patience']


def on_fit_epoch_end(trainer):
    global best_fitness, wait_count
    
    if hasattr(trainer, "metrics") and trainer.metrics:
        current_fitness = trainer.metrics.get("metrics/mAP50(B)", 0.0)
        
        print(f"\n--- Epoch {trainer.epoch + 1} Validation Results ---")
        for k, v in trainer.metrics.items():
            print(f"{k}: {v}")
            
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            wait_count = 0
            print(f"[Early Stopping] Improved! Reset wait count to 0")
        else:
            wait_count += 1
            print(f"[Early Stopping] No improvement: {wait_count} out of {patience_limit}")