import torch
from tqdm import tqdm
import os
from torchvision.transforms import v2
import torch.nn.functional as F
from sklearn.metrics import recall_score

def train(model, train_dl, valid_dl, epochs, criterion, optimizer,
          early_stopping_metric='loss', patience=10, scheduler=None,
          enable_cutmix=False, enable_mixup=False,
          device_name='cpu', save_dir=os.getcwd(),
          disease_class_idx=None, save_best_recall=False,
          best_recall_name='best_recall.pt'):
    
    device = torch.device(device_name)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    os.makedirs(save_dir, exist_ok=True)

    history = []
    best_epoch = 0
    counter = 0
    best_disease_recall = -float('inf') if save_best_recall else None

    metric_name = early_stopping_metric.lower()
    if metric_name in {"loss", "val_loss", "valid_loss"}:
        best_metric = float("inf")
        monitor_mode = "min"
    elif metric_name in {"acc", "accuracy", "val_acc", "valid_acc"}:
        best_metric = -float("inf")
        monitor_mode = "max"
    else:
        raise ValueError(
            f"Unsupported early_stopping_metric='{early_stopping_metric}'. "
            "Use 'loss' or 'acc'/'accuracy'."
        )

    use_early_stopping = patience is not None and patience > 0
    if not use_early_stopping: print(" (early stopping disabled)")

    apply_cutmix_or_mixup = None
    num_classes = len(train_dl.dataset.classes)
    if enable_cutmix and enable_mixup:
        cutmix = v2.CutMix(num_classes=num_classes)
        mixup = v2.MixUp(num_classes=num_classes)
        apply_cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    elif enable_cutmix:
        apply_cutmix_or_mixup = v2.CutMix(num_classes=num_classes)

    elif enable_mixup:
        apply_cutmix_or_mixup = v2.MixUp(num_classes=num_classes)

    if apply_cutmix_or_mixup is not None:
        criterion = F.cross_entropy

    for epoch in range(epochs):
        all_train_loss = []
        all_train_acc = []

        model.train()
        for images, labels in tqdm(train_dl, desc=f"Epoch: {epoch + 1}"):
            images = images.to(device)
            labels = labels.to(device)

            if apply_cutmix_or_mixup is not None:
                images, labels = apply_cutmix_or_mixup(images, labels)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, dim=1)

            hard_labels = labels.argmax(dim=1) if labels.ndim > 1 else labels

            all_train_loss.append(loss.item())
            all_train_acc.append((preds == hard_labels).float().mean().item())

        train_loss = sum(all_train_loss) / len(all_train_loss)
        train_acc = sum(all_train_acc) / len(all_train_acc)

        all_val_loss = []
        all_val_acc = []
        all_val_labels = []
        all_val_preds = []

        model.eval()
        with torch.no_grad():
            for images, labels in valid_dl:
                images = images.to(device)
                labels = labels.to(device)

                if use_amp:
                    with torch.autocast(device.type):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, dim=1)
                hard_labels = labels.argmax(dim=1) if labels.ndim > 1 else labels

                all_val_loss.append(loss.item())
                all_val_acc.append((preds == hard_labels).float().mean().item())
                all_val_labels.extend(hard_labels.detach().cpu().tolist())
                all_val_preds.extend(preds.detach().cpu().tolist())

        valid_loss = sum(all_val_loss) / len(all_val_loss)
        valid_acc = sum(all_val_acc) / len(all_val_acc)

        if metric_name in {"loss", "val_loss", "valid_loss"}:
            current_metric = valid_loss
        else:
            current_metric = valid_acc

        if save_best_recall and disease_class_idx is not None:
            disease_recall = recall_score(all_val_labels, all_val_preds, pos_label=disease_class_idx)
            if disease_recall > best_disease_recall:
                best_disease_recall = disease_recall
                torch.save(model.state_dict(), os.path.join(save_dir, best_recall_name))
                print(f"New best saved - disease recall: {best_disease_recall:.4f}")
        else:
            disease_recall = None

        print(
            f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f}, Valid Accuracy: {valid_acc:.4f}"
            + (f", Disease Recall: {disease_recall:.4f}" if disease_recall is not None else "")
        )

        print(f"Epoch {epoch+1}: LR = {optimizer.param_groups[0]['lr']}")

        history.append({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "monitor_metric": current_metric,
            "disease_recall": disease_recall,
        })

        if scheduler is not None:
            if hasattr(scheduler, "step"):
                if hasattr(torch.optim.lr_scheduler, "ReduceLROnPlateau") and isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step(current_metric)
                else:
                    scheduler.step()

        if monitor_mode == "min":
            improved = current_metric < best_metric
        else:
            improved = current_metric > best_metric

        if use_early_stopping:
            if improved:
                best_metric = current_metric
                best_epoch = epoch
                counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                print(f"  New best saved - {early_stopping_metric}: {current_metric:.4f}")
            else:
                counter += 1
                print(f"  No improvement ({counter}/{patience})")
                if counter >= patience:
                    print("Early stopping triggered.")
                    break
            

    torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))

    best_path = os.path.join(save_dir, "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("Best Model Loaded Successfully!")
    else:
        print("No best model found, using last model state.")

    return model, history, best_epoch