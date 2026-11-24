from __future__ import annotations
import argparse, csv, random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from src.data.dataset import FundusDataset, load_or_create_class_names
from src.models.build_model import build_model

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def compute_class_weights(dataset: FundusDataset, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.maximum(np.bincount([lbl for _, lbl in dataset.samples], minlength=num_classes).astype(np.float32), 1.0)
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)

def run_epoch(model, loader, criterion, optimizer, device, scaler):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if is_train: optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, targets)
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    avg_loss = total_loss / max(len(all_targets), 1)
    acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    macro_f1 = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))
    return avg_loss, acc, macro_f1

def train_architecture(cfg: dict, arch: str, fast: bool = False) -> float:
    seed_everything(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training {arch} on {device} (fast={fast})...', flush=True)
    split_dir, models_dir, reports_dir = Path(cfg['data']['split_dir']), Path('models'), Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_or_create_class_names(split_dir, models_dir / 'class_names.json')
    num_classes = len(class_names)
    max_samples = 8 if fast else None
    train_ds = FundusDataset(split_dir / 'train', class_names, cfg['data']['image_size'], cfg['data']['clahe'], cfg['data']['green_channel'], True, max_samples)
    val_ds = FundusDataset(split_dir / 'val', class_names, cfg['data']['image_size'], cfg['data']['clahe'], cfg['data']['green_channel'], False, max_samples)
    workers = 0 if fast else cfg['training'].get('workers', 2)
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=workers)
    model = build_model(arch, num_classes=num_classes, pretrained=cfg['model']['pretrained']).to(device)
    weights = compute_class_weights(train_ds, num_classes, device) if cfg['training']['class_weighted_loss'] else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg['training']['label_smoothing'])
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    epochs_head = 1 if fast else (0 if arch == 'baseline_cnn' else cfg['training']['epochs_head'])
    epochs_ft = 1 if fast else cfg['training']['epochs_finetune']
    patience = cfg['training']['patience']
    history, best_f1, stale = [], -1.0, 0
    ckpt_path = models_dir / ('baseline_best.pt' if arch == 'baseline_cnn' else 'best.pt')
    if epochs_head > 0:
        for name, param in model.named_parameters(): param.requires_grad = any(k in name for k in ('classifier', 'fc', 'head'))
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['training']['lr_head'], weight_decay=cfg['training']['weight_decay'])
        for epoch in range(1, epochs_head + 1):
            tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
            va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, None, device, None)
            print(f'[{arch} | Head {epoch}/{epochs_head}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}', flush=True)
            history.append({'phase': 'head', 'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'train_f1': tr_f1, 'val_loss': va_loss, 'val_acc': va_acc, 'val_f1': va_f1})
            if va_f1 > best_f1:
                best_f1 = va_f1; torch.save({'architecture': arch, 'state_dict': model.state_dict(), 'val_f1': best_f1}, ckpt_path)
    for param in model.parameters(): param.requires_grad = True
    lr_ft = cfg['training']['lr_head'] if arch == 'baseline_cnn' else cfg['training']['lr_finetune']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_ft, weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs_ft, 1))
    for epoch in range(1, epochs_ft + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()
        print(f'[{arch} | FT {epoch}/{epochs_ft}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}', flush=True)
        history.append({'phase': 'finetune', 'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'train_f1': tr_f1, 'val_loss': va_loss, 'val_acc': va_acc, 'val_f1': va_f1})
        if va_f1 > best_f1:
            best_f1 = va_f1; stale = 0; torch.save({'architecture': arch, 'state_dict': model.state_dict(), 'val_f1': best_f1}, ckpt_path)
        else:
            stale += 1
            if stale >= patience: print(f'Early stopping {arch} at epoch {epoch}'); break
    with (reports_dir / f'training_log_{arch}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['phase', 'epoch', 'train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1'])
        w.writeheader(); w.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['train_loss'] for r in history], label='train'); axes[0].plot([r['val_loss'] for r in history], label='val'); axes[0].set_title(f'{arch} Loss'); axes[0].legend()
    axes[1].plot([r['train_f1'] for r in history], label='train F1'); axes[1].plot([r['val_f1'] for r in history], label='val F1'); axes[1].set_title(f'{arch} Macro-F1'); axes[1].legend()
    fig.tight_layout(); fig.savefig(reports_dir / 'figures' / f'training_curves_{arch}.png', dpi=150); plt.close(fig)
    print(f'Saved {ckpt_path} (best val macro-F1={best_f1:.4f})')
    return best_f1

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--arch', default=None)
    p.add_argument('--fast', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train_architecture(cfg, arch=(args.arch or cfg['model']['architecture']), fast=args.fast)

if __name__ == '__main__': main()
