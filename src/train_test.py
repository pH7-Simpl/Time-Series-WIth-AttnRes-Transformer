import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path
import os

def create_folder(path:str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

# ============================================================
# TRAINING FUNCTION - Saves model + config once
# ============================================================
def train_model(model, train_loader, val_loader, config, model_name, out_path:str | None = None):
    device = config['device']
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    parameters = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Parameters: {parameters:,}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        scheduler.step(avg_val)

        if epoch % 5 == 0 or epoch == config['epochs']-1:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s | Best Val Loss: {best_val_loss:.6f}")

    # ============================================================
    # SAVE ONCE: model weights + config + history
    # ============================================================
    save_dir = Path(create_folder(os.getenv("MODEL_PATH")))

    weights_path = save_dir / f"{model_name}_weights.pt"
    torch.save(model.state_dict(), weights_path)

    config_path = save_dir / f"{model_name}_config.json"
    outed_dir = create_folder(Path(out_path) / "model-config")
    outed_path = Path(outed_dir) / f"{model_name}_config.json"

    save_data = {
        'model_name': model_name,
        'parameters': f"{parameters:,}",
        'config': {k: v for k, v in config.items() if k != 'device'},
        'history': history,
        'best_val_loss': float(best_val_loss),
        'training_time': elapsed,
        'final_epoch': epoch
    }

    with open(config_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    with open(outed_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Saved to: {weights_path} and {config_path}")
    print(f"Cached to: {outed_path}")

    return model, history


# ============================================================
# LOAD SAVED MODEL
# ============================================================
def load_saved_model(model_class, model_kwargs, model_name, device='cpu'):
    """
    Load a saved model from disk.

    Args:
        model_class: The model class (e.g., LSTMModel)
        model_kwargs: Dict of kwargs to instantiate the class
        model_name: Name used when saving (e.g., 'LSTM')
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded model ready for evaluation
        history: Training history dict
    """
    save_dir = Path("saved_models")
    weights_path = save_dir / f"{model_name}_weights.pt"
    config_path = save_dir / f"{model_name}_config.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model not found: {weights_path}. Train first!")

    with open(config_path, 'r') as f:
        saved_data = json.load(f)

    model = model_class(**model_kwargs)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} from {weights_path}")
    print(f"  Best val loss: {saved_data['best_val_loss']:.6f}")
    print(f"  Trained epochs: {saved_data['final_epoch']}")

    return model, saved_data['history']


# ============================================================
# EVALUATION FUNCTION (uses loaded model)
# ============================================================
def evaluate_model(model, test_loader, scaler, config, model_name):
    device = config['device']
    model.eval()

    criterion = nn.MSELoss()
    predictions, actuals = [], []
    test_losses = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            test_losses.append(loss.item())

            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    def inverse_ot(y_scaled):
        N, H = y_scaled.shape
        result = np.zeros((N, H))
        for h in range(H):
            dummy = np.zeros((N, 7))
            dummy[:, 6] = y_scaled[:, h]
            inv = scaler.inverse_transform(dummy)
            result[:, h] = inv[:, 6]
        return result

    pred_inv = inverse_ot(predictions)
    actual_inv = inverse_ot(actuals)

    mae = np.mean(np.abs(pred_inv - actual_inv))
    rmse = np.sqrt(np.mean((pred_inv - actual_inv)**2))

    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Test Loss (scaled): {np.mean(test_losses):.6f}")
    print(f"MAE  (original °C): {mae:.4f}")
    print(f"RMSE (original °C): {rmse:.4f}")

    return {
        'model_name': model_name,
        'predictions': pred_inv,
        'actuals': actual_inv,
        'mae': mae,
        'rmse': rmse,
        'test_loss': np.mean(test_losses)
    }