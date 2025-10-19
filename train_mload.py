import os
import sys
import yaml
import argparse
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import load_config
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from core.load_estimator import Config, DataPreprocessor, MloadModel, train_model, save_checkpoint

def load_yaml_dataset(yaml_path: str):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'数据集文件不存在: {yaml_path}')
    with open(yaml_path, 'r') as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f'数据集文件为空: {yaml_path}')
        data = yaml.safe_load(content) or {}
    samples = data.get('samples', [])
    labels = data.get('labels', [])
    if not labels and samples:
        labels = [float(s.get('training_time', 0.0)) for s in samples]
    if not samples:
        raise ValueError('未找到任何样本（samples 为空）')
    if not labels:
        raise ValueError('未找到任何标签（labels 为空），且无法从样本推断')
    if len(samples) != len(labels):
        raise ValueError(f'样本数({len(samples)})与标签数({len(labels)})不一致')
    return (samples, labels)

def main():
    parser = argparse.ArgumentParser(description='Train Mload from YAML dataset')
    parser.add_argument('--data', type=str, default=os.path.join(project_root, 'dataset', 'Mload_dataset', 'fastgnas-a6000-batch.yaml'), help='输入数据集YAML路径 (默认: dataset/Mload_dataset/fastgnas-a6000-batch.yaml)')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率 (默认: 0.0005)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例 (0-1之间，默认: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--acc_tol', type=float, default=0.2, help='测试精度容忍度，预测在真实值±(tol*真实值)内视为正确 (默认: 0.2，即20%)')
    parser.add_argument('--batch_size', type=int, default=512, help='训练批大小 (默认: 64)')
    parser.add_argument('--max_epochs', type=int, default=400, help='最大训练轮数 (默认: 400)')
    parser.add_argument('--output', type=str, default=os.path.join(project_root, 'mload_checkpoint.pth'), help='模型保存路径 (默认: ./mload_checkpoint.pth)')
    args = parser.parse_args()
    train_cfg_dir = f'./configs/train.yaml'
    train_cfg = load_config(train_cfg_dir)
    device_type = train_cfg['training']['data_collection']['device_type']
    print(f'加载数据集: {args.data}')
    samples, labels = load_yaml_dataset(args.data)
    print(f'数据样本数: {len(samples)}')
    config = Config()
    preprocessor = DataPreprocessor(config)
    X_train, y_train = preprocessor.fit_transform(samples, labels)
    input_dim = int(X_train.shape[1])
    print(f'输入特征维度: {input_dim}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'使用设备: {device}')
    num_samples = X_train.shape[0]
    print(f'总样本数 (用于训练)：{num_samples}')
    X_tr = X_train.to(device)
    y_tr = y_train.to(device)
    X_te = None
    y_te = None
    model = MloadModel(input_dim=input_dim).to(device)
    y_tr_log = torch.log1p(torch.clamp(y_tr, min=0.0))
    y_te_log = torch.log1p(torch.clamp(y_te, min=0.0)) if y_te is not None else None
    batch_size = int(max(1, min(1024, int(args.batch_size)))) if hasattr(args, 'batch_size') else min(256, max(16, X_tr.shape[0]))
    train_ds = TensorDataset(X_tr, y_tr_log)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val = math.inf
    patience = 10
    wait = 0
    max_epochs = int(max(50, min(int(args.max_epochs), X_tr.shape[0]) if hasattr(args, 'max_epochs') else max(50, X_tr.shape[0])))
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= X_tr.shape[0]
        if X_te is not None and y_te_log is not None and (X_te.shape[0] > 0):
            model.eval()
            with torch.no_grad():
                val_pred = model(X_te)
                val_loss = torch.mean((val_pred - y_te_log) ** 2).item()
        else:
            val_loss = epoch_loss
        scheduler.step(val_loss)
        if val_loss < best_val - 1e-06:
            best_val = val_loss
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            print(f'早停: 无改进 {patience} 轮，停止训练。最佳验证损失: {best_val:.6f}')
            break
    if 'best_state' in locals():
        model.load_state_dict(best_state)
    test_ratio = max(0.0, min(1.0, float(args.test_ratio)))
    num_test = int(round(num_samples * test_ratio))
    if num_test > 0:
        g = torch.Generator()
        g.manual_seed(int(args.seed))
        perm = torch.randperm(num_samples, generator=g)
        test_idx = perm[:num_test]
        X_te = X_train[test_idx].to(device)
        y_te = y_train[test_idx].to(device)
        model.eval()
        with torch.no_grad():
            y_pred_log = model(X_te)
            y_pred = torch.expm1(y_pred_log)
            y_te_log = torch.log1p(torch.clamp(y_te, min=0.0)) if y_te is not None else None
        y_true = torch.expm1(y_te_log) if y_te_log is not None else y_te
        mse = torch.mean((y_pred - y_true) ** 2).item()
        mae = torch.mean(torch.abs(y_pred - y_true)).item()
        if y_te_log is not None:
            mse_log = torch.mean((y_pred_log - y_te_log) ** 2).item()
            mae_log = torch.mean(torch.abs(y_pred_log - y_te_log)).item()
        else:
            mse_log = float('nan')
            mae_log = float('nan')
        tol = max(0.0, float(args.acc_tol))
        denom = torch.clamp(torch.abs(y_te), min=1e-08)
        correct = (torch.abs(y_pred - y_true) <= tol * denom).float().mean().item()
        print(f'评估样本数: {num_test}，测试集: MSE={mse:.6f}, MAE={mae:.6f}, Accuracy@{tol * 100:.1f}%={correct * 100:.2f}%')
        print(f'log-space: MSE_log={mse_log:.6f}, MAE_log={mae_log:.6f} (在 log1p 空间)')

    def _evaluate_on_test(model_to_eval, X_eval, y_eval):
        if X_eval is None or y_eval is None or X_eval.shape[0] == 0:
            return (float('nan'), float('nan'), float('nan'), float('nan'))
        model_to_eval.eval()
        with torch.no_grad():
            y_pred_log_tmp = model_to_eval(X_eval)
            y_pred_tmp = torch.expm1(y_pred_log_tmp)
            y_te_log_tmp = torch.log1p(torch.clamp(y_eval, min=0.0))
        y_true_tmp = torch.expm1(y_te_log_tmp)
        mse_tmp = torch.mean((y_pred_tmp - y_true_tmp) ** 2).item()
        mae_tmp = torch.mean(torch.abs(y_pred_tmp - y_true_tmp)).item()
        mse_log_tmp = torch.mean((y_pred_log_tmp - y_te_log_tmp) ** 2).item()
        tol_tmp = max(0.0, float(args.acc_tol))
        denom_tmp = torch.clamp(torch.abs(y_eval), min=1e-08)
        acc_tmp = (torch.abs(y_pred_tmp - y_true_tmp) <= tol_tmp * denom_tmp).float().mean().item()
        return (mse_log_tmp, mse_tmp, mae_tmp, acc_tmp)
    new_mse_log = locals().get('mse_log', float('nan'))
    new_mse = locals().get('mse', float('nan'))
    new_mae = locals().get('mae', float('nan'))
    new_acc = locals().get('correct', float('nan'))
    if os.path.exists(args.output):
        try:
            old_ckpt = torch.load(args.output, map_location=device, weights_only=True)
            old_model = MloadModel(input_dim=input_dim).to(device)
            try:
                old_model.load_state_dict(old_ckpt['model_state_dict'])
                old_mse_log, old_mse, old_mae, old_acc = _evaluate_on_test(old_model, X_te, y_te)
                print(f'已检测到旧模型: {args.output}，旧模型 log-space MSE={old_mse_log:.6f}, 原尺度 MSE={old_mse:.6f}')
                prefer_new = False
                if not math.isnan(new_mse_log) and (not math.isnan(old_mse_log)):
                    prefer_new = new_mse_log <= old_mse_log
                elif math.isnan(new_mse_log) and math.isnan(old_mse_log):
                    prefer_new = True
                elif not math.isnan(new_mse) and (not math.isnan(old_mse)):
                    prefer_new = new_mse <= old_mse
                else:
                    prefer_new = True
                if prefer_new:
                    save_checkpoint(model, preprocessor, args.output)
                    print(f'新模型优于或等于旧模型，已保存到: {args.output}')
                else:
                    print(f'旧模型保留（旧模型优于新模型），未替换: {args.output}')
            except Exception as e_load:
                print(f'警告：无法加载旧模型 checkpoint（{args.output}），原因: {e_load}。将覆盖保存新模型。')
                save_checkpoint(model, preprocessor, args.output)
                print(f'模型已保存到: {args.output}')
        except Exception as e:
            print(f'读取旧模型时出错，原因: {e}。直接保存新模型。')
            save_checkpoint(model, preprocessor, args.output)
            print(f'模型已保存到: {args.output}')
    else:
        save_checkpoint(model, preprocessor, args.output)
        print(f'模型已保存到: {args.output}')
if __name__ == '__main__':
    main()