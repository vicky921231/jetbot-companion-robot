# -*- coding: utf-8 -*-
"""碰撞避障二分類模型訓練。

用法(在你的筆電,Windows PowerShell):
    python train_collision.py --data dataset
    python train_collision.py --data dataset --arch alexnet --epochs 20

相對於官方 train_model.ipynb 修掉的問題:

1. **印出並檢查類別編號**。PyTorch 的 ImageFolder 依資料夾名稱字母排序編號,
   blocked=0、free=1。推論端的 `F.softmax(...)[0]` 取的就是 blocked。
   資料夾一旦改名(例如 safe/danger),編號會反過來而且不會報錯,
   機器人會變成「看到路就停、看到牆就衝」。所以這裡強制檢查。

2. **測試集改成比例分割**。官方寫死 `[len(dataset) - 50, 50]`,
   資料少於 51 張會直接崩潰,資料很多時 50 張又太少。

3. **分層抽樣**。官方的隨機分割在資料不平衡時,測試集可能整組都是同一類,
   準確率會變得沒有意義。

4. **印出混淆矩陣**。只看準確率會掩蓋錯誤的方向。
   對避障來說「把有障礙判成安全」遠比反過來危險——前者會撞上去。

5. **檢查類別平衡**。兩類數量差太多時,模型只要全猜多數類就有高準確率。

6. **存下類別對照表**。跟模型放在一起,推論端可以驗證。
"""

import argparse
import json
import os
import random

# torch 與 conda 的 OpenMP 會衝突,要在 import torch 之前設
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# 推論端假設的順序。改資料夾名稱就會對不上,所以在這裡擋下來。
EXPECTED_CLASSES = {'blocked': 0, 'free': 1}

# 輸入尺寸。4:3,與相機輸出比例一致(感測器讀出 816x616 ≈ 1.325:1)。
#
# ⚠️ 這兩個數字必須與另外兩個地方一致,否則模型在推論時看到的畫面
#    形狀與訓練時不同,準確率會無聲無息地掉下來:
#      1. data_collection_plus.ipynb 的 Camera.instance(width=224, height=168)
#      2. live_demo_following_sm.ipynb 的 preprocess() 裡 cv2.resize(x, (224, 168))
#
#    注意兩邊的參數順序相反:torchvision 的 Resize 吃 (高, 寬),
#    cv2.resize 吃 (寬, 高)。
INPUT_H, INPUT_W = 168, 224


def build_transforms():
    """訓練用的前處理。

    水平翻轉是安全的資料增強:障礙物在左邊或右邊,對「前方通不通」
    這個二分類來說意義相同,等於免費把資料量變兩倍。
    正規化的數值沿用 ImageNet 的統計量,要與推論端一致。
    """
    return transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((INPUT_H, INPUT_W)),   # (高, 寬)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def stratified_split(targets, test_ratio, seed):
    """依類別分層抽樣,確保測試集兩類都有。

    官方用的 random_split 在資料不平衡時,測試集可能整組同一類,
    這時準確率不管多高都沒有意義。
    """
    by_class = {}
    for idx, label in enumerate(targets):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_idx, test_idx = [], []
    for label, idxs in sorted(by_class.items()):
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_ratio)))
        n_test = min(n_test, len(idxs) - 1)      # 至少留一張給訓練
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
    return train_idx, test_idx


def build_model(arch, num_classes=2):
    """遷移學習:載入預訓練權重,只把最後一層換成 2 類輸出。

    第一次執行會從網路下載預訓練權重,需要連線。
    """
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.classifier[6] = torch.nn.Linear(
            model.classifier[6].in_features, num_classes)
    else:
        raise ValueError('不支援的架構:%s' % arch)
    return model


def evaluate(model, loader, device):
    """回傳 (準確率, 混淆矩陣)。

    混淆矩陣 confusion[真實][預測],所以 confusion[0][1] 是
    「實際被擋住,卻被判成安全」——這是會撞上去的那一種錯誤。
    """
    model.eval()
    confusion = [[0, 0], [0, 0]]
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            for truth, pred in zip(labels.tolist(), preds.tolist()):
                confusion[truth][pred] += 1
    total = sum(sum(row) for row in confusion)
    correct = confusion[0][0] + confusion[1][1]
    return (correct / total if total else 0.0), confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset', help='資料集資料夾(內含 blocked/ 與 free/)')
    ap.add_argument('--arch', default='resnet18', choices=['resnet18', 'alexnet'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--test-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None, help='輸出檔名,預設 best_model_<arch>.pth')
    args = ap.parse_args()

    out_path = args.out or ('best_model_%s.pth' % args.arch)
    torch.manual_seed(args.seed)

    # ---------- 載入資料 ----------
    if not os.path.isdir(args.data):
        raise SystemExit('找不到資料夾:%s' % args.data)

    dataset = datasets.ImageFolder(args.data, build_transforms())

    print('=' * 60)
    print('類別編號對照:', dataset.class_to_idx)
    if dataset.class_to_idx != EXPECTED_CLASSES:
        raise SystemExit(
            '\n[停止] 類別編號與推論端的假設不符。\n'
            '  預期:%s\n  實際:%s\n'
            '  推論端用 softmax(...)[0] 取「被擋住」的機率,\n'
            '  所以資料夾必須叫 blocked 和 free。改名會讓機器人行為完全相反。'
            % (EXPECTED_CLASSES, dataset.class_to_idx))

    counts = {}
    for label in dataset.targets:
        name = dataset.classes[label]
        counts[name] = counts.get(name, 0) + 1
    print('各類數量:', counts, ' 合計', len(dataset))

    if len(dataset) < 100:
        print('\n[警告] 總數少於 100 張,模型很可能學不起來。官方建議每類至少 100 張。')

    if counts:
        most, least = max(counts.values()), min(counts.values())
        if least == 0:
            raise SystemExit('[停止] 有一類是空的,無法訓練。')
        if most / least > 2.0:
            print('\n[警告] 兩類數量相差超過兩倍(%d vs %d)。' % (most, least))
            print('        模型只要全猜多數類就有高準確率,這種「高分」沒有意義。')
            print('        建議把少的那一類補到接近。')

    # ---------- 分割 ----------
    train_idx, test_idx = stratified_split(dataset.targets, args.test_ratio, args.seed)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    print('訓練 %d 張 / 測試 %d 張(分層抽樣,兩類都有)' % (len(train_set), len(test_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch, shuffle=False, num_workers=0)

    # ---------- 模型 ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('裝置:%s　架構:%s　輸入 %dx%d(寬x高,4:3)'
          % (device, args.arch, INPUT_W, INPUT_H))
    if device.type == 'cpu':
        print('（CPU 訓練，數百張圖大約需要十幾到三十分鐘）')
    model = build_model(args.arch).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # ---------- 訓練 ----------
    print('=' * 60)
    best_accuracy = 0.0
    best_confusion = None

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

        accuracy, confusion = evaluate(model, test_loader, device)
        missed = confusion[0][1]          # 被擋住卻判成安全 = 會撞上去
        flag = '  ← 最佳' if accuracy > best_accuracy else ''
        print('epoch %2d/%d　準確率 %.3f　漏報 %d 張%s'
              % (epoch + 1, args.epochs, accuracy, missed, flag))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_confusion = confusion
            torch.save(model.state_dict(), out_path)

    # ---------- 結果 ----------
    print('=' * 60)
    print('最佳準確率:%.3f　模型已存到 %s' % (best_accuracy, out_path))

    if best_confusion:
        b2b, b2f = best_confusion[0]
        f2b, f2f = best_confusion[1]
        print('\n混淆矩陣（列=實際，行=預測）')
        print('              判成 blocked   判成 free')
        print('  實際 blocked    %6d      %6d' % (b2b, b2f))
        print('  實際 free       %6d      %6d' % (f2b, f2f))
        print('\n  漏報 %d 張：實際被擋住卻判成安全 → 會撞上去，這是危險的錯誤' % b2f)
        print('  誤報 %d 張：實際安全卻判成被擋住 → 會多繞路，不危險但影響跟隨' % f2b)
        if b2f > 0:
            print('\n[注意] 漏報不是零。補拍那些會漏判的場景，重新訓練。')

    meta = {
        'arch': args.arch,
        'class_to_idx': dataset.class_to_idx,
        'input_height': INPUT_H,
        'input_width': INPUT_W,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'best_accuracy': round(best_accuracy, 4),
        'confusion': best_confusion,
        'counts': counts,
    }
    meta_path = os.path.splitext(out_path)[0] + '_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('\n設定檔已存到 %s（推論端可以對照，避免類別順序弄反）' % meta_path)


if __name__ == '__main__':
    main()
