## Attention Gate Residual UNet
程式練習...

基於原始 UNet 並結合 Attention Gate 與 Residual Block 的深度學習模型，用來分割靜脈影像(以下範例為手背靜脈)。

### 參考文獻
- UNet: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Modified UNet Code: [https://github.com/zhixuhao/unet](https://github.com/zhixuhao/unet)
- Attention Gate: [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
- Residual Block: [Road Extraction by Deep Residual U-Net](https://arxiv.org/abs/1711.10684)

### 壓縮檔內容
- `data/membrane` - 用於模型訓練、驗證與測試的影像。
- `main.py` - 主程式。
- `data_loader.py` - 訓練資料前處理與資料增生。
- `blocks.py` - 內為模型所用到的 block(像是 attention gates、residual blocks 等等...)。
- `model.py` - 模型主架構。
- `my_metrics.py` - 參考 stack overflow 或其他教學網站自定義的模型評估指標(像是 Dice、IoU 等等...)。
- `requirements.txt` - Python3.9.2 用到的函式庫及其版本。

## 資料庫
模型訓練資料庫為右側連結提供 [點此連結到資料庫](https://github.com/wilchesf/dorsalhandveins)。

該資料庫包含 1782 張手背靜脈影像。

實驗將資料庫拆分為 7:2:1 for 訓練、驗證與測試:
- 訓練: `data/membrane/train/image` 與 `data/membrane/train/label`
- 驗證: `data/membrane/val/val_image` 與 `data/membrane/val/val_label`
- 測試: `data/membrane/test`

記得將所有影像 resize 為 256×256。

## 模型架構 (點擊縮圖可放大)
- 主架構:
  
![main](image/0.png)


- 殘差塊(Residual block):

![Residual](image/1.png)


- 注意力門(Attention gate):

![Attention](image/2.png)


## 📊 測試結果 (點擊縮圖可放大)

![metrics](image/3.png)

- 模型測試階段預測範例:

| Input Image | Prediction Image |
|-------------|------------------|
| ![input](data/membrane/test/4.png) | ![prediction](data/membrane/test/4_predict.png) |

## 如何使用
請輸入以下指令建置 Python3.9.2 環境用到的函式庫及其版本:
```
pip install -r .\requirements.txt
```
將` main.py` 的 image path 替換為您的資料庫路徑，並輸入以下指令執行程式進行模型訓練:
```
python .\main.py 
```
