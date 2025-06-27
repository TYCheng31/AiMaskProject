# AiMaskProject
## 專題動機
* 偵測人員是否正確配戴口罩
  * 至今還是有些人戴口罩會把鼻子露出來或是佩戴在下巴
* 製作無人警示系統
  * 疫情期間常需要額外請一個人檢查人員口罩是否配戴正確，雖然現在沒有強制戴口罩了，但是我們希望如果在場所內超出一定比例沒戴口罩可以自動發出警示，提醒大家暴露於風險中。
* 傳送分析報告給用戶
  * 我們希望利用收集到的數據整理出分析報告，報告中顯示各個時段配戴口罩的百分比，可以分析出哪個時段風險最高。

## 架構
<img src="https://github.com/user-attachments/assets/b3279d30-c62a-4dd2-afa3-9c2e21b88737" alt="image" width="800" />

## 模型訓練  
* Roboflow資料集標註
* 資料集預處理、擴增
* YOLOv8-l模型訓練

### 資料集  
* 手動標記 848 張圖片並擴增至 2036 張  
  * 隨機旋轉(-45度~45度)
  * 隨機亮度(-15%~15%)
  * 隨機加入雜訊(每個像素0.15%)
<img src="https://github.com/user-attachments/assets/19feef47-5665-4d04-a6aa-6779994c15b8" alt="image" width="800" />


* 類別:
  * 沒有正確配戴口罩
  * 有配戴口罩
  * 無配戴口罩
<img src="https://github.com/user-attachments/assets/b05eadad-40f9-4888-acac-b9be84533db3" alt="image" width="800" />  

### 驗證集結果  
<img src="https://github.com/user-attachments/assets/9f48398e-ac8c-4876-9cb2-f2231b005d53" alt="image" width="800" />  

### 訓練結果  
<img src="https://github.com/user-attachments/assets/e1333b4d-078b-4950-8d41-773f93629dfd" alt="image" width="800" /> 

### 混淆矩陣
<img src="https://github.com/user-attachments/assets/072fb3b6-35d1-4693-9766-86a5393b4324" alt="image" width="800" />  

* YOLOv8-l (100 epoch)
  * mAP@0.5：0.677
  * Avg Precision：1.00
  * Avg Recll：0.85
  * F1  分數：0.68

## 介面  
*未超出閾值  
<img src="https://github.com/user-attachments/assets/6c4539d4-f5b7-46cb-9934-194cc402ea5e" alt="image" width="800" />  

*超出閾值  
<img src="https://github.com/user-attachments/assets/aef14310-7e80-4387-9c0f-1ead891b8cfe" alt="image" width="800" />


