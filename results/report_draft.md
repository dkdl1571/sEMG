## 모델 성능 비교 (Precision/Recall/F1 = macro 평균)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| SVM (RBF) | 0.8242 | 0.8235 | 0.8242 | 0.8228 |
| Random Forest | 0.8274 | 0.8263 | 0.8274 | 0.8264 |
| k-NN (k=5) | 0.7832 | 0.7857 | 0.7832 | 0.7824 |

- F1-score 가장 높은 모델: **Random Forest**
- F1-score 가장 낮은 모델: **k-NN (k=5)**

### SVM (RBF)
![SVM (RBF)](results/svm_cm.png)

- 가장 잘 분류된 클래스: A (재현율 0.989)
- 가장 많이 오분류된 클래스: C (재현율 0.700)
- 주요 오분류 유형: 실제 D → C로 예측 (40개)
- 오분류가 발생한 이유: (직접 작성)

### Random Forest
![Random Forest](results/random_cm.png)

- 가장 잘 분류된 클래스: A (재현율 0.989)
- 가장 많이 오분류된 클래스: C (재현율 0.663)
- 주요 오분류 유형: 실제 C → D로 예측 (42개)
- 오분류가 발생한 이유: (직접 작성)

### k-NN (k=5)
![k-NN (k=5)](results/knn_cm.png)

- 가장 잘 분류된 클래스: A (재현율 0.984)
- 가장 많이 오분류된 클래스: D (재현율 0.658)
- 주요 오분류 유형: 실제 D → C로 예측 (46개)
- 오분류가 발생한 이유: (직접 작성)
