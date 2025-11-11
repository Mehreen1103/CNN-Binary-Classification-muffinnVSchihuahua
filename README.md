# 🧠 Muffin vs Chihuahua Image Classification Using CNN

This project implements and fine-tunes various **Convolutional Neural Network (CNN)** architectures to distinguish between **muffins** 🧁 and **chihuahuas** 🐶 — a popular example of a challenging visual classification task due to their visual similarity.

The goal was to:
1. Experiment with different CNN architectures (both **custom** and **pre-trained**),
2. Apply multiple **image augmentation** techniques,
3. Optimize model hyperparameters,
4. Evaluate model performance using the **ROC-AUC curve**,
5. Compare CNN performance against a previous **fully connected Neural Network (NN)** model using metrics such as accuracy, precision, recall, and F1-score.

---

## 📂 Dataset

**Source:** [Muffin vs Chihuahua Dataset (Kaggle)](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/data)

The dataset contains labeled images of muffins and chihuahuas for binary image classification.

| Class | Example Count | Example Image |
|:------|:---------------|:---------------|
| Muffin 🧁 | ~250 | <img src="https://storage.googleapis.com/kaggle-datasets-images/1665289/2734384/59a37f4da4b4a35b5d1b4e1a07de4dd5/dataset-cover.jpg?t=2021-09-23T19%3A19%3A29Z" width="100"/> |
| Chihuahua 🐶 | ~250 | same dataset source |

---

## 🧩 CNN Architectures Explored

- **Custom CNN** (2–3 convolutional layers with ReLU and MaxPooling)
- **VGG16** (pre-trained on ImageNet, fine-tuned for binary classification)
- **ResNet50** (fine-tuned)
- **MobileNetV2** (lightweight architecture for comparison)

Each model was fine-tuned and evaluated on the same dataset split to ensure fair comparison.

---

## 🧪 Data Augmentation Techniques

At least three augmentation techniques were applied to improve generalization:

- **Rotation:** ±15 degrees  
- **Horizontal Flip**  
- **Zoom & Width Shift**

Implemented using Keras `ImageDataGenerator` or `tf.keras.preprocessing.image_dataset_from_directory`.

---

## ⚙️ Hyperparameter Tuning

The following hyperparameters were adjusted:
- Learning rate: `1e-3 → 1e-5`
- Batch size: `16 / 32`
- Optimizers: `Adam`, `RMSprop`
- Epochs: `15–30`

Best model performance was obtained with **VGG16** using a reduced learning rate and early stopping based on validation loss.

---

## 📈 Evaluation Metrics

### 🧮 Metrics Used
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**

### 🧾 Sample Results
| Model         | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------|-----------|------------|----------|-----------|----------|
| Custom CNN     | 0.85      | 0.84       | 0.83     | 0.83      | 0.88     |
| VGG16 (Fine-tuned) | **0.92** | **0.91**  | **0.90** | **0.90**  | **0.95** |
| ResNet50       | 0.89      | 0.87       | 0.88     | 0.87      | 0.93     |
| MobileNetV2    | 0.87      | 0.86       | 0.85     | 0.85      | 0.91     |

---

## 📊 ROC-AUC Curve

The ROC-AUC curve was used to determine the **optimal operating point** for the binary classification task.

![ROC Curve](results/roc_auc_curve.png)

---

## 🔍 Comparative Analysis

A comparison between the best CNN (VGG16) and the previously implemented **fully connected NN model** was made.

| Metric | CNN (VGG16) | Previous NN |
|:-------|:-------------|:-------------|
| Accuracy | **0.92** | 0.81 |
| F1-Score | **0.90** | 0.79 |
| ROC-AUC | **0.95** | 0.86 |

The CNN significantly outperformed the previous NN model in all key metrics, demonstrating the advantage of convolutional feature extraction in visual tasks.

---

## ⚒️ Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn

---

## 📁 Repository Structure

