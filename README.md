# CelebA Super-Resolution Data Preparation

##  Project Description
This project prepares the **CelebA dataset** for training a **Super-Resolution model**.  
It downloads the dataset from **Kaggle**, extracts aligned face images, generates **low-resolution (LR)** and **high-resolution (HR)** pairs, and splits them into **training**, **validation**, and **test** sets.

This phase focuses on **data preparation and visualization**.  
In the **next phase**, we will use the **SRGAN (Super-Resolution Generative Adversarial Network)** model to train and generate higher-quality, realistic HR images from LR inputs.

---

##  Team Information:
- **Team Name:** PixelForge.  
- **Members:**
- 
  - Yosr Sellami LVYMK8
  - Liang Wenlong DGED6M
  - Lu Yijia DX29TC
  - Yarmammadova Aysel Q7K238


##  Dataset Source
**Dataset:** [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

The **CelebA (CelebFaces Attributes Dataset)** contains more than **200,000 aligned celebrity face images** with detailed annotations.  
Each image has a consistent face alignment and resolution of 178×218 pixels — ideal for face recognition, attribute prediction, and **super-resolution** tasks.

---

## Data Preparation Steps

### 1️: Downloading the Dataset
- The dataset is downloaded automatically using the **Kaggle API** (`kaggle datasets download`).
- A `kaggle.json` API key must be uploaded in Colab for authentication.
- Files are extracted to the directory:
  /content/celeba

---

### 2️: Data Exploration
- The script scans directories using `os.walk` and `glob` to locate `.jpg` images.
- A sample of images is displayed using **Matplotlib** to confirm successful extraction.

---

### 3️: Data Transformation
Each image is transformed as follows:
- **High-Resolution (HR):** resized to **128×128** pixels.  
- **Low-Resolution (LR):** created by downscaling HR images by a factor of 2 → **64×64** pixels.  
- Both HR and LR images are converted to **PyTorch tensors** and normalized.

---

### 4️: Visualization
To confirm correct preprocessing:
- Eight LR↔HR image pairs are visualized side by side.  
-  result:
- Total images in dataset: 202599
LR batch: (16, 3, 64, 64) HR batch: (16, 3, 128, 128)

---

### 5️: Splitting the Dataset
Processed tensors are divided into:
-  **Training set**
-  **Validation set**
-  **Test set**

Each split is stored as `.pt` tensor files for faster data loading.  

 output:
✔ Saved train: 182341 items into 45 shard(s)
✔ Saved val: 10129 items into 5 shard(s)
✔ Saved test: 10129 items into 5 shard(s)
All done. Shards at: /content/celeba/final_tensors_sharded_fast


---

##  How to Run

### **Run on Google Colab**
1. Upload  **kaggle.json** file when prompted.  
2. Execute all notebook cells sequentially.  
3. Follow processing steps below.

### **Processing Steps**
1. Dataset download and extraction  
2. Image resizing and tensor conversion  
3. Visualization of LR↔HR pairs  
4. Dataset splitting and saving  

### **Outputs**
- Preprocessed LR↔HR tensors ready for training  
- Metadata manifests for `train`, `val`, and `test` splits  

---

##  Next Steps: SRGAN Model
In the next phase, this prepared dataset will be used to train a **Super-Resolution Generative Adversarial Network (SRGAN)** to reconstruct high-quality HR images from LR inputs.

---





