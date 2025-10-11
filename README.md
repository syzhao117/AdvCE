
# AdvCE: Hellinger-based Training on News & TCGA

This repository provides code to train and evaluate our Hellinger-based method (**AdvCE**) on two datasets: **News** and **TCGA**.
The **proofs** and a **training-framework pseudocode sketch** are included in `ICASSP2026_ITE_supplementary.pdf`.


## 1) Data Preparation

### 1.1 News

* The **processed data files are already provided**. No extra downloads or preprocessing are required.
* Keep the files in the repository’s default data directory (the training script will read them automatically).

### 1.2 TCGA

1. Obtain the **raw TCGA data** via :
   [https://paperdatasets.s3.amazonaws.com/tcga.db]
2. After downloading, run:

   ```bash
   python process_tcga.py
   ```

   This script **generates the training-ready files** in the default data directory used by our training code.

> If you use a custom data path, adjust paths in the script or training arguments accordingly.

---

## 2) Quick Start

### 2.1 Train on **News**

```bash
python main_Hellinger_pz.py \
  --setting news \
  --num_epochs 600 \
  --beta 0.1 \
  --k_critic 100 \
  --k_shuffle 10 \
  --score_clip 5.0 \
  --save_path models_news/AdvCE_news.pth
```

### 2.2 Train on **TCGA**

```bash
python main_Hellinger_pz.py \
  --setting tcga \
  --num_epochs 600 \
  --beta 0.1 \
  --k_critic 75 \
  --k_shuffle 5 \
  --score_clip 5.0 \
  --save_path models_tcga/AdvCE_tcga.pth
```

---

## 3) Key Arguments

* `--setting {news|tcga}`: dataset-specific configuration.
* `--num_epochs`: total training epochs (e.g., 600).
* `--beta`: weight for the Hellinger/adversarial term (e.g., 0.1).
* `--k_critic`: critic (discriminator) update steps per iteration (News=100, TCGA=75 in our runs).
* `--k_shuffle`: shuffle/negative-sampling related steps (News=10, TCGA=5).
* `--score_clip`: clipping bound for scores/gradients (e.g., 1.0).
* `--save_path`: output path for checkpoints.

---

## 4) Logs & Outputs

Typical console logs:

```
Epoch [600/600] — loss: 2.4821 | MSE: 2.4944 | Hellinger(inner/outer J): {J_inner_avg:.4f}/0.1229
Model saved to: models_news/AdvCE_news.pth
MISE: 2.466...
PE:   2.492...
```

```
Epoch [600/600] — loss: -0.0990 | MSE: 0.0154 | Hellinger(inner/outer J): {J_inner_avg:.4f}/1.1440
Model saved to: models_tcga/AdvCE_tcga.pth
TCGA MISE: 0.546...
TCGA PE:   0.387...
```

* **Metrics**:

  * News: `MISE`, `PE`
  * TCGA: `MISE`, `PE`
* **Checkpoints** are written to the path given by `--save_path`.
