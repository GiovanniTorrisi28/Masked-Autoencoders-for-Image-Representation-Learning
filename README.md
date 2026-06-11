# Masked Autoencoders per l'Apprendimento di Rappresentazioni Visive

[![Report](https://img.shields.io/badge/Report-REPORT_IT.md-blue)](docs/REPORT_IT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👥 Informazioni Gruppo e Progetto
- **ID Gruppo**: G21
- **ID Progetto**: 11
- **Studente**: Giovanni Torrisi

## 📝 Descrizione del Progetto

Implementazione di un **Masked Autoencoder (MAE)** per l'apprendimento auto-supervisionato di rappresentazioni visive su ImageNet100. Un Vision Transformer (ViT-Base/16) viene pre-addestrato mascherando il 75% delle patch dell'immagine e ricostruendo i pixel mancanti, apprendendo rappresentazioni visive ricche senza utilizzare etichette. L'encoder pre-addestrato viene poi valutato tramite linear probing e confrontato con un ViT supervisionato addestrato da zero. Uno studio aggiuntivo sulla data efficiency analizza il comportamento di entrambi gli approcci quando sono disponibili solo il 10% o il 20% dei dati di training etichettati.

> 📖 **Report Ufficiale**: Per i dettagli teorici, l'analisi dell'architettura e la discussione dei risultati, consultare **[REPORT.md](docs/REPORT.md)**.

---

## 🛠 Riproducibilità Tecnica

### 1. Configurazione dell'Ambiente

**Requisiti**: Python 3.10+, GPU con supporto CUDA raccomandata.

```bash
git clone https://github.com/GiovanniTorrisi28/Masked-Autoencoders-for-Image-Representation-Learning.git
cd Masked-Autoencoders-for-Image-Representation-Learning
pip install -r requirements.txt
```

### 2. Dataset

Utilizzare lo script fornito per scaricare, estrarre e riorganizzare automaticamente **ImageNet100** da Kaggle:

```bash
python src/datasets/download_imagenet100.py --extract
```

**Credenziali Kaggle richieste** — una delle seguenti opzioni:
- Posizionare `kaggle.json` in `~/.kaggle/kaggle.json` (configurazione standard Kaggle)
- Passare `--kaggle-json /percorso/kaggle.json`
- Impostare le variabili d'ambiente `KAGGLE_USERNAME` e `KAGGLE_KEY`

Lo script scarica [ambityga/imagenet100](https://www.kaggle.com/datasets/ambityga/imagenet100), estrae gli archivi e riorganizza il layout Kaggle nativo nella struttura unificata richiesta sia dagli script di training che dai notebook:

```
data/raw/imagenet100/
├── train/
│   ├── n01440764/   # una cartella per classe
│   └── ...
└── val/
    └── ...
```

**Statistiche del dataset**: ~130.000 immagini di training · 5.000 immagini di validation · 100 classi · ~1.300 immagini/classe (train) · 50 immagini/classe (val).

---

### 3. Training

Tutte le configurazioni si trovano in `experiments/configs/`. I log (TensorBoard + W&B) vengono salvati in `experiments/logs/`, i checkpoint in `experiments/checkpoints/`.

#### Step 1 — ViT Supervisionato Baseline (200 epoche)

```bash
python train_supervised.py --config experiments/configs/supervised_vit_cluster.yaml
```

Iperparametri principali: ViT-Base/16 · AdamW lr=1e-3 · weight_decay=0.05 · cosine decay · warmup 20 ep · label smoothing 0.1.

#### Step 2 — Pre-Training MAE (500 epoche)

```bash
python train_mae.py --config experiments/configs/mae_pretrain_cluster.yaml
```

Iperparametri principali: mask_ratio=0.75 · norm_pix_loss=True · AdamW lr=7.5e-5 · beta2=0.95 · weight_decay=0.05 · cosine decay · warmup 30 ep.

Per riprendere da un checkpoint:
```bash
python train_mae.py --config experiments/configs/mae_pretrain_cluster.yaml \
    --resume experiments/checkpoints/mae_pretrain/checkpoint_latest.pth
```

#### Step 3 — Linear Probe (200 epoche)

```bash
python train_linear_probe.py --config experiments/configs/linear_probe_cluster.yaml \
    --mae-checkpoint experiments/checkpoints/mae_pretrain/checkpoint_best.pth
```

Iperparametri principali: encoder congelato · AdamW lr=1e-2 · weight_decay=0.0 · cosine decay fino a 1e-4 · warmup 10 ep.

#### Esperimenti Data Efficiency (10% e 20% dei dati etichettati)

```bash
# ViT Supervisionato — 10% dati di training
python train_supervised.py --config experiments/configs/supervised_vit_small_cluster.yaml

# Linear Probe — 10% dati di training (stesso sottoinsieme, stratificato per classe)
python train_linear_probe.py --config experiments/configs/linear_probe_small_cluster.yaml \
    --mae-checkpoint experiments/checkpoints/mae_pretrain/checkpoint_best.pth

# ViT Supervisionato — 20% dati di training
python train_supervised.py --config "experiments/configs/supervised_vit_small_dataset_20%_cluster.yaml"

# Linear Probe — 20% dati di training
python train_linear_probe.py --config "experiments/configs/linear_probe_small_dataset_20%_cluster.yaml" \
    --mae-checkpoint experiments/checkpoints/mae_pretrain/checkpoint_best.pth
```

I sottoinsieme sono stratificati (stesso numero di immagini per classe) con seed fisso, garantendo che supervised e linear probe usino le stesse immagini identiche.

---

### 4. Valutazione

Eseguire ogni modello separatamente (richiede il checkpoint corrispondente):

```bash
python evaluate.py \
    --config experiments/configs/supervised_vit_cluster.yaml \
    --checkpoint experiments/checkpoints/supervised_vit_baseline_200/checkpoint_best.pth \
    --label "ViT Supervisionato - Dataset Completo (200 ep)"

python evaluate.py \
    --config experiments/configs/linear_probe_cluster.yaml \
    --checkpoint experiments/checkpoints/linear_probe_200/checkpoint_best.pth \
    --label "MAE + Linear Probe - Dataset Completo (200 ep)"

python evaluate.py \
    --config experiments/configs/supervised_vit_small_cluster.yaml \
    --checkpoint "experiments/checkpoints/supervised_vit_small_dataset_10%/checkpoint_best.pth" \
    --label "ViT Supervisionato - 10% Dataset (100 ep)"

python evaluate.py \
    --config experiments/configs/linear_probe_small_cluster.yaml \
    --checkpoint "experiments/checkpoints/linear_probe_small_dataset_10%/checkpoint_best.pth" \
    --label "MAE + Linear Probe - 10% Dataset (100 ep)"

python evaluate.py \
    --config "experiments/configs/supervised_vit_small_dataset_20%_cluster.yaml" \
    --checkpoint "experiments/checkpoints/supervised_vit_small_dataset_20%/checkpoint_best.pth" \
    --label "ViT Supervisionato - 20% Dataset (100 ep)"

python evaluate.py \
    --config "experiments/configs/linear_probe_small_dataset_20%_cluster.yaml" \
    --checkpoint "experiments/checkpoints/linear_probe_small_dataset_20%/checkpoint_best.pth" \
    --label "MAE + Linear Probe - 20% Dataset (100 ep)"
```

Per valutare tutti gli esperimenti contemporaneamente sul cluster GPU:

```bash
sbatch scripts/slurm/evaluate.sh
```

---

### 5. Visualizzazione Ricostruzioni MAE

Genera una griglia affiancata (Originale | Input Mascherato | Ricostruzione) dalle immagini di validation:

```bash
python visualize_mae.py
```

Output salvato in `figures/reconstruction_grid.png`. Flag opzionali:

```bash
python visualize_mae.py --num-images 8 --seed 0 --split val
```

---

### 6. Monitoraggio

**TensorBoard** (in locale):
```bash
tensorboard --logdir experiments/logs
```

---

## 📊 Riepilogo Risultati

| Modello | Dati di Training | Top-1 Acc. | Top-5 Acc. |
|:---|:---:|:---:|:---:|
| ViT Supervisionato (baseline) | 100% | 72.94% | 88.62% |
| MAE + Linear Probe | 100% | 71.02% | 91.04% |
| ViT Supervisionato | 20% | 11.66% | 31.88% |
| MAE + Linear Probe | 20% | 66.62% | 88.48% |
| ViT Supervisionato | 10% | 8.30% | 25.92% |
| MAE + Linear Probe | 10% | 63.12% | 86.90% |

Tutti i modelli valutati sull'intero validation set (5.000 immagini, 100 classi).

---

## 🗂 Struttura del Progetto

```
├── src/
│   ├── models/
│   │   ├── mae.py                  # MAE completo (patchify, masking, loss)
│   │   ├── mae_decoder.py          # Decoder ViT leggero
│   │   ├── vit_encoder.py          # Encoder ViT-Base
│   │   ├── vit_classifier.py       # ViT + testa di classificazione
│   │   └── patch_embed.py          # Layer di patch embedding
│   ├── datasets/
│   │   └── imagenet100.py          # DataLoader con supporto sottoinsieme stratificato
│   ├── training/
│   │   ├── trainer_mae.py          # Loop di training MAE
│   │   ├── trainer_supervised.py   # Loop di training supervisionato
│   │   ├── trainer_linear_probe.py # Loop di training linear probe
│   │   ├── optimizer.py            # AdamW + scheduler cosine
│   │   └── losses.py               # Loss MAE e cross-entropy
│   ├── evaluation/
│   │   ├── evaluator.py            # Evaluator Top-1/Top-5
│   │   └── metrics.py              # AverageMeter
│   └── utils/
│       ├── checkpoint.py           # Salvataggio/caricamento checkpoint
│       ├── config.py               # Loader configurazioni YAML
│       ├── logger.py               # Logger unificato W&B + TensorBoard
│       └── misc.py                 # Seed, device, conteggio parametri
├── experiments/
│   ├── configs/                    # Config YAML per tutti gli esperimenti
│   ├── checkpoints/                # Pesi dei modelli (esclusi da git)
│   └── logs/                       # Log TensorBoard + W&B (esclusi da git)
├── scripts/
│   ├── slurm/                      # Script SLURM per il cluster GPU
│   ├── sync_to_cluster.ps1         # Carica codice sul cluster (Windows)
│   └── sync_from_cluster.ps1       # Scarica risultati dal cluster (Windows)
├── notebooks/                      # Notebook EDA e visualizzazione patch
├── figures/                        # Figure di output (ricostruzioni MAE, ecc.)
├── docs/                           # Report e slide per la presentazione
├── train_mae.py                    # Entrypoint pre-training MAE
├── train_supervised.py             # Entrypoint baseline supervisionato
├── train_linear_probe.py           # Entrypoint linear probe
├── evaluate.py                     # Entrypoint valutazione standalone
└── visualize_mae.py                # Visualizzazione ricostruzioni MAE
```

---

## 🖥 Esecuzione su Cluster HPC

Il codice Python è completamente portabile — gira su qualsiasi macchina con PyTorch e una GPU CUDA senza modifiche.

Gli script SLURM in `scripts/slurm/` sono stati configurati per il cluster GPU dell'Università di Catania. Per eseguire su un altro cluster HPC è necessario adattare i parametri `--partition`, `--account`, `--qos` e il comando di esecuzione del container alle specifiche della propria infrastruttura.

---
