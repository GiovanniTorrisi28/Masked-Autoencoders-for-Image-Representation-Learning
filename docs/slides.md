---
marp: true
theme: dl
paginate: true
lang: it
---

<!-- _class: lead -->
<!-- _paginate: false -->

<div class="course">Deep Learning · Advanced Models &amp; Methods</div>

# Imparare a vedere<br>senza etichette

<div class="track">Track 11 — Masked Autoencoders for Image Representation Learning</div>

<div class="meta">
<strong>Giovanni Torrisi</strong> · Gruppo G21<br>
<span class="repo">github.com/GiovanniTorrisi28/Masked-Autoencoders-for-Image-Representation-Learning</span>
</div>

---

# L'etichettatura è il collo di bottiglia della visione

<div class="cols">
<div class="narrow">

Annotare milioni di immagini è **costoso e lento**.

**Domanda:** un Vision Transformer può imparare rappresentazioni utili *prima* di vedere una sola etichetta?

**Idea (self-supervised):** nascondi gran parte dell'immagine e chiedi al modello di **ricostruirla**.

<div class="card" style="margin-bottom:0">
<strong>ImageNet100</strong> · 100 classi · ~130K train · 5K val
</div>

</div>
<div class="wide">

![w:720](../figures/02_masked_vs_visible.png)
<figcaption>Mascheriamo il 75% delle patch: resta visibile solo 1 patch su 4.</figcaption>

</div>
</div>

---

# Il MAE ricostruisce il 75% nascosto da un quarto di immagine

<div class="cols">
<div>
<div class="card">
<h3>1 · Masking</h3>
Immagine → 196 patch 16×16.<br>
Si nasconde il <strong>75%</strong> (147 patch).
</div>
</div>
<div>
<div class="card solid">
<h3>2 · Encoder ViT-Base</h3>
Vede <strong>solo le 49 patch visibili</strong>.<br>
~3× più leggero ed efficiente.
</div>
</div>
<div>
<div class="card">
<h3>3 · Decoder leggero</h3>
Reinserisce i <em>mask token</em> e<br>
<strong>ricostruisce tutte</strong> le patch.
</div>
</div>
</div>

<div class="card" style="margin-top:18px">
<strong>Loss:</strong> errore quadratico (MSE) sui pixel, calcolato <strong>solo sulle patch mascherate</strong> e normalizzato per patch → l'encoder è forzato a imparare la <em>struttura</em>, non i colori medi.
</div>

---

# Una pipeline, due confronti equi

<div class="cols">
<div>

<div class="card solid">
<strong>Pre-training MAE</strong><br>
500 epoche · 130K immagini · <strong>nessuna etichetta</strong>
</div>

<p class="center small">↓ encoder pre-addestrato</p>

<div class="card">
<strong>A) Linear Probe</strong><br>
Encoder <strong>congelato</strong> + un solo layer lineare (768→100).
</div>

<div class="card">
<strong>B) Baseline ViT supervisionato</strong><br>
Stesso ViT addestrato <strong>da zero</strong> con le etichette.
</div>

</div>
<div class="narrow">

<div class="card">
<h3>Studio data-efficiency</h3>
Ripetiamo A e B con solo il<br>
<strong>10%</strong> e <strong>20%</strong> delle etichette.

<p class="small">Subset stratificato (seed 42): A e B usano <strong>le stesse immagini</strong> → confronto onesto.</p>
</div>

</div>
</div>

---

# Scelte chiave del progetto

<div class="cols">
<div>

<div class="card">
<h3>Architettura</h3>

- **ViT-Base/16** · ~86M parametri
- Patch 16×16 → 196 token
- Pos-embed sinusoidali 2D (fissi)
- Decoder: 512 dim · 8 layer
</div>

</div>
<div>

<div class="card">
<h3>Training</h3>

- **Mask ratio 0.75**
- Loss su **pixel normalizzati**
- AdamW + cosine schedule
- No color-jitter nel pre-training
</div>

</div>
</div>

<div class="card solid center" style="margin-top:14px">
Tre <em>trick</em> chiave del paper MAE riprodotti: <strong>masking alto</strong> · <strong>encoder asimmetrico</strong> · <strong>target normalizzati per patch</strong>.
</div>

---

# Quasi alla pari col supervisionato — senza etichette

<div class="cols">
<div class="narrow">

| Modello | Top-1 | Top-5 |
|:--|:--:|:--:|
| ViT Superv. | **72.9** | 88.6 |
| MAE + LP | 71.0 | **91.0** |

Solo **−1.9 pp** in Top-1.

In **Top-5 il MAE supera** il supervisionato (+2.4 pp): la classe giusta è quasi sempre tra le prime 5.

</div>
<div class="wide">

![w:600](../figures/slides/top1_top5_full.png)

</div>
</div>

<p class="small">Dataset completo (100%) · l'encoder MAE struttura lo spazio delle feature <strong>senza mai vedere un'etichetta</strong> nel pre-training.</p>

---

# Con poche etichette, il pre-training cambia tutto

<div class="cols">
<div class="wide">

![w:680](../figures/slides/data_efficiency_curve.png)

</div>
<div class="narrow">

Al **10% dei dati**:

<div class="kpi" style="flex-direction:column">
<div class="item"><div class="num">63.1%</div><div class="lbl">MAE + Linear Probe</div></div>
<div class="item"><div class="num" style="color:#9AA0A6">8.3%</div><div class="lbl">ViT Supervisionato</div></div>
</div>

Il ViT da zero **collassa** (best epoch = 2): 86M parametri non si addestrano con ~130 img/classe.

</div>
</div>

---

# Cosa impara — e dove sbaglia

<div class="cols">
<div class="narrow">

![w:300](../figures/slides/reconstruction_compact.png)
<figcaption>Originale · Mascherato 75% · Ricostruzione</figcaption>

</div>
<div class="wide">

<p class="mt0"><strong>MAE vince sulle texture complesse</strong> — invertebrati e animali acquatici, dove ricostruire i pixel forza pattern fini:</p>

![w:560](../figures/slides/per_class_highlights.png)

<p class="small"><strong>Failure case</strong> comuni a tutti i modelli: classi <em>fine-grained</em> quasi identiche (17 specie di serpenti, anfibi).</p>

</div>
</div>

---

# Il pre-training MAE rende le rappresentazioni quasi indipendenti dalle etichette

<div class="cols">
<div>

<div class="card">

**−1.9 pp** con dati pieni…

**+54.8 pp** quando le etichette scarseggiano.

</div>

</div>
<div>

<div class="card">

**Takeaway**
- Self-supervised = robustezza in regime di poche etichette
- ViT senza bias induttivi ha bisogno di dati **o** pre-training
- Top-5 superiore → feature semanticamente strutturate

</div>

</div>
</div>

---

# Iperparametri

| | MAE pre-train | Supervisionato | Linear Probe |
|:--|:--:|:--:|:--:|
| Epoche | 500 | 200 | 200 |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 7.5e-5 | 1e-3 | 1e-2 |
| Weight decay | 0.05 | 0.05 | 0.0 |
| Schedule | cosine | cosine | cosine |
| Batch size | 128 | 128 | 128 |
| Note | β=(0.9,0.95) · clip 1.0 | label smooth 0.1 | encoder congelato |

---

# Tutti i risultati

| Modello | Dati | Top-1 | Top-5 | Val loss |
|:--|:--:|:--:|:--:|:--:|
| ViT Supervisionato | 100% | 72.94 | 88.62 | 1.313 |
| MAE + Linear Probe | 100% | 71.02 | 91.04 | 1.091 |
| ViT Supervisionato | 20% | 11.66 | 31.88 | 3.836 |
| MAE + Linear Probe | 20% | 66.62 | 88.48 | 1.280 |
| ViT Supervisionato | 10% | 8.30 | 25.92 | 3.999 |
| MAE + Linear Probe | 10% | 63.12 | 86.90 | 1.425 |

<p class="small">Architettura: ViT-Base/16 (~86M param) · decoder 512 dim / 8 layer / 16 head · pos-embed sinusoidali 2D · valutazione su 5.000 immagini di validation.</p>

---

# Limiti e lavoro futuro

<div class="cols">
<div>

<div class="card">
<h3>Limiti riconosciuti</h3>

- Pre-training **500 ep** (vs 1600 del paper)
- **ImageNet100**, non ImageNet-1K
- Solo frazioni 10% / 20% testate
</div>

</div>
<div>

<div class="card">
<h3>Prossimi passi</h3>

- **Ablation** del mask ratio (25–90%)
- Confronto con metodi **contrastivi** (SimCLR, MoCo)
- **Full fine-tuning** dell'encoder MAE
</div>

</div>
</div>
