# Masked Autoencoders per l'Apprendimento di Rappresentazioni Visive
- **ID Gruppo**: G21
- **ID Progetto**: 11
- **Studente**: Giovanni Torrisi

---

## 1. Introduzione e Obiettivo

L'apprendimento auto-supervisionato mira ad apprendere rappresentazioni visive ricche a partire da dati non etichettati, affrontando il principale collo di bottiglia del deep learning moderno: il costo dell'annotazione umana su larga scala. I Masked Autoencoder (MAE), introdotti da He et al. (2022), rappresentano un passo avanti significativo in questo ambito: un Vision Transformer viene pre-addestrato mascherando casualmente il 75% delle patch di un'immagine e ricostruendo i valori dei pixel mancanti a partire dal contesto visivo rimanente. Il modello deve quindi comprendere la struttura semantica delle immagini per predire contenuti plausibili per le regioni nascoste — un compito che richiede un ragionamento visivo profondo.

L'ipotesi centrale di questo progetto è che un encoder pre-addestrato con MAE sviluppi rappresentazioni più ricche e generalizzabili rispetto a un encoder inizializzato casualmente e addestrato in modalità puramente supervisionata — in particolare in scenari di scarsità di dati etichettati. Questo si traduce in un confronto sperimentale concreto: un ViT baseline supervisionato contro un encoder pre-addestrato con MAE valutato tramite linear probing, sia sull'intero dataset sia su sottoinsiemi ridotti (10% e 20% dei dati di training etichettati).

---

## 2. Contributo e Valore Aggiunto

L'intera pipeline è stata implementata da zero in PyTorch, senza ricorrere a implementazioni MAE preesistenti:

- **Encoder ViT-Base/16**: implementazione custom con positional embedding sinusoidali 2D, multi-head self-attention e blocchi MLP
- **Architettura MAE asimmetrica**: l'encoder processa solo il 25% delle patch visibili, mentre un decoder leggero ricostruisce tutte le 196 patch utilizzando mask token appresi
- **Mascheratura casuale delle patch**: strategia efficiente basata su shuffle che produce insiemi di indici visibili/mascherati passati separatamente a encoder e decoder
- **Loss su pixel normalizzati**: normalizzazione media/varianza per patch dei target di ricostruzione, migliorando la stabilità del training
- **Subsampling stratificato**: riduzione riproducibile del dataset basata su frazione (stesso sottoinsieme per supervised e linear probe tramite seed fisso), che abilita lo studio sulla data efficiency
- **Visualizzazione ricostruzioni MAE**: script che genera griglie affiancate di immagine originale, input mascherato e output del decoder

---

## 3. Dati Utilizzati

**Dataset**: ImageNet100 ([Kaggle: ambityga/imagenet100](https://www.kaggle.com/datasets/ambityga/imagenet100)), un sottoinsieme bilanciato di 100 classi di ImageNet-1K.

| Split | Immagini | Classi | Immagini/classe |
|:---|:---:|:---:|:---:|
| Training | ~130.000 | 100 | ~1.300 |
| Validation | 5.000 | 100 | 50 |

**Pre-elaborazione per training supervisionato e linear probe:**
- *Training*: `RandomResizedCrop(224)` · `RandomHorizontalFlip()` · `ColorJitter(0.4, 0.4, 0.4, 0.1)` · normalizzazione ImageNet
- *Validation*: `Resize(256)` · `CenterCrop(224)` · normalizzazione ImageNet

**Pre-elaborazione per il pre-training MAE:**
- `RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=BICUBIC)` · `RandomHorizontalFlip()` · normalizzazione ImageNet
- Il color jitter è escluso intenzionalmente: il decoder deve ricostruire valori esatti dei pixel, quindi target di colore coerenti sono necessari

**Esperimenti di data efficiency**: sottoinsiemi casuali stratificati con `seed=42` fisso, selezionando esattamente `floor(dimensione_classe × frazione)` immagini per classe. Gli esperimenti supervisionati e di linear probe alla stessa frazione utilizzano lo stesso identico sottoinsieme, garantendo un confronto equo.

---

## 4. Metodologia e Architettura

### 4.1 Encoder ViT-Base/16

Sia il baseline supervisionato che l'encoder MAE condividono lo stesso backbone:

| Parametro | Valore |
|:---|:---:|
| Dimensione immagine | 224 × 224 |
| Dimensione patch | 16 × 16 |
| Numero di patch | 196 |
| Dimensione embedding | 768 |
| Profondità Transformer | 12 |
| Teste di attenzione | 12 |
| Rapporto MLP | 4.0 |
| Parametri addestrabili | ~86M |

I positional embedding sono sinusoidali 2D fissi (non appresi), seguendo il paper MAE. I pesi sono inizializzati con distribuzione normale troncata (std=0.02).

### 4.2 Pre-Training MAE

**Strategia di mascheratura**: per ogni immagine nel batch, viene generata una permutazione casuale degli indici delle patch. Il primo 25% (49 patch) è mantenuto visibile; il restante 75% (147 patch) viene mascherato. L'encoder processa solo le 49 patch visibili (più il token CLS), riducendo il suo costo computazionale di circa 3×.

**Decoder**: un ViT leggero con embed_dim=512, depth=8, num_heads=16. Riceve i token visibili codificati proiettati alla dimensione del decoder, più i mask token appresi nelle posizioni mascherate — tutti con positional embedding aggiunti. Produce predizioni per tutte le 196 patch.

**Funzione di loss**: MSE calcolata esclusivamente sulle 147 patch mascherate:

$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| \hat{p}_i - p_i \right\|^2$$

dove i target $p_i$ sono normalizzati per patch (normalizzazione media/varianza), stabilizzando il training e focalizzando il modello sul contenuto strutturale piuttosto che sulle basse frequenze di colore.

**Configurazione di training**:

| Iperparametro | Valore |
|:---|:---:|
| Epoche | 500 |
| Ottimizzatore | AdamW |
| Learning rate | 7.5 × 10⁻⁵ |
| Weight decay | 0.05 |
| β₁, β₂ | 0.9, 0.95 |
| Schedule LR | Cosine decay |
| Epoche di warmup | 30 |
| LR minimo | 10⁻⁶ |
| Gradient clipping | 1.0 |
| Batch size | 128 |

### 4.3 ViT Supervisionato Baseline

Lo stesso encoder ViT-Base/16 viene addestrato end-to-end da inizializzazione casuale con una testa di classificazione lineare su 100 classi.

| Iperparametro | Valore |
|:---|:---:|
| Epoche | 200 |
| Ottimizzatore | AdamW |
| Learning rate | 10⁻³ |
| Weight decay | 0.05 |
| Schedule LR | Cosine decay (warmup 20 ep) |
| Label smoothing | 0.1 |
| Dropout | 0.1 |
| Batch size | 128 |

### 4.4 Linear Probe

L'encoder MAE viene congelato (tutti i parametri fissi, eseguito in modalità eval). Viene addestrato solo un singolo layer lineare che mappa le feature del token CLS (768 dimensioni) alle 100 classi.

| Iperparametro | Valore |
|:---|:---:|
| Epoche | 200 |
| Ottimizzatore | AdamW |
| Learning rate | 10⁻² |
| Weight decay | 0.0 |
| Schedule LR | Cosine decay fino a 10⁻⁴ (warmup 10 ep) |
| Batch size | 128 |

Il weight decay è impostato a 0 perché il linear probe è un problema di ottimizzazione convessa — la regolarizzazione tramite il cosine decay dello schedule è sufficiente.

### 4.5 Esperimenti di Data Efficiency

Per studiare il comportamento di ciascun approccio in condizioni di scarsità di etichette, sia il ViT supervisionato che il linear probe sono stati addestrati su sottoinsiemi stratificati del 10% (~13.000 immagini) e del 20% (~26.000 immagini) del training set. L'encoder MAE utilizzato per i linear probe rimane quello pre-addestrato sull'intero dataset non etichettato — viene ridotto solo il training set etichettato per la testa lineare. Entrambi gli esperimenti utilizzano 100 epoche.

---

## 5. Risultati e Discussione

### 5.1 Dataset Completo

**Tabella 1**: Risultati sull'intero validation set (5.000 immagini, 100 classi).

| Modello | Epoche | Top-1 Acc. | Top-5 Acc. | Val Loss |
|:---|:---:|:---:|:---:|:---:|
| ViT Supervisionato (baseline) | 200 | 72.94% | 88.62% | 1.3133 |
| MAE + Linear Probe | 200 | 71.02% | 91.04% | 1.0910 |

Il MAE + linear probe raggiunge il **71.02% Top-1**, solamente 1.92 punti percentuali al di sotto del modello completamente supervisionato che addestra tutti gli 86M parametri end-to-end con le etichette. È notevole che la **Top-5 accuracy del MAE + linear probe (91.04%) superi la baseline supervisionata (88.62%)**, suggerendo che l'encoder MAE organizza lo spazio delle feature in modo che la classe corretta sia quasi sempre nei 5 candidati principali — un segnale di rappresentazioni semanticamente ben strutturate. Questo risultato è notevole considerando che durante il pre-training l'encoder non ha mai osservato una singola etichetta.

### 5.2 Studio sulla Data Efficiency

**Tabella 2**: Impatto della riduzione dei dati di training etichettati.

| Modello | Dati di Training | Top-1 Acc. | Top-5 Acc. | Epoca Migliore |
|:---|:---:|:---:|:---:|:---:|
| ViT Supervisionato | 100% | 72.94% | 88.62% | 192 |
| MAE + Linear Probe | 100% | 71.02% | 91.04% | 191 |
| ViT Supervisionato | 20% | 11.66% | 31.88% | **2** |
| MAE + Linear Probe | 20% | 66.62% | 88.48% | 86 |
| ViT Supervisionato | 10% | 8.30% | 25.92% | **2** |
| MAE + Linear Probe | 10% | 63.12% | 86.90% | 84 |

I risultati rivelano un'asimmetria netta. Con il 10% e il 20% dei dati etichettati, il ViT supervisionato **non riesce ad apprendere nulla di utile**, con il miglior checkpoint di validation all'epoca 2 in entrambi i casi — il che significa che il modello inizia immediatamente ad andare in overfitting e degrada per le restanti 98 epoche. L'accuratezza finale (8.30% e 11.66%) è appena superiore alla casualità (1% per 100 classi). Non si tratta di overfitting classico (dove la train accuracy è alta ma la val è bassa), ma di un **fallimento totale nell'estrazione di feature significative**: anche la training loss rimane alta (~4.3), confermando che 86M parametri non possono essere appresi da sole 130–260 immagini per classe senza bias induttivi. Il ViT, a differenza delle CNN, manca di connettività locale ed equivarianza alla traslazione — deve apprenderle dai dati, richiedendo un numero di esempi molto più elevato.

Al contrario, MAE + linear probe **mantiene il 63.12% con il 10% e il 66.62% con il 20%** dei dati etichettati — un calo di soli 7.9 e 4.4 punti percentuali rispetto al dataset completo. L'encoder MAE è stato pre-addestrato su tutte le 130.000 immagini non etichettate, quindi le sue rappresentazioni codificano già feature strutturali ricche. La testa lineare deve solo trovare il confine di decisione in questo spazio delle feature già ben organizzato, operazione che può essere eseguita in modo affidabile anche con 130 esempi etichettati per classe.

Il divario di data efficiency con il 10% dei dati etichettati è di **54.82 punti percentuali** (63.12% vs 8.30%), dimostrando il vantaggio fondamentale del pre-training auto-supervisionato in regimi a basse etichette.

### 5.3 Qualità della Ricostruzione MAE

Il decoder produce ricostruzioni visivamente coerenti delle patch mascherate. Il modello inferisce correttamente texture, colori e forme approssimative delle regioni nascoste dal 25% di contesto visibile. Alcuni dettagli di fine grained (bordi netti, oggetti piccoli) rimangono imperfetti, come atteso dato il rapporto di mascheratura del 75% — il decoder deve ricostruire la maggior parte del contenuto dell'immagine. La qualità della ricostruzione è coerente con una training loss finale di ~0.46 (MSE normalizzato).

---

## 6. Conclusioni e Limitazioni

Questo progetto dimostra che il pre-training MAE produce rappresentazioni competitive con il training completamente supervisionato su ImageNet100 (72.94% vs 71.02% Top-1), risultando al contempo molto più efficiente in termini di dati. Nello scenario più estremo testato (10% dati etichettati), MAE + linear probe supera il ViT supervisionato di oltre 54 punti.

**Limitazioni:**
- Il training è stato eseguito per 500 epoche, mentre il paper originale MAE ne utilizza 1.600 su ImageNet-1K completo. Un pre-training più lungo migliorerebbe verosimilmente l'accuratezza del linear probe
- Il dataset utilizzato (ImageNet100) è significativamente più piccolo di ImageNet-1K completo (~130K vs ~1.28M immagini), il che può limitare la qualità delle rappresentazioni apprese
- Sono state testate solo due frazioni di dati (10% e 20%). Un'analisi più fine (ad esempio 1%, 5%, 30%, 50%) fornirebbe un quadro più completo del punto di crossover dove il training supervisionato diventa competitivo

**Possibili esperimenti futuri:**
- Ablazione sul rapporto di mascheratura (25%, 50%, 75%, 90%) per studiare la relazione tra difficoltà della mascheratura e accuratezza downstream
- Confronto con metodi contrastivi auto-supervisionati (SimCLR, MoCo) nelle stesse condizioni di data efficiency
- Fine-tuning completo dell'encoder MAE (invece del linear probe) per misurare il limite superiore delle rappresentazioni pre-addestrate

---

## 7. Informazioni Aggiuntive

### 7.1 Suddivisione dei Contributi

- **Giovanni Torrisi**: intero progetto — pipeline di pre-elaborazione del dataset, implementazione encoder/decoder ViT, loop di training MAE, trainer supervisionato e linear probe, framework di valutazione, esperimenti di data efficiency, visualizzazione ricostruzioni, configurazione e training sul cluster GPU

### 7.2 Utilizzo dell'Intelligenza Artificiale

**Claude Code** (Anthropic) è stato utilizzato durante l'intero progetto come assistente di programmazione per: scrittura e debug delle implementazioni PyTorch, configurazione dell'infrastruttura di training (script SLURM, logging, gestione dei checkpoint), progettazione del sistema di configurazione degli esperimenti e redazione della documentazione. Tutte le scelte architetturali, le decisioni di design sperimentale e le interpretazioni dei risultati sono state effettuate dallo studente. L'implementazione è stata revisionata, compresa e validata ad ogni passaggio.
