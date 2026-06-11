# Clarity NLP Project – Political Response Classification with Longformer

Questo progetto implementa una pipeline completa di Natural Language Processing per la classificazione della chiarezza delle risposte politiche utilizzando il modello **Longformer-base-4096**.

L'obiettivo è classificare una coppia **domanda-risposta** in una delle seguenti categorie:

* **Clear Reply**
* **Clear Non-Reply**
* **Ambivalent**

Il progetto utilizza il dataset **QEvasion** e sfrutta la capacità del Longformer di gestire sequenze lunghe mantenendo il contesto completo dell'interazione.

---

## Struttura del progetto

```bash
clarity_nlp_project/
│
├── configs/
│   ├── data/
│   │   └── clarity.yaml
│   ├── model/
│   ├── training/
│   └── default.yaml
│
├── data/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── checkpoints/
│   └── final/
│
├── reports/
│
├── src/
│   └── clarity_nlp_project/
│       ├── data/
│       │   ├── loader.py
│       │   ├── preprocess.py
│       │   ├── splits.py
│       │   └── tokenizer_utils.py
│       │
│       ├── models/
│       │   └── hf_classifier.py
│       │
│       ├── training/
│       │   └── trainer.py
│       │
│       ├── __init__.py
│       └── main.py
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Dataset

Il progetto utilizza il dataset:

```bash
ailsntua/QEvasion
```

contenente interviste presidenziali annotate secondo il livello di chiarezza delle risposte.

Le classi considerate sono:

| Classe          | Descrizione                                |
| --------------- | ------------------------------------------ |
| Clear Reply     | Risposta diretta e pertinente              |
| Clear Non-Reply | Risposta evasiva                           |
| Ambivalent      | Risposta parzialmente pertinente o ambigua |

---

## Preprocessing

Ogni esempio viene convertito nel formato:

```text
<QUESTION>
Question text
</QUESTION>

<ANSWER>
Answer text
</ANSWER>
```

Per migliorare la separazione semantica tra domanda e risposta vengono introdotti token speciali:

```text
<QUESTION>
</QUESTION>
<ANSWER>
</ANSWER>
```

Successivamente il testo viene tokenizzato utilizzando il tokenizer del Longformer.

---

## Modello

Il classificatore è basato su:

```bash
allenai/longformer-base-4096
```

Longformer è una variante dei Transformer progettata per gestire documenti lunghi tramite un meccanismo di attenzione locale e globale, riducendo il costo computazionale rispetto alla self-attention standard.

---

## Global Attention

Per sfruttare le caratteristiche del Longformer viene utilizzata una strategia di attenzione globale sui token più informativi:

* token iniziale della sequenza
* token `<QUESTION>`
* token `<ANSWER>`
* primi token della risposta

Questa configurazione consente al modello di concentrarsi sulle parti più rilevanti dell'interazione domanda-risposta.

---

## Bilanciamento delle classi

Poiché il dataset presenta una distribuzione sbilanciata delle classi, durante il training vengono utilizzate tecniche di riequilibrio:

* Weighted Random Sampling
* Class Weights nella funzione di loss

L'obiettivo è ridurre il bias verso la classe maggioritaria.

---

## Pipeline generale

```bash
dataset
   ↓
preprocessing
   ↓
tokenization
   ↓
global attention masks
   ↓
train / validation split
   ↓
training
   ↓
evaluation
```

---

## Addestramento

L'addestramento viene gestito tramite il modulo:

```bash
src/clarity_nlp_project/training/trainer.py
```

La configurazione degli esperimenti è centralizzata nei file presenti nella cartella:

```bash
configs/
```

consentendo di modificare facilmente:

* learning rate
* batch size
* numero di epoche
* gradient accumulation
* warmup
* weight decay

---

## Avvio del progetto

Installazione delle dipendenze:

```bash
pip install -r requirements.txt
```

Esecuzione del training:

```bash
python -m src.clarity_nlp_project.main
```

---

## Output

Durante l'esecuzione vengono generati:

```bash
models/checkpoints/
```

contenente i checkpoint intermedi del modello,

e

```bash
models/final/
```

contenente il modello finale addestrato.

Le metriche e i report degli esperimenti vengono salvati nella cartella:

```bash
reports/
```

---

## Obiettivo del progetto

Questo lavoro esplora l'utilizzo di modelli Transformer per il riconoscimento automatico di risposte evasive in ambito politico.

L'uso del Longformer permette di mantenere il contesto completo delle interazioni domanda-risposta e rappresenta una soluzione efficace per compiti di classificazione testuale basati su sequenze lunghe.

---

## Autore

**Francesco Lo Vetri**

Master's Degree in Artificial Intelligence and Cybersecurity

University of Enna "Kore"
