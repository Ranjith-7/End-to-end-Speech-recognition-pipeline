# 🎙️ End-to-End Hybrid HMM-RNN Speech-to-Text Pipeline

> A complete, modular Speech-to-Text system combining Hidden Markov Models (HMM) with Recurrent Neural Networks (RNN) for accurate audio transcription.

---

## 📌 Project Overview

This project implements a **Hybrid HMM-RNN Speech-to-Text Pipeline** that combines the sequential modeling power of Hidden Markov Models with the deep learning capabilities of Recurrent Neural Networks. It is trained and evaluated on the **LJSpeech Dataset** and supports optional **language model decoding** via KenLM.

---

## ✨ Features

- 🔊 **Audio Preprocessing** — Voice Activity Detection (VAD), noise reduction, normalization, and MFCC feature extraction
- 🧠 **Acoustic Model** — LSTM-based RNN trained on MFCC features
- 🔁 **Seq2Seq Decoder** — Sequence-to-sequence decoder for transcription output
- 📊 **Evaluation** — Word Error Rate (WER) computation for benchmarking
- 🔍 **Inference** — Predict transcription from any `.wav` audio file
- 📖 **Language Model (Optional)** — KenLM-based beam search decoding for improved accuracy

---

## 🗂️ Project Structure

```
End-to-End-Speech-Recognition/
│
├── main.py                    # End-to-end training and inference pipeline
├── README.md                  # Project documentation
│
└── dataset/
    └── LJSpeech-1.1/
        ├── wavs/              # Audio files (.wav)
        └── metadata.csv       # Transcription labels
```

---

## ⚙️ Tech Stack

| Component | Library / Tool |
|---|---|
| Audio Processing | `Librosa` |
| Deep Learning | `PyTorch` |
| Language Model | `KenLM` (optional) |
| Dataset | `LJSpeech-1.1` |
| Numerical Computing | `NumPy` |
| Progress Tracking | `tqdm` |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Ranjith-7/End-to-end-Speech-recognition-.git
cd End-to-end-Speech-recognition-
```

### 2. Install Dependencies

```bash
pip install torch numpy librosa tqdm
```

> **Optional (for KenLM language model):**
```bash
pip install kenlm
```

### 3. Prepare the Dataset

- Download the [LJSpeech-1.1 Dataset](https://keithito.com/LJ-Speech-Dataset/)
- Extract it to the following path:

```
/content/dataset/LJSpeech-1.1/
```

The folder should contain:
- `wavs/` — Audio files
- `metadata.csv` — Corresponding transcriptions

---

## 🏋️ Training

Run the full end-to-end training pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess audio files
2. Extract MFCC features
3. Train the LSTM acoustic model
4. Train the Seq2Seq decoder
5. Evaluate using Word Error Rate (WER)

---

## 🔎 Inference

Transcribe a custom audio file using the trained model:

```python
from main import inference

hypothesis = inference(
    "path/to/audio.wav",
    input_dim=13,
    hidden_dim=128,
    output_dim=256,
    use_kenlm=False      # Set True to use KenLM language model
)

print("Transcription:", hypothesis)
```

---

## 📐 Model Architecture

```
Audio (.wav)
    │
    ▼
Preprocessing (VAD + Noise Reduction + Normalization)
    │
    ▼
MFCC Feature Extraction (Librosa)
    │
    ▼
LSTM Acoustic Model (PyTorch)
    │
    ▼
Seq2Seq Decoder
    │
    ▼
[Optional] KenLM Beam Search
    │
    ▼
Text Transcription
```

---

## 📊 Evaluation Metric

| Metric | Description |
|---|---|
| **WER (Word Error Rate)** | Measures transcription accuracy. Lower is better. |

```
WER = (Substitutions + Deletions + Insertions) / Total Words in Reference
```

---

## 📦 Requirements

```
torch
numpy
librosa
tqdm
kenlm        # optional
```

Install all at once:

```bash
pip install torch numpy librosa tqdm
```

---

## 🗃️ Dataset

This project uses the **LJSpeech Dataset** — a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from non-fiction books.

- 📥 Download: [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)
- 🎙️ Speaker: Female, American English
- ⏱️ Total Duration: ~24 hours

---

## 🙏 Acknowledgements

- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) — Keith Ito
- [Librosa](https://librosa.org/) — Audio analysis library
- [KenLM](https://github.com/kpu/kenlm) — Language model toolkit
- [PyTorch](https://pytorch.org/) — Deep learning framework

---

## 👤 Author

**Ranjith**
- GitHub: [@Ranjith-7](https://github.com/Ranjith-7)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
