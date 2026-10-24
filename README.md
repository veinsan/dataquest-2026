# DATAQUEST 5.0 — Objective Quest (OQ-DQ5)

Solusi dan catatan eksperimen untuk kompetisi Kaggle [**oq-dq-5**](https://www.kaggle.com/competitions/oq-dq-5).

| | Skor | Peringkat |
|---|---:|---:|
| Public LB | **0.48334** | 5 |
| Private LB | **0.47887** | 9 |

Tidak lolos ke babak final.

## Tugas

Memprediksi `life_satisfaction` (ordinal 1–4) responden survei kesehatan dewasa (berbasis NHIS 2021–2024) dari ±95 ribu baris data tabular. Metrik evaluasinya **Quadratic Weighted Kappa (QWK)**.

## Pendekatan

- **Regresi ordinal:** model memprediksi skor kontinu, lalu tiga *cutpoint* dioptimasi untuk QWK. Cara ini lebih stabil dibanding klasifikasi multikelas biasa.
- **Validasi:** `StratifiedGroupKFold` 5 fold dengan grup per responden. Skor CV hampir sama persis dengan Public LB (V1: 0.4797 → 0.4800, V2: 0.4814 → 0.4819).
- **Submisi final:** *convex blend* dari prediksi OOF/test historis tanpa melatih ulang model:
  `0.533·V2 + 0.216·V5.1 + 0.204·V6.1 + 0.047·V7`, dengan *threshold* `[1.569, 2.012, 2.510]`.
  OOF QWK 0.4819, Public LB 0.48334.

Kesulitan utamanya ada di batas kelas **1 vs 2** (AUC sekitar 0.78). Tidak ada model yang berhasil menembus batas ini secara berarti.

## Riwayat versi

| Versi | Ide | Hasil |
|---|---|---|
| V1 | Baseline GBDT ordinal (LGBM/XGB/CatBoost) | LB 0.48003 |
| V2 | Keragaman representasi dan objektif | LB 0.48193 |
| V3 | Fitur struktural kuesioner (*skip logic*, ontologi, TE) | Tidak mengungguli V2 |
| V4 | Eksplorasi keluarga model (*all-cat*, *retrieval*, MLP) | Negatif |
| V5 / V5.1 | Spesialis kelas 1 (TabM/RealMLP), audit *lookup* forensik | Negatif |
| V6 / V6.1 | Distilasi, CORN, *recovery* V2 + *sidecar* CatBoost | Negatif, tetapi berguna untuk blend |
| V7 | *Foundation model* (TabPFN) + *stack* | Masuk blend final |
| V8–V8.3 | Spesialis kondisional 1/2, EXAONE | Kendala *runtime* |
| V9 | TabICLv2 *full-target* | Negatif |
| V11 | *Ordinal stacker* di atas prediksi historis | Negatif |
| V12 | ModernNCA (TALENT) | Negatif |
| V14 | TabLDM | Negatif |
| V15 | *Denoising* survei + *missingness-aware recovery* | Negatif |
| `v_final` | CatBoost *raw-state* + TabM dengan *stability gate* | Eksperimen terakhir |

V10 (xRFM) gagal di *runtime* dan tidak punya notebook.

## Struktur

```
notebooks/   satu notebook per versi (dijalankan di Kaggle, T4×2)
results/     artefak per versi: OOF, prediksi test, metrik fold, summary JSON
research/    spesifikasi arsitektur akhir, audit, dan bukti diagnostik
docs/        catatan handoff (kontrak eksperimen, riwayat LB, kegagalan)
```

## Menjalankan ulang

1. Unduh data dari halaman kompetisi ke `data/` (`train_final.csv`, `test_final.csv`, `sample_submission.csv`, `data_dictionary.csv`).
2. Buka notebook di Kaggle dengan dataset kompetisi terpasang. Path input mengikuti `/kaggle/input/competitions/...`.

Data kompetisi dan salinan raw-nya tidak disertakan di repo ini (lihat `.gitignore`).
