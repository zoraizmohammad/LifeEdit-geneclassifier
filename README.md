<div align="center">

# Life Edit Cell Classifier

### Telling edited cells from unedited ones using single-cell DNA expression alone

[![Project](https://img.shields.io/badge/Project-DIIG%20Data%20%C3%97%20Life%20Edit-1F6B66?style=flat-square)](#overview)
[![Domain](https://img.shields.io/badge/Domain-Gene%20Editing%20%C2%B7%20Transcriptomics-C2185B?style=flat-square)](#datasets)
[![Model](https://img.shields.io/badge/Model-Random%20Forest-6E56CF?style=flat-square)](code/Elbow_mz/elbowClassifier)
[![License](https://img.shields.io/badge/License-GPL%20v3-3DA639?style=flat-square)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](#getting-started)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](code)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](code/classifier_imt)
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)](code)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](code/classifier_imt/streamlit_app_imt.py)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)](code/classifier_imt/streamlit_app_imt.py)

</div>

An ML classifier by DIIG Data for Life Edit to detect edited vs unedited cells using single-cell DNA expression data.

---

## Contents

- [Overview](#overview)
- [Method](#method)
- [Datasets](#datasets)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [Tech Stack & Techniques](#tech-stack--techniques)
- [Useful Links](#useful-links)
- [Important Notes](#important-notes)
- [For Future Reference](#for-future-reference)
- [Team](#team)
- [License](#license)

## Overview

Gene editing leaves a signature in the transcriptome, but it is not a single marker you can look up — it is a diffuse shift spread across thousands of genes, most of which are noise. This project asks whether that signature is separable from expression data alone, without knowing in advance which genes matter.

The answer the pipeline arrives at is yes, by way of an aggressive feature-selection step. Starting from **39,376 genes across 38 samples**, a statistical filter reduces the space to roughly four thousand genes whose behaviour actually differs between edited and untreated populations. Those genes are then clustered, characterised, and handed to a random forest.

## Method

**1. Normalisation.** Raw NCBI counts are log-transformed with `log2(x + 1)`, then z-scored per gene with `StandardScaler`, so genes with wildly different absolute expression become comparable.

**2. Relevance filtering.** A gene survives if the edited and untreated populations differ meaningfully in either centre or spread:

```
|median_edited − median_untreated| ≥ min_median_difference
    OR  std_edited / std_untreated ≥ 1 + min_std_percent_difference / 100
    OR  std_untreated / std_edited ≥ 1 + min_std_percent_difference / 100
```

Three thresholds were swept, and the gene sets are committed so results are reproducible without rerunning the filter:

| Gene set | Median difference | Std difference | Genes retained |
|---|---:|---:|---:|
| [`relevant_genes_1.2_275.txt`](data/5000%20Gene%20Combinations/relevant_genes_1.2_275.txt) | 1.2 | 275% | 4,122 |
| [`relevant_genes_1.6_250.txt`](data/5000%20Gene%20Combinations/relevant_genes_1.6_250.txt) | 1.6 | 250% | 4,188 |
| [`relevant_genes_3.0_250.txt`](data/5000%20Gene%20Combinations/relevant_genes_3.0_250.txt) | 3.0 | 250% | 4,050 |

Each file is a flat comma-separated list of NCBI GeneIDs.

**3. Dimensionality and structure.** PCA over the retained expression columns establishes how much variance the surviving genes carry, and an elbow sweep on the resulting space settles on **nine clusters**.

**4. Cluster characterisation.** Each cluster is annotated against NCBI gene descriptions and written up in [`data/elbowClusterData/clusterScience/`](data/elbowClusterData/clusterScience) — for example, cluster 0 resolves to a *"Divergent Pseudogene-Enriched Immunoglobulin and Transcriptomic Regulatory Cluster."* This is the step that turns a cluster index into biology.

**5. Classification.** A random forest is trained over the nine-cluster representation at the 1.6 / 250 threshold. The fitted model is committed at [`random_forest_gene_classifier_9Clusters_16_250.pkl`](code/Elbow_mz/elbowClassifier/random_forest_gene_classifier_9Clusters_16_250.pkl), with per-cluster outputs in [`data/elbowClusterData/clusterTestresults/`](data/elbowClusterData/clusterTestresults).

## Datasets

| File | Contents |
|---|---|
| [`GSE218462_raw_counts_GRCh38.p13_NCBI.tsv`](data/GSE218462_raw_counts_GRCh38.p13_NCBI.tsv) | Raw counts, GEO accession GSE218462, GRCh38.p13 |
| [`GSE218463_raw_counts_GRCh38.p13_NCBI.tsv`](data/GSE218463_raw_counts_GRCh38.p13_NCBI.tsv) | Raw counts, GEO accession GSE218463, GRCh38.p13 |
| [`Human.GRCh38.p13.annot.tsv`](data/Human.GRCh38.p13.annot.tsv) | NCBI gene annotations used to join descriptions onto GeneIDs |
| [`labeled_gene_data.csv`](data/labeled_gene_data.csv) | Labelled matrix produced by the pipeline |

Counts are indexed by `GeneID` with one column per `GSM` sample. Edited and untreated groups are split by mechanism within the normalised frame rather than by source file.

## Repository Layout

```
LifeEdit-geneclassifier/
├── code/
│   ├── EDA/                    exploratory analysis
│   │   ├── filtering_sj.ipynb        gene relevance filtering
│   │   ├── graphs_mz.ipynb           expression visualisation
│   │   └── pca_amy.ipynb             principal component analysis
│   ├── Elbow_mz/
│   │   ├── elbowTest_filtered/       elbow sweeps at each filter threshold
│   │   └── elbowClassifier/          nine-cluster random forest and model
│   ├── NLP_mz/                 NCBI description parsing and gene-text analysis
│   └── classifier_imt/
│       ├── forest_classifier_amy.ipynb
│       ├── gene_relevance_imt.ipynb
│       └── streamlit_app_imt.py      interactive PCA and filtering dashboard
├── data/
│   ├── 5000 Gene Combinations/ retained gene sets and filtered descriptions
│   ├── elbowClusterData/       cluster outputs, write-ups, and test results
│   ├── nlpClusterData/         NLP-derived cluster notes
│   └── test/                   scratch space for user-generated output
└── LICENSE
```

## Getting Started

The project targets a virtualenv named `lifeedit` (see [`.python-version`](.python-version)).

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn plotly streamlit jupyter
```

Run the notebooks in dependency order — EDA and filtering first, then the elbow sweep, then the classifier:

```bash
jupyter lab code/
```

Or launch the interactive dashboard, which performs normalisation, gene filtering and PCA on an uploaded counts file:

```bash
streamlit run code/classifier_imt/streamlit_app_imt.py
```

## Tech Stack & Techniques

- Python
- Jupyter
- pandas and NumPy for the expression matrices
- scikit-learn for `StandardScaler`, `PCA`, and the random forest
- Streamlit and Plotly for the interactive dashboard
- Statistical gene relevance filtering, elbow-method cluster selection, and NLP over NCBI gene descriptions

## Useful Links
- [Google Drive](https://drive.google.com/drive/folders/1ohv7aq8I2rCBZCLGXtBFtLMKm3vwiCHX)

## Important Notes
- All User Generated Data will be created in the data/localData folder

## For Future Reference
- Figure out a cleaner data pipeline so we don't need 10 different files for similar data
