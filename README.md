# AIE-CASE: Color-Invariant Saree Design Recognition
A PyTorch-based metric-learning system for recognizing saree surface designs independently of their color palette.
Objective
The goal is to identify the same saree surface design even when the saree appears in different colors, while distinguishing different motifs with similar colors.
# The system supports:
- Design identification / image retrieval
- Image-pair verification
- Compact 256-dimensional design embeddings
- Efficiency measurement
# Approach
The project uses a pretrained ResNet18 backbone followed by a 256-dimensional embedding head.
```text
Saree Image
    ↓
224×224 preprocessing
    ↓
Color / appearance augmentation
    ↓
Pretrained ResNet18
    ↓
512-D feature
    ↓
Embedding Head
512 → 512 → 256
    ↓
L2-normalized 256-D embedding
    ↓
Cosine Similarity
```
The model is trained using Supervised Contrastive Loss so that images belonging to the same design are closer in embedding space while different designs are separated.
Color-invariance is encouraged using ColorJitter, random grayscale, and Gaussian blur.
# Dataset
The available handloom image set contains 165 JPG images.
An additional archive dataset contains broad categories such as Banarasi, Bandhani, Ikat and Pichwai. These broad categories were not used as fine-grained design IDs because the task requires surface-design identity.
# For the current experiment:
- Labeled images: 59
- Design groups: 18
- Unlabeled images: 106


The expanded labels were partly derived from high-similarity candidate pairs. They should therefore be treated as candidate-derived/heuristic labels, not as a fully manually verified ground-truth annotation.
# Data privacy
Do not commit proprietary/source datasets to a public repository. Follow the assignment instructions regarding proprietary DeepLure data and deletion after the exercise.
# Train / Validation / Test Split
The split is performed by design, rather than randomly by image, to reduce design leakage.
 Split	    Designs	 Images
- Train	      12	   41
- Validation   2	   4
- Test	       4	  14
- Total	       18	  59
- Random seed: 42


The test designs are therefore not present in the training split.
# Preprocessing
#  Training
- Resize to 256
- RandomResizedCrop to 224×224
- Random horizontal flip
- ColorJitter
- Random grayscale, probability 0.35
- Gaussian blur
- ImageNet normalization
#  Evaluation
- Resize to 224×224
- Tensor conversion
- ImageNet normalization
The color transformations are specifically intended to reduce dependence on color palette.
# Training Configuration
 Parameter	Value
- Framework:PyTorch
- Backbone:	ResNet18
- Loss:	Supervised Contrastive Loss
- Temperature:	0.07
- Sampling:	P=4, K=4
- Embedding dimension:	256
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay:	1e-4
- Epochs:	20
- Device:	CPU

Training loss decreased from 2.9510 at epoch 1 to 2.0359 at epoch 20.
# Evaluation
Identification
Each test image is compared with the other test images using cosine similarity.
Metric	Result
- Recall@1:	1.0000
- Recall@5:	1.0000
- Recall@10:	1.0000

# Test set:
- 14 images
- 4 designs
# Verification
All test-image pairs were evaluated.
- Total pairs: 91
- Positive pairs: 22
- Negative pairs: 69
# Metric	Result
- ROC-AUC:	1.0000
- EER:	0.0000
- TAR @ FAR 1:	1.0000
- TAR @ FAR 0.1%:	1.0000

# Efficiency
Efficiency was measured using:- python .\scripts\efficiency_code.py

# Metric	Result
Device:	CPU
Parameters:	11,570,496
Trainable parameters:	11,570,496
FLOPs / image:	3.628 GFLOPs
Embedding dimension: 256
Embedding size:	1.00 KB
Average latency: 131 ms/image

# Results are saved to:
outputs/efficiency_results.json

# Project Structure
```text
DeepLure_ColorInvariant_Saree_project/
│
├── configs/
│   └── default.yaml
│
├── data/
│   └── processed/
│       ├── design_lebel.csv
│       ├── expanded_design_labels.csv
│       ├── initial_design_labels.csv
│       ├── manifest.csv
│       └── verified_pairs.csv
│
├── outputs/
│   ├── candidate_pair_sheet.jpg
│   ├── candidate_pairs.csv
│   ├── candidate_review.csv
│   ├── candidates_1.jpg
│   ├── candidates_2.jpg
│   ├── candidates_3.jpg
│   ├── efficiency_results.json
│   ├── evaluation_results.csv
│   └── handloom_contact_sheet.jpg
│
├── scripts/
│   ├── build_manifest.py
│   ├── create_initial_labels.py
│   ├── efficiency_code.py
│   ├── evaluate_metric.py
│   ├── expand_design_labels.py
│   ├── find_candidate_groups.py
│   ├── make_candidate_pair_sheet.py
│   ├── make_candidate_sheets.py
│   ├── make_handloom_contact_sheet.py
│   ├── make_split.py
│   ├── split_manifest.py
│   └── train_metric.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── model.py
│   └── utils.py
│
├── APPROACH_NOTE.md
├── EXPERIMENT_PLAN.md
├── README.md
├── requirements.txt
└── .gitignore
```
# Main folders
- .vscode/ — VS Code project configuration.
- configs/ — experiment configuration files.
- data/RAW/ — raw datasets. These should not be committed if they contain proprietary/source data.
- data/processed/ — design labels, verified pairs and train/validation/test manifests.
- outputs/ — generated candidate sheets, trained checkpoint, evaluation results and efficiency results.
- scripts/ — scripts for dataset preparation, labeling, training, evaluation and efficiency measurement.
- src/ — reusable Python modules for data handling, model definition, losses, evaluation and utilities.

# Important files
 ## File	Purpose
- scripts/train_metric.py	Trains the color-invariant metric-learning model
- scripts/evaluate_metric.py	Evaluates identification and verification
- scripts/efficiency_code.py	Measures parameters, FLOPs, embedding size and latency
- scripts/find_candidate_groups.py	Finds visually similar candidate image pairs
- scripts/create_initial_labels.py	Creates initial design labels
- scripts/expand_design_labels.py	Expands candidate-derived design labels
- scripts/make_split.py	Creates the design-level train/validation/test split
- data/processed/manifest.csv	Final dataset split manifest
- outputs/color_invariant_resnet18_best.pt	Trained model checkpoint
- outputs/evaluation_results.csv	Evaluation results
- outputs/efficiency_results.json	Efficiency measurements
- APPROACH_NOTE.md	Short technical approach
- EXPERIMENT_PLAN.md	Detailed experiment plan and results


# Installation
Create and activate a Python environment, then install the required packages:
pip install -r requirements.txt
The project uses PyTorch and related Python packages for image processing, model training and evaluation.


# Running the Project
 # Prepare / update the manifest
Use the project scripts for candidate generation, labeling and splitting when rebuilding the dataset.
 # Train
python .\scripts\train_metric.py
The trained checkpoint is saved as:
outputs/color_invariant_resnet18_best.pt
 # Evaluate
python .\scripts\evaluate_metric.py
## This reports:
- Recall@1
- Recall@5
- Recall@10
- ROC-AUC
- EER
- TAR@FAR 1%
- TAR@FAR 0.1%
 # Measure efficiency
python .\scripts\efficiency_code.py
# Limitations
The current results should be interpreted as a prototype experiment.
The test set contains only 14 images from 4 designs, and the expanded labels were partly generated using high-similarity candidate propagation. Therefore, the perfect test metrics should not be interpreted as 100% real-world accuracy.
# A stronger evaluation would require:
- More manually verified design IDs
- More images per design
- A larger design-disjoint test set
- Multiple colorways per design
- Additional external validation data
# Conclusion
The current prototype demonstrates the feasibility of learning compact saree-design embeddings that are robust to color and appearance changes.
On the current small design-disjoint test set, the model achieved:
- Recall@1: 100%
- ROC-AUC: 1.000
- EER: 0.000
- 3.628 GFLOPs/image
- 131 ms/image CPU latency
- 256-D embeddings


These results are encouraging for the prototype, while larger manually verified datasets are needed to establish generalization.
Assignment Compliance
The implementation uses PyTorch, provides an end-to-end training/evaluation pipeline, documents preprocessing and training choices, reports identification and verification results, and includes model-efficiency measurements.
The pretrained ResNet18 backbone is disclosed as part of the approach.
