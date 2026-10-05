# Experiment Plan & Results
1. #  Objective
The task is to recognize a saree's surface design independently of its color palette. Images with the same motif/design but different colors should have similar embeddings, while different motifs should remain separated. The required outputs are identification/ranking and image-pair verification. The assignment also asks for an end-to-end PyTorch implementation and optional efficiency measurements.

2. # Dataset and Labeling
# Available data
- Handloom/DeepLure-derived image set: 165 JPG images.
- Additional archive dataset: organized into broad categories such as Banarasi, Bandhani, Ikat and Pichwai.
The assignment identifies the DeepLure saree corpus and Indian Saree Patterns Kaggle dataset as the intended data sources and allows cleaning/relabeling/re-splitting when documented.

# Design labels
The broad archive categories were not used as design IDs because they represent saree categories rather than the required fine-grained surface-design identity.
For the handloom set:
- Initial manually checked candidate groups: 11 design groups / 36 images.
- Candidate expansion produced 18 design groups / 59 labeled images.
- 106 images remained unlabeled.
Important limitation: the expanded labels were propagated from high-similarity candidate pairs (cosine similarity ≥ 0.93) rather than being independently verified for every image. Therefore these labels should be described as candidate-derived/heuristic labels, not a fully manually annotated ground-truth dataset.

3. # Train/Validation/Test Split
The split was performed at the design level, not at the image level, to reduce design leakage.
- Total labeled images: 59
- Total designs: 18
- Train: 12 designs / 41 images
- Validation: 2 designs / 4 images
- Test: 4 designs / 14 images
- Random seed: 42
This creates a novel-design evaluation: test design IDs are not present in training, while multiple images from each test design allow retrieval and verification.

4. # Model Architecture
# Backbone
- Pretrained ResNet18.
- Final classification layer removed.
- Backbone output: 512-D feature representation.
# Embedding head
- Linear 512 → 512
- ReLU
- Dropout 0.2
- Linear 512 → 256
- L2 normalization during embedding/evaluation.
The final representation is therefore a compact 256-dimensional embedding suitable for cosine-similarity retrieval.

5. # Preprocessing and Augmentation
# Training:
- Resize to 256.
- RandomResizedCrop to 224×224, scale 0.70–1.00.
- Random horizontal flip.
- ColorJitter: brightness/contrast/saturation/hue variation.
- Random grayscale with probability 0.35.
- Gaussian blur.
- ImageNet normalization.
# Evaluation:
- Resize to 224×224.
- Tensor conversion.
- ImageNet normalization.
The color transformations are intentional because the task requires color-invariant recognition.

6. # Training Strategy
- Framework: PyTorch.
- Loss: Supervised Contrastive Loss.
- Temperature: 0.07.
- Batch sampling: P=4 designs × K=4 samples.
- Optimizer: AdamW.
- Learning rate: 1e-4.
- Weight decay: 1e-4.
- Epochs: 20.
- Hardware: CPU for the reported run.
The training loss decreased from 2.9510 at epoch 1 to 2.0359 at epoch 20.
The checkpoint was selected using the training-loss criterion in the current training script. Validation data was retained for the design-disjoint split but was not used for automated early stopping/model selection.

7. # Identification Evaluation
Each test image is used as a query and compared against the other test images using cosine similarity.

eported results:
Metric	Result
Recall@1	1.0000
Recall@5	1.0000
Recall@10	1.0000
Test images	14
Test designs 4

Recall@k is counted as successful when at least one image of the same design appears in the top-k retrieved results.

8. # Verification Evaluation
All test-image pairs were evaluated:
- Total pairs: 91
- Positive pairs: 22
- Negative pairs: 69
# Metric	Result
ROC-AUC	1.0000
EER	0.0000
TAR @ FAR 1%	1.0000
TAR @ FAR 0.1%	1.0000

9. # Efficiency Evaluation
Measured using scripts/efficiency_code.py.
Metric	Result
Device	CPU
Parameters	11,570,496
Trainable parameters	11,570,496
FLOPs/image	3,627,909,120
GFLOPs/image	3.628
Embedding dimension	256
Embedding size	1.00 KB
Average latency	131 ms/image

The efficiency result was saved to outputs/efficiency_results.json.
10. # Interpretation
The reported test metrics are perfect on the current small test set. This indicates that the learned embedding separates the four held-out design groups successfully in this prototype experiment.
However, the result should not be interpreted as proof of generalization to a large real-world saree collection. The test set contains only 14 images from 4 designs, and the available design labels were partly generated through similarity-based candidate expansion. A larger manually verified design-level dataset is required for a stronger claim.

11. # Reproducibility Commands
From the project root:
python .\scripts	rain_metric.py
python .\scripts\evaluate_metric.py
python .\scripts\efficiency_code.py
The assignment requires a working PyTorch end-to-end implementation and evaluation protocol/results, with efficiency measurements as an optional bonus.

12. # Disclosure
The solution uses a pretrained ResNet18 backbone and PyTorch. The assignment permits pretrained backbones/checkpoints when disclosed.

The current experiment did not use the broad archive class labels as fine-grained design IDs. The final repository should keep the proprietary/source dataset out of public redistribution and should follow the assignment's instruction concerning deletion of proprietary DeepLure copies after the exercise.