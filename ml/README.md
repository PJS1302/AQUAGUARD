# AquaGuard ML workspace

This directory is the research/training workspace. The production backend loads only artifacts that you provide or explicitly promote after a real evaluation. No trained model is included in this starter because the user's original model, preprocessing objects, and validation results were not supplied.

- `preprocessing/`: contract for integrating the user's original preprocessing pipeline; none is supplied in this project snapshot.
- `regression/`: continuous-target experiments.
- `classification/`: supervised classification experiments, including a gain-ratio C4.5-style tree.
- `clustering/`: unsupervised grouping experiments.
- `dimensionality_reduction/`: PCA experiments.
- `train.py`: placeholder entry point for the user's training code.
- `evaluate.py`: placeholder entry point for the user's evaluation code.

Use `backend/models/feature_config.example.json` as a schema reference. Do not copy it to `feature_config.json` until the feature names and target match your actual trained artifact.
