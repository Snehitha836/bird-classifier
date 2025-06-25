Overview
This repository contains the code for a high-performance solution to the BirdCLEF 2025 Kaggle competition. The primary objective is to accurately identify 206 bird species from complex, noisy audio soundscapes.

The pipeline achieves a competitive score of 0.874 macro-averaged ROC-AUC on the private leaderboard by leveraging a sophisticated, multi-stage model ensemble accelerated with the Intel OpenVINO toolkit for efficient CPU inference.

The Inference Pipeline
The solution follows a carefully orchestrated pipeline to process raw audio and generate final predictions. This end-to-end process is the core of the project.

1. Audio Preprocessing & Mel Spectrogram Generation
The pipeline begins by loading the raw audio soundscapes. To prepare the data for the deep learning models, the audio is converted into 2D image-like representations called Mel spectrograms. This critical step is handled by both Librosa and Torchaudio, as different models in the ensemble require specific, fine-tuned preprocessing pipelines to match their original training conditions and maximize performance.

2. Accelerated Inference with Intel OpenVINO
To ensure a fast and efficient pipeline, all deep learning models are run using the Intel OpenVINO toolkit. Instead of running the original PyTorch models directly, this solution uses pre-converted and optimized versions (in .xml format). This significantly accelerates the inference process on standard CPU hardware, making it possible to process long audio files quickly without requiring a GPU.

3. Multi-Stage Model Architecture
The OpenVINO inference engine processes the spectrograms through three distinct model stages:

Stage 1: 3-Fold NFNet Ensemble
An ensemble of three Normalizer-Free Network (NFNet) models processes the spectrograms to produce a stable, high-quality baseline prediction.
Stage 2: SE-ResNeXt with Test-Time Augmentation (TTA)
A Squeeze-and-Excitation ResNeXt (SE-ResNeXt) model applies Test-Time Augmentation (TTA). This technique improves robustness by analyzing multiple, slightly shifted temporal windows of the audio features. By averaging the predictions from these augmented windows, the model becomes less sensitive to the exact position of a bird call.
Stage 3: EfficientNet-B0 & RegNetY-008 Blend
A final ensemble combines the outputs of an EfficientNet-B0 and a RegNetY-008 model, two powerful and efficient architectures adapted for spectrogram analysis.
4. Final Blending & Submission
The probability outputs from all three stages are combined using a carefully tuned weighted average ([0.05, 0.60, 0.35]). This final blended result forms the submission file, containing the likelihood of each of the 206 bird species being present in each 5-second window.

Technology Stack
Core ML/DL: PyTorch, Intel OpenVINO Toolkit
Audio Processing: Librosa, Torchaudio, soundfile
Data Manipulation: NumPy, Pandas
Environment: Python 3.11+, Kaggle Notebooks
Performance
Evaluation Metric: The official competition metric is macro-averaged ROC-AUC. This metric stringently assesses the model's ability to distinguish between classes, giving equal weight to both rare and common birds.
Final Score: The solution achieves a final score of 0.874
