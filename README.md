 
# Image Similarity & Digital Forensics Analyzer

A Python-based computer vision and digital forensics research prototype for image similarity detection, duplicate identification, and analysis of image transformations.

## Project Overview

This project investigates how image compression, Gaussian noise, and resizing affect visual similarity measurements.

It also integrates a pretrained CLIP model for AI-powered text-to-image retrieval.

## Key Features

### 1. Image Similarity Detection
- Compare images using perceptual difference hashing (dHash).
- Identify exact duplicate files using SHA-256.
- Analyze color histogram similarity.
- Rank images using a weighted similarity score.

### 2. Digital Forensics Experiments
- Simulate JPEG compression at different quality levels.
- Generate reproducible Gaussian noise.
- Test the effects of image resizing.
- Calculate Mean Squared Error (MSE).
- Calculate Peak Signal-to-Noise Ratio (PSNR).
- Export experimental results to CSV.

### 3. AI-Powered Image Search
- Integrate pretrained OpenAI CLIP embeddings.
- Use PyTorch and Hugging Face Transformers.
- Retrieve images using natural-language descriptions.
- Rank image matches using text-image cosine similarity.

## Technologies

- Python
- NumPy
- Pillow
- Streamlit
- PyTorch
- Hugging Face Transformers
- CLIP
- SHA-256
- Perceptual hashing

## Experimental Results

Results from preliminary tests using synthetic sample images:

| Transformation | Similarity | MSE | PSNR |
|---|---:|---:|---:|
| JPEG Quality 90 | 99.99% | 6.72 | 39.86 dB |
| JPEG Quality 35 | 91.64% | 27.31 | 33.77 dB |
| Gaussian Noise (sigma=16) | 92.79% | 232.97 | 24.46 dB |
| Downsize and Resize | 98.82% | 41.04 | 32.00 dB |

These preliminary experiments demonstrate how different image transformations affect visual similarity metrics.

## Installation

Clone the repository:

```bash
git clone https://github.com/kaurnitu222/image-forensics-similarity-analyzer.git
cd image-forensics-similarity-analyzer
```

Create a virtual environment on Windows:

```powershell
py -m venv .venv
```

Install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Open http://localhost:8501 in your browser.

The first CLIP search may require downloading model weights.

## Research Applications

The project explores foundational techniques relevant to:

- Computer vision
- Image similarity and retrieval
- Multimedia security
- Image quality analysis
- Digital forensics research
- Machine learning for image analysis

## Limitations

The similarity score is a heuristic ranking metric, not a probability of authenticity.

SHA-256 identifies identical file contents, not all visually identical images.

These experiments do not establish whether an image has been manipulated or whether digital evidence is authentic.

Results are preliminary and based on synthetic sample images, not a validated forensic benchmark dataset.

## Future Improvements

- Evaluate larger real-world image datasets.
- Compare additional perceptual hashing methods.
- Investigate image tampering localization.
- Benchmark robustness to different noise levels.
- Evaluate retrieval accuracy using standard metrics.

## Author

Nitu Kaur  
B.S. Computer Science  
University at Albany, SUNY
