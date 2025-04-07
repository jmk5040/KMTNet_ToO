 Pipeline

A real-time data reduction pipeline specifically designed for processing Korea Microlensing Telescope Network (KMTNet) images, optimized for Target-of-Opportunity (ToO) observations of transient astronomical events.

## Overview

The pipeline automates:
- Photometric and astrometric calibrations.
- Reference image and catalog generation.
- Image subtraction for transient identification.
- Real/Bogus (R/B) classification for efficient transient candidate filtering.

## Features

- **Real-Time Processing:** Rapid reduction and analysis of transient events.
- **Automated Calibration:** Includes astrometric alignment and photometric zero-point scaling.
- **Transient Identification:** Uses image subtraction methods optimized for KMTNet.
- **Machine Learning Integration:** Real/Bogus classification model implementation for reliable transient detections.

## Pipeline Workflow

1. **Data Preparation:**
   - Raw images collected and stored systematically.

2. **Astrometric Calibration:**
   - Automatic referencing and alignment with catalogs (e.g., APASS).

3. **Photometric Calibration:**
   - Real-time zero-point adjustments and scaling.

4. **Image Stacking & Subtraction:**
   - Generation of high-quality stacked images.
   - Robust subtraction to detect transient sources effectively.

5. **Catalog Generation:**
   - Comprehensive catalogs of detected sources and transient candidates.

6. **Real/Bogus Classification:**
   - AI-driven filtering of transient candidates.

## Usage

### Requirements
- Python >3.8
- Astropy
- Astroquery
- NumPy
- Matplotlib
- KMTNet-specific calibration files

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/jmk5040/KMTNet_ToO.git
cd KMTNet_ToO
pip install -r requirements.txt
```

### Running the Pipeline
Execute the pipeline for a specific observation date:
```bash
python KMTNet_ToO_pipeline.py YYYYMMDD
```
Replace `YYYYMMDD` with your observation date or `AUTO` for input data monitoring.

## Tutorial
Detailed tutorials and examples are provided in the [Tutorial Folder](#), showcasing the pipeline step-by-step with example data and results.

## Contributing
Contributions, bug reports, and feature requests are welcome. Please submit an issue or pull request on GitHub.

## License
Distributed under the MIT License. See `LICENSE` for more information.

---

Maintained by Mankeun Jeong.



