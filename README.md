# KMTNet_ToO Pipeline

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
- **Modular Design:** Well-organized utility functions for easy maintenance and extension.
- **Reference Image Generation:** Support for both KS4 and Pan-STARRS reference images.

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

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/jmk5040/KMTNet_ToO.git
cd KMTNet_ToO

# Install core dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Full Installation (with optional features)

```bash
# Install with all optional dependencies
pip install -e .[all]

# Or install optional dependencies separately
pip install -r requirements-optional.txt
```

### Optional Dependencies

Some features require additional packages:

- **`astroscrappy`**: For cosmic ray rejection in ToO functions
- **`scikit-learn`**: For machine learning features (Real/Bogus classification)
- **`PanStitch`**: For Pan-STARRS reference image generation (fallback when KS4 unavailable)

## Usage

### Running the Pipeline
Execute the pipeline for a specific observation date:
```bash
python KMTNet_ToO_pipeline.py YYYYMMDD
```
Replace `YYYYMMDD` with your observation date or `AUTO` for input data monitoring.

### Using Individual Modules

```python
import KMTNet_util_functions as util
import KMTNet_ToO_functions as too
import KMTNet_REF_functions as ref

# Example: Bad pixel correction
corrected_img = ref.badpixel_clear(
    img='science.fits',
    mask='mask.fits', 
    outname='corrected.fits',
    path_cfg='/path/to/config/'
)

# Example: Photometry
too.ks4_photometry(
    img='corrected.fits',
    mask='mask.fits',
    path_output='./output/',
    path_cfg='/path/to/config/'
)
```

### Reference Image Generation

The pipeline supports multiple reference image sources:

1. **KS4 Reference Images** (preferred)
2. **Pan-STARRS Reference Images** (fallback, requires PanStitch)

```python
# Generate Pan-STARRS reference (requires PanStitch)
ref_img = ref.generate_panstarrs_reference(
    field='1234.5',
    cra='12:34:56.7',
    cdec='+12:34:56.7',
    path_output='./references/',
    path_cfg='/path/to/config/'
)
```

## File Structure

```
KMTNet_ToO/
├── KMTNet_util_functions.py      # Common utility functions
├── KMTNet_ToO_functions.py       # Target of Opportunity functions
├── KMTNet_REF_functions.py       # Reference image functions
├── KMTNet_ToO_pipeline.py        # Main pipeline script
├── KMTNet_ToO_observation.py     # Observation management
├── KMTNet_ToO_database.py        # Database operations
├── requirements.txt               # Core dependencies
├── requirements-optional.txt      # Optional dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Dependencies

### Core Dependencies
- `numpy>=1.20.0`
- `astropy>=4.0.0`
- `matplotlib>=3.0.0`
- `scipy>=1.7.0`
- `astroquery>=0.4.0`

### Optional Dependencies
- `astroscrappy>=1.1.0` (cosmic ray rejection)
- `scikit-learn>=1.0.0` (machine learning)
- `PanStitch>=1.0.0` (Pan-STARRS reference images)

## Error Handling

The pipeline includes graceful error handling for optional dependencies:

- If `PanStitch` is not installed, Pan-STARRS reference generation will raise a clear error message
- If `astroscrappy` is not installed, cosmic ray rejection will raise a clear error message
- Users can install optional dependencies as needed for their specific use cases

## Tutorial
Detailed tutorials and examples are provided in the [Tutorial Folder](#), showcasing the pipeline step-by-step with example data and results.

## Contributing
Contributions, bug reports, and feature requests are welcome. Please submit an issue or pull request on GitHub.

## License
Distributed under the MIT License. See `LICENSE` for more information.

---

**Maintained by Mankeun Jeong.**