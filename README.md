# KMTNet ToO Pipeline

A comprehensive, real-time data reduction pipeline specifically designed for processing Korea Microlensing Telescope Network (KMTNet) images, optimized for Target-of-Opportunity (ToO) observations of transient astronomical events.

## 🌟 Overview

The KMTNet ToO Pipeline is a sophisticated astronomical data processing system that automates the complete workflow from raw telescope images to transient candidate identification. It's designed to handle the unique challenges of KMTNet's multi-site, multi-band observations and provides robust, automated processing for time-critical transient follow-up observations.

### What the Pipeline Does:
- **Amplifier Combination**: Combines 32 amplifier extensions into 4 chip images (kk, mm, tt, nn)
- **Astrometric Calibration**: Automatic referencing and alignment with GAIA/APASS catalogs
- **Photometric Calibration**: Real-time zero-point adjustments and spatial homogenization
- **Quality Assurance**: Comprehensive quality control and bad pixel mapping
- **Image Stacking**: High-quality stacked image generation using SWarp
- **Transient Detection**: Advanced image subtraction with HOTPANTs for transient identification
- **Source Classification**: Multi-flag filtering system to minimize false positives

## ✨ Key Features
- Robust Quality Control: Real-time identification and filtering of readout errors and tracking artifacts to ensure data integrity from the start.

- Precision Astrometry & QA: High-accuracy alignment across the entire field of view using SCAMP integrated with the Gaia reference catalog.

- Photometric Calibration: Advanced mitigation of zero-point systematics and cross-field homogenization for consistent flux measurements.

- Artifact & Pixel Masking: Automated generation of comprehensive bad-pixel maps to neutralize cross-talk, pixel bleeding, and other detector-level defects.

- Optimized Image Stacking: Seamless construction of deep co-added science and mask images, re-projected to predefined field coordinates.

- Difference Image Analysis (DIA): State-of-the-art subtraction engine utilizing science and reference frames to detect transients via photometric parameters and Machine Learning classifiers.

- Multi-Site Integration: Fully scalable architecture designed to handle and synchronize observations across all three KMTNet sites (CTIO, SAAO, and SSO).

## 🔄 Pipeline Workflow

The pipeline follows a systematic approach to process KMTNet images from raw data to transient candidates:

### 1. **Data Ingestion & Preparation**
   - Raw KMTNet FITS files (32 amplifier extensions) are automatically detected
   - Directory structure is created and organized by observation date
   - Quality control checks identify and isolate poor-quality images

### 2. **Amplifier Combination (`ampcom`)**
   - Combines 32 amplifier extensions into 4 individual chip images (kk, mm, tt, nn)
   - Performs differential sky subtraction across amplifiers
   - Quality control: isolates images with bad CCD errors, poor seeing, or tracking issues

### 3. **Astrometric Calibration (`astrom`)**
   - Runs SExtractor to detect sources in each chip image
   - Uses SCAMP for astrometric solution with GAIA/UCAC-4 reference catalogs
   - Applies TPV (Tangent Plane) projection for accurate coordinate transformation
   - Iterative threshold adjustment for optimal astrometric precision

### 4. **Quality Assurance (`qatest`)**
   - Comprehensive quality assessment for each individual image
   - Generates bad-pixel masks (cosmic rays, cross-talk, pixel bleeding)
   - Matches sources with reference catalogs for WCS validation
   - Sectional analysis divides the image into a grid for detailed quality assessment
   - **Edge-focused astrometric QA**: the pass/fail decision is made only from the
     outermost ring of grid sections, where residual distortion is largest. The
     interior sections, which are almost always well constrained, are ignored. The
     scrutiny zone and rejection threshold are tunable via `qa_edge_ring` (ring
     width, default 1) and `qa_max_edge_bad` (default: reject if ≥2 outer-ring
     sections are bad)

### 5. **Zero-Point Scaling (`zpscale`)**
   - Photometric calibration and homogenization across amplifier regions
   - Estimates zero-points for each of 8 amplifier areas discretely
   - Corrects geometric tendencies for uniform photometric quality
   - Supports both 1D linear and 2D polynomial fitting modes

### 6. **Bad Pixel Map Update (`BPM_update`)**
   - Combines cosmic ray masks, bad pixel maps, and bad amplifier information
   - Creates comprehensive bad pixel mask with different flag values
   - Integrates multiple types of pixel defects into single mask

### 7. **Image Stacking (`stacking`)**
   - Collects complete sets of four chip images (kk, mm, tt, nn)
   - Uses SWarp for high-quality image coaddition
   - Performs quality control and coordinate alignment
   - Generates both science and mask stacks

### 8. **Source Catalog Generation (`catalogmaker`)**
   - Processes SExtractor output catalogs from stacked images
   - Calibrates existing photometric measurements against reference catalogs
   - Calculates 5-sigma detection limiting magnitude
   - Generates zero-point corrected source catalogs

### 9. **Image Subtraction & Transient Detection (`subtraction`)**
   - Performs difference image analysis using HOTPANTs
   - Applies comprehensive 10-flag filtering system to identify artifacts
   - Generates cutout images for transient candidates
   - Creates final transient candidate catalogs with quality metrics
   - **Known-object override**: an optional target list (e.g. gravitational-wave
     host-galaxy candidates or already-known transients) forces snapshot generation
     for any detection that matches a target's coordinates, regardless of its flags.
     Matched snapshots are tagged in their FITS header with `KNOWNOBJ=T` and the
     matched `TARGET` name (see [Tracking Known Targets](#tracking-known-targets))

### 10. **Transient Validation (Post-Processing)**
   - **PSF Analysis**: Use PSFEx for detailed point-spread function analysis of candidates
   - **Real/Bogus Classification**: Apply KMTNet-specific machine learning model (Lee et al., submitted to AJ)
   - **Visual Inspection**: Manual review of cutout images for final candidate validation

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Astronomical Software Dependencies**:
  - SExtractor (source extraction)
  - SCAMP (astrometric calibration)
  - SWarp (image coaddition)
  - HOTPANTs (image subtraction)
  - PSFEx (PSF analysis, optional)

### Python Dependencies
The pipeline requires several Python packages:
```bash
# Core astronomical packages
astropy>=5.0
astroquery>=0.4
numpy>=1.20
matplotlib>=3.5
scipy>=1.8

# Additional packages
astroscrappy  # cosmic ray rejection
watchdog      # file system monitoring
```

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/jmk5040/KMTNet_ToO.git
   cd KMTNet_ToO
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up directory structure:**
   The pipeline will automatically create the necessary directory structure when first run.

### Basic Usage

#### Running the Complete Pipeline
Execute the pipeline for a specific observation run (the raw-data directory name):
```bash
python pipe/KMTNet_ToO_pipeline.py 250212_CTIO
```
The positional argument is the raw-data sub-directory under `data/raw/`
(e.g. `250212_CTIO`, `YYMMDD_SITE`).

#### Monitoring Mode
For real-time processing with automatic data detection:
```bash
python pipe/KMTNet_ToO_pipeline.py AUTO
```

#### Tracking Known Targets
To force snapshots for known sources (GW host-galaxy candidates, known transients,
etc.) even when they are flagged, supply a target list with `--known-obj`:
```bash
python pipe/KMTNet_ToO_pipeline.py 250212_CTIO --known-obj S250206dm/S250206dm.csv
```
- The path is resolved relative to the `catalog/` directory (absolute paths also
  work), so the example above points to `catalog/S250206dm/S250206dm.csv`.
- The CSV must contain `Name`, `RA`, `Dec` columns (RA/Dec in decimal degrees).
  An optional `radius` column overrides the default 2″ match radius per row:

```csv
Name,RA,Dec,radius
GW_host_A,226.587365,-69.012134,3.0
KnownSN_B,230.060042,-69.020603,2.0
```
- Any detection within the match radius of a target gets a snapshot regardless of
  its flag status, and the snapshot header records `KNOWNOBJ=T` together with the
  matched `TARGET` name for provenance.

#### Individual Function Usage
You can also run individual pipeline functions:
```python
from pipe.KMTNet_ToO_functions import ampcom, astrom, qatest, zpscale, stacking, subtraction

# Example: Run amplifier combination
ampcom(path_data='/path/to/raw/data/', path_cfg='/path/to/config/')

# Example: Run astrometric calibration
astrom(path_data='/path/to/chip/images/', path_cfg='/path/to/config/', 
       path_cat='/path/to/catalogs/')

# Example: Run image subtraction with PSF analysis
subtraction(sciimg='/path/to/science.fits', path_ref='/path/to/reference/',
           path_cat='/path/to/catalogs/', path_output='/path/to/output/',
           psf_analysis=True)  # Enable PSFEx analysis
```

### Configuration
The pipeline uses a centralized configuration system:
- **Path Configuration**: `config/working_directory_structure.py`
- **SExtractor Config**: `config/kmtnet.sex`, `config/kmtnet.param`
- **SCAMP Config**: `config/kmtnet.scamp`
- **SWarp Config**: `config/kmtnet.swarp`

### Input Data Requirements
- **Raw KMTNet FITS files** with 32 amplifier extensions
- **Reference catalogs** (GAIA XP, APASS) in `catalog/` directory
- **Configuration files** in `config/` directory
- **Bad pixel maps** for each observatory in `config/badpixelmap/`

### Output Structure
The pipeline creates organized output directories:
```
data/
├── raw/          # Raw input images
├── scaled/       # Calibrated chip images
├── stack/        # Stacked images
├── subt/         # Subtraction results
└── tmpl/         # Template images

result/
├── plot/         # Diagnostic plots
└── log/          # Processing logs
```

## 📚 Documentation

The pipeline includes comprehensive documentation:
- **Function Docstrings**: Detailed documentation for all pipeline functions following NumPy/SciPy conventions
- **Inline Comments**: Extensive comments explaining complex algorithms and workflows
- **Configuration Files**: Well-documented configuration files with explanatory comments
- **Code Structure**: Modular design with clear separation of concerns

### Key Documentation Files:
- `pipe/KMTNet_ToO_functions.py`: Core pipeline functions with detailed docstrings
- `pipe/KMTNet_REF_functions.py`: Reference image processing functions
- `config/working_directory_structure.py`: Centralized path configuration with extensive comments
- `pipe/KMTNet_ToO_pipeline.py`: Main pipeline script with workflow documentation

## 🔧 Advanced Usage

### 2D Zero-Point Calibration for KS4 DR1 Reference Images

For 2-dimensional zero-point calibration of KS4 DR1 reference images, use the `zeropoint_homogenization` function in `KMTNet_REF_functions.py`. This function provides advanced spatial homogenization of photometric zero-points across the image field.

#### Usage:
```python
from pipe.KMTNet_REF_functions import zeropoint_homogenization

# Apply 2D zero-point correction
zeropoint_homogenization(img='/path/to/reference_image.fits', 
                        path_map='/path/to/correction_maps/', 
                        outname='corrected_image.fits',
                        aperture='APER5', 
                        mode='single', 
                        zp_to_scale=30)
```

#### Download Correction Maps:
The required correction maps for KS4 DR1 reference images can be downloaded from:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17336381.svg)](https://doi.org/10.5281/zenodo.17336381)
### Customizing the Pipeline
The pipeline is designed to be highly customizable:
- **Configuration Files**: Modify SExtractor, SCAMP, and SWarp parameters
- **Quality Cuts**: Adjust filtering criteria in individual functions
- **Output Formats**: Customize output file naming and organization
- **Processing Modes**: Enable/disable specific processing steps

### Performance Optimization
- **Parallel Processing**: Multi-core support for image stacking and transient detection
- **Memory Management**: Efficient handling of large KMTNet images
- **Disk I/O**: Optimized file operations for high-throughput processing
- **Vectorized Masking**: Saturation-bleed masking is fully NumPy-vectorized (with
  subsampled background statistics), reducing per-chip masking time from minutes to
  a few seconds
- **Fault Tolerance**: Per-image and per-step error isolation so that a single bad
  chip never aborts a batch run. Optional steps (cosmic-ray rejection, asteroid
  look-ups, crosstalk flagging) degrade gracefully when external services or
  metadata are unavailable

### Transient Candidate Validation
For complete transient validation, the pipeline output should be processed with additional tools:

#### PSF Analysis with PSFEx
```python
# Enable PSF analysis in the subtraction function
subtraction(..., psf_analysis=True)

# This will:
# 1. Run PSFEx to model the point-spread function
# 2. Apply PSF photometry to transient candidates
# 3. Flag sources with poor PSF fits (flag_9)
```

#### Real/Bogus Classification
The pipeline generates cutout images for visual inspection, but for automated classification:
- **KMTNet-Specific Model**: A dedicated Real/Bogus classification model for KMTNet images (Lee et al., submitted to AJ) will be publicly available soon
- **Current Workflow**: Manual visual inspection of cutout images in the `snap/` directory
- **Future Integration**: The Real/Bogus model will be integrated for automated candidate filtering

#### Complete Validation Workflow
1. **Pipeline Processing**: Run the complete KMTNet ToO pipeline
2. **PSF Analysis**: Enable PSFEx analysis for detailed source characterization
3. **Real/Bogus Classification**: Apply the KMTNet-specific machine learning model (when available)
4. **Visual Inspection**: Review cutout images for final candidate validation
5. **Follow-up Observations**: Schedule additional observations for confirmed transients

## 🐛 Troubleshooting

### Common Issues:
1. **Missing Dependencies**: Ensure all astronomical software (SExtractor, SCAMP, etc.) is installed
2. **Path Issues**: Check that all configuration files are in the correct locations
3. **Memory Errors**: For large datasets, consider processing in smaller batches
4. **Permission Errors**: Ensure write permissions for output directories

### Getting Help:
- Check the comprehensive function docstrings for detailed parameter descriptions
- Review the inline comments in configuration files
- Submit issues on GitHub with detailed error messages and system information

## 🤝 Contributing

We welcome contributions to improve the KMTNet ToO Pipeline:
- **Bug Reports**: Submit detailed issue reports with system information
- **Feature Requests**: Propose new functionality or improvements
- **Code Contributions**: Submit pull requests with well-documented code
- **Documentation**: Help improve documentation and examples

### Development Guidelines:
- Follow existing code style and documentation conventions
- Add comprehensive docstrings for new functions
- Include inline comments for complex algorithms
- Test changes with sample data before submitting

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📖 Citation

If you use this pipeline in your research, please cite:

Jeong, M. et al. (2026) - KMTNet Synoptic Survey of Southern Sky II: Data Reduction and Real-Time Transient Detection Pipeline, arXiv:2603.17442 [astro-ph.IM]
*Note: The full citation will be updated once the paper is published.*

## 👨‍💻 Maintainer

**Mankeun Jeong** - Pipeline development and maintenance

---



