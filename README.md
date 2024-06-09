# Analysis of Machine Learning Methodologies for Hydrodynamic Modeling at a Local Scale

This project presents the Master's Thesis by Mirko Rupani, a student in the Master in Data Science program (2023/2024) at the Universidad de Cantabria and the Universidad Internacional Menéndez Pelayo. The thesis was co-directed by Professors Ana Julia Abascal Santillana and Rodrigo García Manzanas.

---

## 📖 Overview

The project involves high-resolution hydrodynamic modeling of the Bay of Santander using various machine learning techniques. The focus is on currents, sea level, temperature, and salinity.

### Code Structure

The project adheres to object-oriented programming (OOP) principles, allowing for a modular and reusable structure. LowerCamelCase is used for naming variables, scripts, and other elements, except when calling external libraries that do not follow this convention.

### Execution

The project structure is designed for flexibility and ease of use. All code is executed from a single main script (`main.py`), which takes parameters from a configuration file (`config.json`). This file contains all adjustable variables and parameters, allowing users to modify it for different case studies without changing the code directly.

---

## 🛠 Configuration File (`config.json`)

The configuration file contains various parameters essential for the project's execution. Below is a detailed explanation of each section and its possible options:

### randomState
- **Description**: Seed for random number generators to ensure reproducibility.
- **Possible Values**: Any integer (e.g., `42`).

### predictands
- **predictandsFolder**: Folder path for predictand data.
- **station**: Station ID number.
- **hisFile**: List of file paths for historical data.
- **removeTimesteps**: Number of timesteps to remove from the start.
- **variables**: List of variables to model (e.g., `["u_x", "u_y", "waterlevel"]`).
- **sigmaLayer**: Sigma layer for vertical level.
- **resample**: Method for resampling data (e.g., `"mean"`).

### predictors
- **predictorsFolder**: Folder path for predictor data.
- **wind**: Source of wind data (e.g., `"meteogalicia"`).
- **hydro**:
  - **dataset_id**: List of CMEMS dataset IDs for hydrodynamic data.
  - **point**: Coordinates [latitude, longitude]. The closest available data from the specified point will be retrieved.
  - **variables**: List of lists of variables (2D and 3D) to use as predictors. Different models will be trained for each list (e.g., `[["ubar", "vbar", "zos"], ["thetao", "so"]]` will train two models), except in the case of AdaBoost (always one model per variable).
- **discharge**: File path for .mat discharge data.
- **tidalRange**: File path for .mat tidal range data.

### preprocess
- **trainTestSplit**:
  - **method**: Method for splitting data (e.g., `"temporal"`).
  - **testSize**: Proportion of data to use for testing.
- **scale**:
  - **method**: Method for scaling data (e.g., `"standard"`).
- **dimReduction**:
  - **method**: Method for dimensionality reduction (e.g., `"pca"`).
  - **nComponents**: Number of components to keep, or explained variance in the case of "pca", if using dimensionality reduction.

### model
- **method**: Machine learning method to use (`"analogues"`, `"adaboost"`, or `"lstm"`).

#### analogues
- **clustering**: Clustering method (e.g., `"spectral"`).
- **nAnalogues**: Number of analogues.
- **regressor**: Regressor type (`"knn"` or `"krr"`) to reconstruct the time-series.
- **knn**:
  - **n_neighbors**: Number of neighbors.
  - **weights**: Weight function (e.g., `"distance"`).
  - **metric**: Distance metric (e.g., `"minkowski"`).
- **krr**:
  - **kernel**: Kernel type (e.g., `"rbf"`).
  - **alpha**: Regularization parameter.
  - **gamma**: Kernel coefficient.

#### adaBoost
- **nSplits**: Number of splits for cross-validation.
- **estimator**:
  - **maxDepth**: List of maximum tree depths.
  - **criterion**: List of criteria for splitting (e.g., `["squared_error"]`).
  - **splitter**: List of split strategies (e.g., `["best"]`).
  - **minSamplesSplit**: List of minimum samples required to split.
  - **minSamplesLeaf**: List of minimum samples required at a leaf node.
- **nEstimators**: List of estimator counts.
- **learningRate**: List of learning rates.
- **loss**: List of loss functions (e.g., `["square", "exponential", "linear"]`).
- **scoring**: Scoring metric (e.g., `"neg_mean_squared_error"`).
- **nJobs**: Number of parallel jobs.

#### lstm
- **differentNetworks**: List of lists with variable groups for different networks (e.g., `[["u_x", "u_y", "waterlevel"]]`).
- **nTimesteps**: Number of timesteps for input sequences.
- **stepSize**: Step size for sequences.
- **lstmLayers**:
  - **minLstmLayers**: Minimum number of LSTM layers.
  - **maxLstmLayers**: Maximum number of LSTM layers.
  - **minLstmUnits**: Minimum units per LSTM layer.
  - **maxLstmUnits**: Maximum units per LSTM layer.
  - **stepLstmUnits**: Step size for units.
- **dropout**:
  - **minDropout**: Minimum dropout rate.
  - **maxDropout**: Maximum dropout rate.
  - **stepDropout**: Step size for dropout rate.
- **denseLayers**:
  - **minDenseLayers**: Minimum number of dense layers.
  - **maxDenseLayers**: Maximum number of dense layers.
  - **minDenseUnits**: Minimum units per dense layer.
  - **maxDenseUnits**: Maximum units per dense layer.
  - **stepDenseUnits**: Step size for units.
- **train**:
  - **optimizer**: Optimizer (e.g., `"adam"`).
  - **loss**: Loss function (e.g., `"mean_squared_error"`).
  - **metrics**: List of metrics (e.g., `"mean_squared_error"`, `"mean_absolute_error"`).
  - **learningRates**: List of learning rates.
  - **earlyStopping**:
    - **monitor**: Metric to monitor (e.g., `"val_loss"`).
    - **patience**: Number of epochs with no improvement for early stopping.
  - **batch**:
    - **minBatchSize**: Minimum batch size.
    - **maxBatchSize**: Maximum batch size.
    - **stepBatchSize**: Step size for batch size.
- **hyperband**:
  - **objective**: Objective to optimize (e.g., `"val_loss"`).
  - **direction**: Optimization direction (e.g., `"min"`).
  - **maxEpochs**: Maximum number of epochs.
  - **factor**: Reduction factor for Hyperband.
  - **overwrite**: Whether to overwrite previous results.
  - **directory**: Directory for Hyperband results.
  - **projectName**: Project name for Hyperband results.

---

## 🌱 Environment Setup

To set up the environment, use the provided `environment.yml` file. This file includes all dependencies required for the project.

---

## 🚀 Running the Project

1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```sh
    cd <repository-directory>
    ```

3. Create and activate the conda environment:
    ```sh
    conda env create -f environment.yml
    conda activate hydrodynamic-modeling
    ```

4. Run the main script:
    ```sh
    python main.py
    ```

5. Adjust the `config.json` file as needed for different scenarios and re-run the main script.

---

## 🤝 Contributions

Contributions are welcome! Please follow the standard GitHub flow for contributions: fork the repository, create a feature branch, make your changes, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For any questions or issues, please contact Mirko Rupani at [mirko.rupani@tecnalia.com].