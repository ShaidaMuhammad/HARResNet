# Enhanced Human Activity Recognition Using Inertial Sensor Data from Smart Wearables: A Neural Network Approach with Residual Connections

This project/code is based on the paper:

**S. Muhammad, K. Hamza, H. A. Imran and S. Wazir, "Enhanced Human Activity Recognition Using Inertial Sensor Data from Smart Wearables: A Neural Network Approach with Residual Connections," 2024 International Conference on Engineering & Computing Technologies (ICECT), Islamabad, Pakistan, 2024, pp. 1-6, doi: 10.1109/ICECT61618.2024.10581358.**

## Structure

- `data/`: Contains the WISDM dataset.
- `src/`: Contains all source code.
  - `data_processing.py`: Functions for data loading and preprocessing.
  - `models.py`: Defines the neural network model.
  - `train.py`: Training and evaluation routines.
  - `utils.py`: Utility functions.
  - `evaluate.py`: Model evaluation script.
- `main.py`: Main script to run the project.
- `requirements.txt`: List of dependencies.

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To train the model, place the WISDM dataset in the `data/` directory in `.txt` format and run:

   ```bash
   python main.py
   ```

The training script will process the data, train the neural network, and save the best model as `best_model.keras`.

## Evaluating the Model

To evaluate or use the pre-trained model (`best_model.keras`), you need to convert your test dataset to a CSV file. The CSV file should have the following columns:

- `user`
- `activity`
- `timestamp`
- `x-axis`
- `y-axis`
- `z-axis`

Ensure the `z-axis` values are formatted as float without any semicolon.

Place your test CSV file in the desired directory and run the evaluation script:

   ```bash
   python src/evaluate.py best_model.keras path/to/test_data.csv
   ```

## Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{muhammad2024enhanced,
  title={Enhanced Human Activity Recognition Using Inertial Sensor Data from Smart Wearables: A Neural Network Approach with Residual Connections},
  author={Muhammad, Shaida and Hamza, Kiran and Imran, Hamza Ali and Wazir, Saad},
  booktitle={2024 International Conference on Engineering \& Computing Technologies (ICECT)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
