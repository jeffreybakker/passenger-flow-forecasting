## Passenger Flow Forecasting in Public Transportation Networks Under Event Conditions
_Thesis will be published within 2 weeks, after which a link will appear over here._

This GitHub repository contains the implementations for the various passenger flow forecasting algorithms explored in the Master thesis of Computer Science at the University of Twente in collaboration with Info Support.

The code is copyrighted by Info Support B.V. in the Netherlands and published under the Apache-2 license.

---

### Repository structure
The repository is structured as follows:
- `models/` contains the implementations of the various Machine Learning algorithms.
- `train/` contains the Jupyter notebooks used to train the Machine Learning algorithms.
- `analysis.ipynb` is used for general data analysis and to determine the SARIMA order.
- `eventsize.ipynb` is used to visualize the relationship between the capacity of an event's venue and additional passengers in public transit (Section 4).
- `generate-results.ipynb` uses the trained ML algorithms to generate forecasting results up to 72 hours into the future.
- `forecasting-results.ipynb` visualizes the performance of the ML algorithms in tables and figures.

_Any files related to the private NS dataset are omitted from this code repository._

### Getting started
The steps below describe what is needed to get started with the code in this repository:
1. Create a [Poetry](https://python-poetry.org/) virtual environment based on Python 3.11 and install the dependencies from `pyproject.toml`.
2. Manually install [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) in the virtual Python environment.
3. Download the hourly OD ridership data from the BART website to `data/bart-od-{year}.csv.gz`: [https://www.bart.gov/about/reports/ridership](https://www.bart.gov/about/reports/ridership).
4. Parse and convert OD matrices to edge-level passenger flow data using `read_bart(...)` and `compute_flow(...)` from `util/graph.py`.
5. Train ML models with the Jupyter notebooks in `train/`.
6. Evaluate the results with `generate-results.ipynb` and `forecasting-results.ipynb`.

