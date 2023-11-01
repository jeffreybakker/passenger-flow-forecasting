# Passenger Flow Forecasting in Public Transportation Networks Under Event Conditions
_University of Twente Master thesis: [essay.utwente.nl/97536/](https://essay.utwente.nl/97536/)_
_(Link will work starting 2023-11-06)_

This GitHub repository contains the implementations for the various passenger flow forecasting algorithms explored in the Master thesis of Computer Science at the University of Twente in collaboration with Info Support.

The code is copyrighted by Info Support B.V. in the Netherlands and published under the Apache-2 license.

---

## Why?

Public transportation networks are an essential part of public infrastructure and have the potential to reduce society's dependency on personal cars. Making public transportation a more attractive option requires better passenger demand forecasts, as these can be used to guarantee enough seating capacity for everyone, especially under event conditions (such as concerts, festivals, and sports).

Even though passenger demand in public transportation tends to be quite regular, accurately forecasting the additional peaks of passengers caused by large events has proven to be quite challenging. This research attempts to accurately forecast these peaks in additional demand based on some details about the events. The results of this research did not confirm the hypothesis, but some positive results show a potential for future works that have access to better quality data. _Read the thesis for more details: [essay.utwente.nl/97536/](https://essay.utwente.nl/97536/)_
_(Link will work starting 2023-11-06)_

![Visualized passenger flow](assets/flow.gif)


## Running the code

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

