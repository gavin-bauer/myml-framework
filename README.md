# MyML FRAMEWORK

<p align="center">
  <img alt="GitHub" src="https://img.shields.io/github/license/gavin-bauer/myml-framework">
</p>

---

**MyML Framework** is a personal machine learning framework to automate the implementation of models, including the preprocessing steps by _trying different algorithms and topologies, to minimize error when the model is used on test data._

This project is still in its early stages as:
- The data is assumed to have already been collected, cleaned and prepared (as the project was tested on Kaggle datasets);
- The author is experimenting with modular programming (vs. notebooks);
- The implementation of a GUI after deployment will be added in the future.

<br/>

## The project structure

The main app is located in the _src_ folder which itself is divided in:
- _data_ contains the scripts to download the datasets;
- _features_ contains the scripts to turn raw data into features for modeling;
- _models_ contains the scripts to train models and then use trained models to make predictions.

<br/>

## Installation

#### Downloading & editing the app

1. Clone the repository with the 'clone' command, or just download the zip.

```
$ git clone git@github.com:https://github.com/gavin-bauer/myml-framework.git
```

2. Install the requirements.

3. Download or Open your IDE (i.e. [Visual Studio Code](https://code.visualstudio.com/)) and start editing the files in _src_.

#### User guide

1. **Downloading and preprocessing datasets**: Open the 3 Python scripts in the folders _data_ and _features_, fill the path variables. In the terminal, run the scripts in their numerical order using the following command:

```
python src/[sub_folder]/[script_name].py
```

2. **Benchmark models**: Open _dispatcher.py_ and fill the _MODELS_ dictionary. For each model, use the name of the algorithm as the _key_, and the scikit-learn function corresponding to the model as the _value_ (e.g. "SVC": svm.SVC(kernel="linear")). Then, in the terminal, run the following command:

```
sh run.sh
```

3. **Training, finetuning and testing a model**: Open _run.sh_, comment the last 4 lines except the script you want to run (depending whether you would like to train, finetune or test a model). Then, rerun the above command in the terminal.

<br/>

## Built with

* [Python](https://www.python.org/) - Programming language
* [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/) & [Matplotlib](https://matplotlib.org/) - Data centric Python packages
* [Scikit-Learn](https://scikit-learn.org/stable/#) - Machine Learning library for Python
* [Visual Studio Code](https://code.visualstudio.com/) - Source-code editor

## Author

* **Gavin Bauer** - Data Analyst of 5+ years experience | Current: 🦉[@KeringGroup](https://www.kering.com/) | Past: ⚡[@Total](https://www.total.com/en), 🌱[@YvesRocherFR](https://groupe-rocher.com/en)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgements
This project was inspired by:
* [Abhishek Thakur](https://www.linkedin.com/in/abhi1thakur/)'s Linkedin post [Approching (Almost) Any Machine Learning Problem](https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur/)
* [Aurélien Géron](https://www.linkedin.com/in/aurelien-geron/)'s book [Hands on Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
* [Cookie-Cutter](http://drivendata.github.io/cookiecutter-data-science/)'s Data Science project template 

## Appendix

#### 1. Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
