# Gender-Based-Data-Bias-And-Model-Fairness-Evaluation-Framework
A systematic machine learning fairness evaluation framework integrating gender-based data bias quantification (via Earth Mover's Distance) and model-level fairness assessment. Tested on 74 disease prediction datasets across 7 machine learning algorithms and 2 fairness metrics (Equalised Odds and Treatment Equality), contributing to the development of equitable AI and supports robust informed decision-making.




### Open-access Disease Prediction Datasets 
Complete dataset List  [Click](https://github.com/MornLiang/Gender-Based-Data-Bias-And-Model-Fairness-Evaluation-Framework/blob/main/open-access%20dataset%20list.xlsx)


### Environment Configuration 
Please use "pyproject.toml".


### Models Used
- Decision Tree
- Random Forest
- Logistic Regression
- Artificial Neural Networks
- Support Vector Machine
- K-Nearest Neighbours
- Naïve Bayes



### Fairness Metrics Used
- Equalised Odds
- Treatment Equality

### File Description
Below is the detailed organization of the core components:
```
.
├── 📂Fairness Evaluation of Deployed ML Algorithm/
│   ├── 📂dataset_test/                # Experimental process files
│   ├── 📊experiment_records.xlsx      # Comprehensive log of experimental results
│   └── 📜fairtl_statisticaltest.py    # Core library for fairness statistical test
|
├──📂Inherent Bias Evaluation of Benchmark dataset
│   ├── 📂dataset_test/                # Experimental process files
│   ├── 📊experiment_records.xlsx      # Comprehensive log of experimental results
│   └── 📜fairtl_statisticaltest.py    # Core library for Data Bias test
|
├──📂Utilities
|   └── 📜fairtl_utils.py              # Useful Tools
|
|── LICENSE
├── 📖README.md                        # Project documentation
├── 📅open-access dataset list.xlsx    # Complete Open-access dataset list
└── ⚙️pyproject.toml                   # Project configuration and dependencies
```

### Declaration
The repository will be actively updated. If you have any questions, please feel free to contact: huanmorningliang@gmail.com

### Reference
 Uddin, S., Liang, H., and Guo, H., Gender-based data bias and model fairness evaluation in benchmarked open-access disease prediction datasets. Computers in Biology and Medicine. 



