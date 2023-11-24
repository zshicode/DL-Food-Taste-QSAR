# Deep learning-based food-compound QSAR of food tastes

Machine learning-based high-throughout virtual screening (HTVS) has been applied for molecular taste prediction using in food chemistry (Rojas et al, 2023; Song et al., 2023). We adopt neural networks (CNN or LSTM) for quantitative structure-activity relationship (QSAR) of food taste response using food-compound matrix.

## Requirements

The code has been tested running under Python 3.9.12, with the following packages and their dependencies installed:

```bash
numpy
pandas
pytorch
sklearn
matplotlib
seaborn
rdkit
```

## Usage

1. Run `dataprepare.py` to preprocess the data.

2. Run `score.py` to run the model.

```bash
python main.py --data 1 --model LSTM --taste bitter
```

After running `score.py`, the script outputs the predicted taste score of foods (see Method), and saves the result as CSV file.

For comparison, this repo also implements SVM and LightGBM for this task, see `baseline.py`.

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--data', type=int, default=1,
                    help='Dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='Model')
parser.add_argument('--taste', type=str, default='bitter',
                    help='Taste')               

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

## Datasets

Herein, `taste.csv` is collected from https://github.com/songyu2022/taste_predict (Song et al., 2023). Each molecule in `taste.csv` includes three labels: bitter, sweet and umami. The label is 1 if the molecule is bitter (or sweet or umami), else 0.

In each dataset, `food.csv` includes names and categories of foods. `compound.csv` includes names and SMILES codes of compounds. `food-compound.csv` includes food-compound associations.

||Data source|Paper|GitHub link|
|:--:|:--:|:--:|:--:|
|Data1|[FooDB](https://foodb.ca/)|(Su et al., 2023)|https://github.com/TinaCausality/NutriFD_Dataset|
|Data2|(Ahn et al., 2011)|(Ahn et al., 2011)|https://github.com/lingcheng99/Flavor-Network|
|Data3|[FooDB](https://foodb.ca/)|(Rahman et al., 2021)|https://github.com/mostafiz67/FDMine_Framework|

Food-compound matrix is with elements as the score of food-compound pairs. 

In Data1, suppose the content of compound k in food i is a mg/100g (+- 1e-3 mg), then `score[i,k] = 0.1*(log10(a+1e-5)+5)` can scale the amount into [0,1] (since 100g = 1e5 mg).

In Data2, if there is compound k in food i, then `score[i,k] = 1`, otherwise `score[i,k] = 0`.

Data3 is a refined version of Data1. However, the score is defined as contribution rate of compound content (Rahman et al., 2021), scaling into [0,1].

## Method

We adopt Morgan fingerprint (length=1024, radius=3, i.e. ECFP6) as molecular feature. The model is trained and evaluated by 2-fold cross validation for classification of the data in `taste.csv`. Then, the model is adopt for bitter (or sweet or umami) taste prediction of food-compounds in Data1 (or Data2 or Data3). The taste prediction of a food is the weighted mean value of the compounds in this food, weighted by the score of food-compound pairs.

## References

Ahn et al., Flavor network and the principles of food pairing, Sci Rep, 2011

Rahman et al., A novel graph mining approach to predict and evaluate food-drug interactions, Sci Rep, 2021

Rojas et al., Classification-based machine learning approaches to predict the taste of molecules: A review, Food Res Int, 2023

Song et al., A Comprehensive Comparative Analysis of Deep Learning Based Feature Representations for Molecular Taste Prediction, Foods, 2023

Su et al., NutriFD: Proving the medicinal value of food nutrition based on food-disease association and treatment networks, arXiv, 2023