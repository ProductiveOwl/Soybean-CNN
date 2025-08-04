# Soybean-CNN
Research about the Soybean CNN for the BIBM Conference.

We also applied saliency map approach measure phenotype contribution for genome wide association study.
Creates output files on average saliency graph along with top SNPs with highest average saliency.
On program completion, provides average PCC (Pearson correlation coefficient) along with prompt
for user to search saliency based on SNP name.
The program is implemented using Keras3.5 and Tensorflow backend with python 3.1

### Prerequisites

Python packages are required,

```
absl_py==2.1.0 
astunparse==1.6.3 
certifi==2025.1.31 
charset_normalizer==3.4.1 
contourpy==1.3.1 
cycler==0.12.1 
flatbuffers==25.1.24 
fonttools==4.56.0 
gast==0.6.0 
google-pasta==0.2.0 
grpcio==1.67.0 
h5py==3.12.0 
idna==3.10 
joblib==1.4.2 
keras==3.5.0 
kiwisolver==1.4.8 
libclang==14.0.1 
Markdown==3.7 
markdown_it_py==3.0.0 
MarkupSafe==2.1.5 
matplotlib==3.10.0 
mdurl==0.1.2 
ml_dtypes==0.4.0 
namex==0.0.8 
np_utils==0.6.0
numpy==1.26.4 
opt_einsum==3.4.0 
optree==0.12.1 
packaging==24.2 
pandas==2.2.3 
pillow==11.1.0 
protobuf==4.25.4 
pygments==2.19.1 
pyparsing==3.2.3
python_dateutil==2.9.0.post0 
pytz==2025.2
requests==2.32.3 
rich==13.9.4 
scikit_learn==1.5.2 
scipy==1.15.1 
six==1.17.0 
sklearn==0.0 
tensorboard==2.17.1 
tensorboard_data_server==0.7.2 
tensorflow==2.17.0 
tensorflow_io_gcs_filesystem==0.32.0 
termcolor==2.5.0 
threadpoolctl==3.6.0 
typing_extensions==4.12.2 
tzdata==2025.2
urllib3==2.3.0 
werkzeug==3.1.3 
wrapt==1.17.2 
```

## Running the program

The scripts train and test model with 10 fold cross validation and plot a comparison of genotype contribution using saliency map value and Wald test value.

* **height.py/moisture.py/oil.py/protein.py** - *Executive scripts.*
* **run_all_folds.py** - *Runs all 10 folds individually.*
* **summarize_folds.py** - *Summarizes all 10 folds without re-running program.*
* **requirements.txt** - *Requirements of Python modules for program.*
* **IMP_height.txt/IMP_moisture.txt/IMP_oil.txt/IMP_protein.txt** - *Inputs of imputed genotype matrix.*
* **QA_height.txt/QA_moisture.txt/QA_oil.txt/QA_protein.txt** - *Inputs of quality assured non-imputed genotype matrix.*
* **polytest.txt** - *Genotype contribution using Wald test score. Run with SoyNAM R package.*
* **saliency_value.txt** - *Genotype contribution calculated using saliency map approach.*

In all three python files, change the global variable NUM_FOLDS to the folds you wish to run.

### Running on Digital Alliance Servers

After SSH'ing into servers, preform the following to run the CNN for the height phenotype.
(Bash file, CNN_setup.sh, provided to install packages)

```
module load python/3.10
virtualenv --no-download ENV
source ENV/bin/activate
cd HEIGHT
python3 height.py

alternative:
python3 height.py --fold 3 (Runs individual fold)
```

Summarize after 10 folds completed:
```
python3 summarize_folds.py

alternative:

python3 height.py --summary
```

The user can change the phenotype from height by changing the directory and program name. For example, run the CNN for the moisture phenotype, substitute 

```
cd HEIGHT
python3 height.py
```
with

```
cd MOISTURE
python3 moisture.py
```

### Running on Docker

Build image (requirements.txt file provides to install needed Python packages):
```
docker build -t snp-gwas-predictor .
```

Run full program:
```
Linux/macOS: docker run --rm -it -v "$(pwd):/app" snp-gwas-predictor
Windows CMD: docker run --rm -it -v "%cd%:/app" snp-gwas-predictor
Windows PowerShell: docker run --rm -it -v "${PWD}:/app" snp-gwas-predictor

alternatives (height.py can be replaced with any other phenotype file):

docker run --rm -v "%cd%:/app" snp-gwas-predictor python3 height.py --fold 2 (Runs selected fold and appends PCC values to export file)
docker run --rm snp-gwas-predictor python3 height.py --fold 2 (Won’t save to PCC export file)
```

Summarize after 10 folds completed (height.py can be replaced with any other phenotype file):
```
Linux/macOS: docker run --rm -it -v "$(pwd):/app" snp-gwas-predictor python3 summarize_folds.py
Windows CMD: docker run --rm -it -v "%cd%:/app" snp-gwas-predictor python3 summarize_folds.py 
Windows PowerShell: docker run --rm -it -v "${PWD}:/app" snp-gwas-predictor python3 summarize_folds.py

alternative:

docker run --rm -it -v "%cd%:/app" snp-gwas-predictor python3 height.py --summary
```

## License
GNU v2.0