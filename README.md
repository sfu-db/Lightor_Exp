# Lightor_Exp
To reproduce all the experiments of Lightor system
## Demo
See [here](https://ruochenj.com/channels/highlights/)

Plots of experiments can be found [here](https://github.com/sfu-db/Lightor_Exp/blob/master/exp.ipynb)
## Environment Setup
In order to reproduce the same results as in paper, Python 3.5 is required. We strong recommend you to set up a virtualenv using Anaconda to reproduce experiments:
* Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/)
* Create a virtualenv of Python 3.5 and activate it ([details](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
```
  > conda create -n py35 python=3.5
  > source activate py35 (for macOS and Linux in Terminal)
  > activate py35 (for Windows in Anaconda Prompt)
```
* Clone the repo
```
  > git clone https://github.com/sfu-db/Lightor_Exp
  > cd Lightor_Exp
```
* Install required packages and jupyter notebook
```
  > pip install -r requirements.txt
  > pip install jupyter
```
* Add py35 kernel to jupyter
```
  > pip install ipykernel
  > python -m ipykernel install --name py35
```
* Run jupyter notebook
```
  > jupyter notebook exp.ipynb
```
* Change the kernel in notebook menu: `Kernel >> Change Kernel >> py35` and run the code blocks
## Details
* All crowdsourcing data has been anonymized. (e.g. replace Worker ID with W1)
* [paper](http://www.sfu.ca/~ruochenj/files/papers/Lightor_paper.pdf) 
