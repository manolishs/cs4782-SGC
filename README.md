# Simplifying Graph Convolutional Networks (SGC) Re-implementation

## Introduction

Priorly, the state-of-the-art model for semi-supervised classification task on graphs were Graph Convolutional Networks (GCNs), which is a variant of Convolutional Neural Networks (CNNs) that operates directly on graph.

This repository contains a re-implementation of the Simplifying Graph Convolutional Networks (SGC) model proposed by Wu et al., which is a simplification of GCNs. The paper shows that removing nonlinearities and collapsing weight matrices in GCNs yields a more lightweight and interpretable but equally if not more effective model.

The original paper can be found [here](https://arxiv.org/abs/1902.07153).

## Chosen Result

Our project aims to re-implement GCN and SGC and measure our testing performances on various citation networks dataset.

For reference, we have attached a subset of Table 2 from Wu et al.'s table, outlining the testing accuracy of GCN from literature as well as the testing accuracy of their implementation of GCN and SGC averaged over 10 runs.

|                                               | Cora       |  Citeseer  | Pubmed      | 
|:-                                             |:-:         |:-:         |:-:          |
| GCN (Kipf et al.'s experiment)                | 81.5       | 70.3       | 79.0        |
| GCN (Wu et al.'s experiment)                  | 81.4 ± 0.4 | 70.9 ± 0.5 | 79.0 ± 0.4  |
| SGC (Wu et al.'s experiment)                  | 81.0 ± 0.0 | 71.9 ± 0.1 | 78.9 ± 0.0  |

## GitHub Contents
    .
    ├── code                # Contains SGC model re-implementation
    ├── data                # Contains datasets used for training and evaluation
    ├── poster              # Contains PDF of the poster used for in-class presentations
    ├── report              # Contains PDF of final report submitted
    ├── results             # Contains results of our re-implementation
    ├── .gitignore
    ├── LICENSE
    └── README.md

## Re-implementation Details

**Datasets**: Citeseer​, Cora, PubMed

We reproduced a GCN and an SGC and trained them on the mentioned three datasets, with the aim of comparing accuracy with each other and those of the original papers.​

**Training Configuration**:​

**GCN​**

Hidden units: 16; Dropout: 50%; Learning rate: 0.01; Optimizer: Adam; Epochs: 200; Loss: Cross-Entropy.​

**SGC​**

One linear layer; No dropout; Learning rate: 0.2; Weight decay = 5e-6; Optimizer: Adam; Epochs: 100; Loss: Cross-Entropy.

## Reproduction Steps
1. Clone and build the repository:

```
$ git clone https://github.com/manolishs/cs4782-SGC.git
$ cd cs4782-SGC
```

2. Ensure that Python is installed:

We recommend using Python 3.8 or later. If you don't have Python installed,
    you can download it from https://www.python.org/downloads/, or use `pyenv`, `conda`, or your package manager (e.g. `brew install python` on macOS).

3. Install dependencies:

To install dependencies, you can run:

```
$ pip install -r code/requirements.txt
```

4. Train and evaluate the models:

You can run the training and evaluation scripts with:

```
$ python code/train_gcn.py
$ python code/train_sgc.py
```

5. Results:

All exported results will be stored in the `results/` directory.

## Results/Insights

The table below compares the average performance of our GCN and SGC implementations on the Citeseer, Cora, and Pubmed citation network datasets (averaged over 10 runs) with those reported by Wu et al. and in previous literature.

|                                     | Cora       |  Citeseer  | Pubmed      | 
|:-                                   |:-:         |:-:         |:-:          |
| GCN (Kipf et al.'s experiment)      | 81.5       | 70.3       | 79.0        |
| GCN (Wu et al.'s experiment)        | 81.4 ± 0.4 | 70.9 ± 0.5 | 79.0 ± 0.4  |
| **GCN (our experiment)**            | **80.7 ± 0.8** | **70.9 ± 0.7** | **79.0 ± 0.5**  |
| SGC (Wu et al.'s experiment)        | 81.0 ± 0.0 | 71.9 ± 0.1 | 78.9 ± 0.0  |
| **SGC (our experiment)**            | **80.6 ± 0.1** | **68.3 ± 0.1** | **77.9 ± 0.1**  |

As seen in the table, our reimplementation for the GCN is on par with the findings found on both papers. As for the SGC, our average is a little below that of We et al.’s experiment. We suspect that the reason for this is because we did not tune the weight decay on each training set. 

## Conclusion

This project demonstrates that Simple Graph Convolution (SGC) achieves accuracy comparable to traditional Graph Convolutional Networks (GCNs) on the Cora citation dataset, while offering significant computational efficiency gains. ​

Our reimplementation supports the findings of Wu et al. (2019), indicating that removing nonlinearities and collapsing weight matrices across layers can be an effective strategy for certain graph learning tasks.

## References

[1] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In Proceedings of the 5th International Conference on Learning Representations (ICLR 2017).

[2] Wu, F., Souza, A., Zhang, T., Fifty, C., Yu, T., & Weinberger, K. (2019). Simplifying Graph Convolutional Networks. In Proceedings of the 36th International Conference on Machine Learning (ICML 2019).

## Acknowledgements
This project was completed as the final project for CS 4782: Deep Learning at Cornell University in Spring 2025. We would like to express our sincere appreciation to Professor Jennifer J. Sun and Professor Kilian Q. Weinberger for their guidance and teaching throughout the semester.
