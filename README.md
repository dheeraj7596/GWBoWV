# Text Classification with Graded Weighted Bag of Word Vectors
This is the implementation of a paper accepted in Coling2016.

## Introduction
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - We propose simple feature construction technique named **Graded Weighted Bag of Word Vectors (GWBoWV).**
  - We demonstrate our method through experiments on multi-class classification on 20newsgroup dataset and multi-label text classification on Reuters-21578 dataset. 

## Testing
There are 2 folders named 20news and Reuters which contains code related to multi-class classification on 20Newsgroup dataset and multi-label classification on Reuters dataset.
#### 20Newsgroup
Change directory to 20news for experimenting on 20Newsgroup dataset and create train and test tsv files as follows:
```sh
$ cd 20news
$ python create_tsv.py
```
Get word vectors for all words in vocabulary:
```sh
$ python Word2Vec.py 200
# Word2Vec.py takes word vector dimension as an argument. We took it as 200.
```
Get Sparse Document Vectors (SCDV) for documents in train and test set and accuracy of prediction on test set:
```sh
$ python gwbowv.py 200 60
# SCDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
#### Reuters
Change directory to Reuters for experimenting on Reuters-21578 dataset. As reuters data is in SGML format, parsing data and creating pickle file of parsed data can be done as follows:
```sh
$ python create_data.py
# We don't save train and test files locally. We split data into train and test whenever needed.
```
Get word vectors for all words in vocabulary: 
```sh
$ python Word2Vec.py 200
# Word2Vec.py takes word vector dimension as an argument. We took it as 200.
```
Get Sparse Document Vectors (SCDV) for documents in train and test set:
```sh
$ python gwBoWV.py 200 60
# SCDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
Get performance metrics on test set:
```sh
$ python metrics.py 200 60
# metrics.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```

## Requirements
Minimum requirements:
  -  Python 2.7+
  -  NumPy 1.8+
  -  Scikit-learn
  -  Pandas
  -  Gensim

For theory and explanation of SCDV, please visit https://aclweb.org/anthology/C/C16/C16-1052.pdf.

If you use the code, please cite this paper:

Vivek Gupta, Harish Karnick, Ashendra Bansal, Dheeraj Mekala, "Product Classification in E-Commerce using Distributional Semantics
", in Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers,
pages 536–546, Osaka, Japan, December 11-17 2016.

@inproceedings{Gupta2016ProductCI,
  title={Product Classification in E-Commerce using Distributional Semantics},
  author={Vivek Gupta and Harish Karnick and Ashendra Bansal and Pradhuman Jhala},
  booktitle={COLING},
  year={2016}
} 

--------------------------------------------------------------------------------------------------------------------



    Note: You neednot download 20Newsgroup or Reuters-21578 dataset. All datasets are present in their respective directories.

[//]: # (We used SGMl parser for parsing Reuters-21578 dataset from  https://gist.github.com/herrfz/7967781)
