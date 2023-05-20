# Approximation of Nearly-Periodic Symplectic Maps via Structure-Preserving Neural Networks

**Authors: [Valentin Duruisseaux](https://sites.google.com/view/valduruisseaux), Joshua W. Burby, and Qi Tang**


<br />

This repository provides a simplified version of the Python/TensorFlow code used to generate some of the results in our paper



   [**Approximation of Nearly-Periodic Symplectic Maps via Structure-Preserving Neural Networks.**](https://arxiv.org/abs/2210.05087)
<br />
   Valentin Duruisseaux, Joshua W. Burby, and Qi Tang.
   <br />
   *Scientific Reports, Collection on "Physics-informed Machine Learning and its real-world applications"*, 2023.




<br />


<hr>

## List of Files


The code is provided in two different formats

```
    NPMap_Learning.py     NPMap_Learning.ipynb
```

<br />

These codes contain detailed explanations with equations in LaTeX which are better rendered in the Jupyter notebook version.

<br />

The directory  [`./TrainingWeights/`](TrainingWeights)  contains the weights of the trained model whose results are presented in Figure 5 of our paper.

These weights can be loaded into the model by setting
```
    train = False
```



<br />


<hr>

## Usage

```
    python ./NPMap_Learning.py
```



<br />
<hr>








## Additional Information

<br />

This code is also published and available at  [https://www.osti.gov/biblio/1972078/](https://www.osti.gov/biblio/1972078/)

<br />

If you use this code in your research, please consider citing:


```bibTeX
@article{Duruisseaux2023NPMap,
	author = {V. Duruisseaux, and J. W. Burby, and Q. Tang},
	title  = {Approximation of nearly-periodic symplectic maps via structure-preserving neural networks},
	url    = {https://arxiv.org/abs/2210.05087},
	year   = {2023}
	
}
```
```bibTeX
@article{Duruisseaux2023NPMapCode,
	title = {Code Demonstration: Approximation of nearly-periodic symplectic maps via structure-preserving neural networks},
	author = {V. Duruisseaux, and J. W. Burby, and Q. Tang},
	doi = {10.2172/1972078},
	url = {https://www.osti.gov/biblio/1972078}, 
	year = {2023}
}
```





The software is available under the [MIT License](https://github.com/vduruiss/SymplecticGyroceptron/blob/main/LICENSE).
