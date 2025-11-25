**A Data Scaling Law of a Manifold's Resolution**
[Click here to view the paper (PDF)](A_Special_Case_of_Nonlinear_Extrapolation_Under_ReLU_and_the_Neural_Tangent_Kernel.pdf)

*Author(s)*: Abiel J. Kim

*Date*: April 2025

*Keywords*: Neural Scaling Law, Data, Machine Learning, Deep Learning, Statistical Learning Theory, Differenial Geometry, Manifold Learning, Intrinsic Dimension, Bound, Lipschitz, Hypothesis Class, Probability Theory

**ABSTRACT**

It has been empirically observed that neural network performance generally assumes the power law formulation for the scaling of its training dataset. 
The experimental evidence is compelling, but the theoretical frontier remains exploratory with respect to a mathematical origin of the observed power law. 
This paper introduces a mathematical framework of the geometric kind that enables the emergence of a bounding of the data scaling law. 
The mathematical framework is predicated on the manifold conjecture and interprets the scaling of a dataset as a finer approximation to the true data manifold space. 
The equations indicate that model loss, $L$, indeed scales as a power law with $L \propto D^{-1/d}$ for the data manifold's intrinsic dimensionality, $d$.

**INTRODUCTION**

As per the question of the neural scaling laws posed by researchers at OpenAI, Kaplan et al. (2020) proposed the empirically observed power law formulations for model capacity (N), dataset size (D), and compute (C).
And as discussed in lecture, we saw the empirical literature surrounding the proposed power law formulation for the scaling of D. 
We also discussed the idea that there does not yet exist a universally accepted theoretical solution regarding the mathematical origin of such observed power laws.

The objective of this paper is to discover a theoretical upper bound for the data scaling law from first principles which has been empirically fitted as a power law formulation. 
The implication of such a discovery would provide either confirming or disconfirming evidence for the fitting of the power law with respect to data scaling. 
In order to achieve this, I will provide firstly a geometric intuition of my framework. 
Then, I will provide a more detailed mathematical derivation that leverages some of the mathematical properties of the geometric framework which leverages the manifold conjecture. 
Note that the manifold conjecture is a mainstream approach in terms of theoretical analysis in the machine learning field. 

> **Status:** This manuscript may be subject to future revisions.
