**A Data Scaling Law of a Manifold's Resolution**
[Click here to view the paper (PDF)](A_Data_Scaling_Law_of_a_Manifold_s_Resolution.pdf)

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

**MANIFOLD RESOLUTION THEOREM**

**The Geometric Intuition**

This geometric framework assumes the manifold conjecture, in which a given data set corresponds to a sampling of an underlying $d$-manifold in multidimensional space. i.e. the linear regression architecture assumes that the dataset is a sampling of a linear $n$-manifold in $n+1$ space with the trivial case of a line manifold embedded in $2$ dimensional space. The decision boundaries correspond to hyperplanes that subdivide the $n$-manifold, and it is the model's objective to distinguish between these hypersubspaces by minimizing the loss between prediction and truth.

The natural extension for a more complex dataset is the definition of a more complicated $d$-manifold geometry. The feature space may no longer be linearly correlated, and hence the underlying manifold structure may assume a highly irregular, nonlinear structure. \textit{We are permitted to interpret the dataset as a discrete sampling of the underlying lower-dimensional $d$-manifold} with the underlying manifold being a continuously defined structure. The objective of the deep neural network should then be the subdivision of the $d$-manifold into hypercubic regions that correspond to class membership mappings.

In the limit, as the size of the dataset, $D$, scales, observe that a better approximation to the underlying $d$-manifold structure is attained. In other words, as $D$ scales, the resolution of the $d$-manifold structure increases and the structure clarifies. Therefore, realize that if $D$ approaches infinity, then a perfect representation of the $d$-manifold is achieved.

**Definitions and Notation**

Let us define the input space, $\mathcal{X} \subseteq \mathbb{R}^N$, of dimension $dim(\mathcal{X})=N$. Embedded within $\mathcal{X}$ there exists the $d$-manifold structure, $\mathcal{M} \subseteq \mathbb{R}^N$, of intrinsic dimensionality $dim(\mathcal{M})=d$ that is smooth and compact with $d<<N$. The consequence of compactness is the assertion of a finite volume $V_{\mathcal{M}} < \infty$ that the manifold inhabits. Further, we assume that the dataset $\{ x_1, x_2, \dots, x_D \} \in \mathcal{M}$ gets sampled from the surface of the data manifold at i.i.d. with uniform probability $p(x)$.

Next, we shall define the hypothesis class, H, of Lipschitz functions that maps a sample on the surface of ℳ to a real number expressed as f : ℳ → ℝ and |f(x)−f(z)| ≤ L‖x−z‖, ∀ f ∈ H for a real positive constant L.
The predictive function f̂ ∈ H : ℳ → ℝ corresponds to our learned model mapping.
The true function f★ : ℳ → ℝ represents the theoretically perfect mapping which may exist outside of the hypothesis class such that f★ ∈ H or f★ ∉ H.
However, we also assume that the true function is Lipschitz, thus |f★(x)−f★(z)| ≤ K‖x−z‖ for a real positive constant K.
The Lipschitz constraints imposed upon H and f★ reduce the set of all possible functions to those that do not oscillate rapidly between arbitrary pairs of neighboring data points upon the surface of the data manifold.

The true risk R(f) for some arbitrary f ∈ H is the expected MSE between f ∈ H and the true function f★.
If we assume that data is sampled i.i.d. from the d-manifold surface at uniform probability p(x) then we formulate true risk as the integral:
R(f) = 𝔼[(f(x) − f★(x))²] = ∫_ℳ (f(x) − f★(x))² p(x) dV_ℳ
for some f ∈ H where x lies on the surface of ℳ.
The empirical risk, R̂_D(f) for some f ∈ H, is the average MSE between f and the true function f★ over D data points.
This is equivalent to the training error and can be simply expressed as:
R̂_D(f) = (1/D) Σ_{i ≤ D} (f(xᵢ) − f★(xᵢ))².

Correspondingly, the true minimizer f_F★ ∈ H is the optimal function with minimum true risk such that f_H★ = argmin_{∀ f ∈ H} R(f).
Then, we shall define the empirical minimizer f̂_D ∈ H that corresponds to the optimal function with minimum empirical risk over D discrete points {x₁, x₂, …, x_D} ∈ ℳ such that f̂_D = argmin_{∀ f ∈ H} R̂_D(f).

If f_H★ is the best approximation from H to f★ over the population dataset and f̂_D is the best approximation from H to f★ over D sampled data points, then we must discover and bound the excess risk R(f̂_D) − R(f_H★) as D → ∞ from first principles.

**Reiteration of Key Assumptions**

Assumption 1: ℳ is smooth and compact i.e. ℳ is differentiable and bounding of a finite volume.
Assumption 2: The dataset {x₁, x₂, …, x_D} ∈ ℳ is distributed uniformly across the data manifold. When sampling, we assume points are taken with a uniform probability distribution at i.i.d.
Assumption 3: The Lipschitz hypothesis class H comprises smooth, non-jagged function surfaces. The true function f★ is also Lipschitz.

**Modeling the Data Manifold Resolution**

As D increases, data points inhabit the data manifold's volume at increasing resolution. If V_ℳ is finite and we have D sample points that are uniformly distributed across V_ℳ then each data point inhabits a region of V_ℳ / D space on average.

The volume of a data point in ℳ can be modeled as the volume of a d-ball in d space, Rᵈ V_d, where V_d is the volume of the unit d-ball and Rᵈ is its radius. For instance: If d = 2 then V_d = π, if d = 3 then V_d = 4π/3, and so forth.

It therefore follows that we can express the average volume that each data point occupies as Rᵈ V_d = V_ℳ / D. We are permitted to express it in this way since we assume the uniform distribution of data points across the manifold. Solve for radius to find R = (V_ℳ / V_d)^(1/d) D^(−1/d).

Furthermore, assumption 2 permits us to interpret R as an approximation to the typical radius. Therefore, we may use R to approximate the average distance between neighboring data points upon the surface of the manifold. Finally, express typical radius as R = φ D^(−1/d) which gives us a measure of the manifold's resolution.

Note that we have modeled the volume of the inhabited region by each data point as a hypersphere and not some other geometry such as a hypercube. This is justified because whether the region of each data point is modeled as a hypercube or any other arbitrary shape in d-space, the formulation for R typically yields the same structure: D^(−1/d) multiplied by some constant. It therefore follows that as D → ∞, the average-volume approximations modeled with different hyper-geometries converge to the same typical radius. The decision of a hypersphere is chosen more for its convenience.

[Click here to view the paper (PDF)](A_Data_Scaling_Law_of_a_Manifold_s_Resolution.pdf)

> **Status:** This manuscript may be subject to future revisions.

> **Material:** This paper was originally written as part of coursework at Simon Fraser University (SFU).

> **Review:** This work received informal feedback from academic researchers at the University of Illinois Urbana–Champaign (UIUC).
