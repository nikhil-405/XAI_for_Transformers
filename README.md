# XAI for Transformers: Even Better Explanations through Interpolated Propagation

This repository builds upon and extends the ideas introduced in  
[XAI for Transformers: Better Explanations through Conservative Propagation](https://arxiv.org/abs/2202.07304)  
by Ameen Ali et al. 
The original codebase can be found [here](https://github.com/ameenali/xai-transformers).

Our work introduces **Interpolated Conservative Propagation**, a generalization of Ali et al.’s method that provides a *continuous trade-off* between sensitivity-based and conservation-based reasoning. We additionally propose **GammaNet**, a learnable γ-scheduler that automatically optimizes explanation quality.

---


# Presentation  
A detailed presentation walking through the high-level overview and theoretical contributions can be found here:

[View Presentation PDF](src/Theoretical%20Overview%20and%20presentation.pdf)

Alternatively, you can also view the slides on [Canva](https://www.canva.com/design/DAG5aimzPOU/MuIAkAPRCUYjrR8rdKgcXg/edit?utm_content=DAG5aimzPOU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

---

# 📌 Overview

Transformer explainability methods often face a trade-off:

- **Gradient-based methods (e.g., GI, IG)**  
  ✔ expressive  
  ✘ break the completeness (conservation) axiom

- **LRP with detached gates (Ali et al.)**  
  ✔ conservative & stable  
  ✘ loses expressiveness by removing gradient flow through softmax / LayerNorm

This repository bridges the gap by introducing:

---

# 🚀 Interpolated Propagation

Instead of a binary choice (GI **or** LRP-AH+LN), we define:

$\hat{p}_{ij} = (1 - \gamma)\, p_{ij}^{\text{detach}} + \gamma\, p_{ij}$


where  
- $\gamma = 0$ → fully conservative (LRP(AH+LN)),  
- $\gamma = 1$ → fully sensitive (GI),  
- $0 \leq \gamma \leq 1$ → **interpolated explanations**.

We apply analogous interpolation to LayerNorm’s scaling term.

### ⭐ Key Benefit  
The completeness breach becomes:

$$
\text{breach}_\gamma = \gamma \cdot \text{breach}_{\text{GI}}
$$



meaning γ directly controls the trade-off between:

- **faithfulness** (AUAC),
- **robustness** (AUMSE),
- **completeness** (conservation error).

---

# Theoretical Contributions

We revisit Ali et al.’s derivations (provided in the attached presentation) and show:

### Attention Heads  
Softmax gradients introduce covariance terms:

$$
2 \cdot \text{Cov}_j(q_{:j}, x)
$$

Our interpolation scales this error by γ.

### LayerNorm  
Rescaling produces the conservation violation:

$$
\sum_i R(x_i)
= \left( 1 - \frac{\text{Var}[x]}{\epsilon + \text{Var}[x]} \right)
\sum_j R(y_j)
$$

Again, interpolation makes this controllable.

---

# Experiments & Evaluation Metrics

We evaluate explanations using three axes:

### **1. Completeness Error**  
Measures how well the relevance sum approximates the model output.  
(Perfect conservation → error = 0)

### **2. AUAC — Area Under the Activation Curve**  
Measures how quickly the model output activates when adding the most relevant features.

Higher AUAC → better identification of truly important features.

### **3. AUMSE — Area Under the Mean-Squared Error Curve**  
Measures output stability when removing the least relevant features.

Lower AUMSE → explanation assigns low relevance to truly unimportant features.

---

# Approach-1: Global γ Search

We perform a grid search over:

$$
\gamma_{\text{AH}},\ \gamma_{\text{LN}} \in \{0, .25, .5, .75, 1\}
$$

Key findings:

- **(0, 0)** → best conservation (matches Ali et al.)  
- **(0.75, 0.25)** → best AUAC  
- **(0.25, 0.75)** → best AUMSE  

###  Insight  
LayerNorm is the *main bottleneck* for conservation error, consistent with Ali et al.’s observations as well as with the theoretical derivations.

---

# Approach-2: GammaNet – Learning γ Automatically

We propose **GammaNet**, a lightweight MLP that predicts layer-wise γ values.

### Input  
- CLS embedding

### Output  
- γ values for each Transformer layer

### Objective (Lagrangian formulation)

$$
L = \lambda_c(f(x) - \sum_i R_i)^2 
- \lambda_a \, \text{AUAC}_{\text{proxy}}
+ \|\gamma\|_1
$$

Where:
- The first term enforces **completeness**  
- The second encourages **expressive explanations**  
- L1 regularization encourages sparse / decisive γ choices

### Training  
AUAC is non-differentiable → we incorporate it via  
- proxy functions, or  
- REINFORCE-style updates

---

#  Results

### Completeness  
GammaNet’s learned γ achieves near-perfect linear conservation — matching or exceeding LRP(AH+LN).

### Expressiveness  
GammaNet consistently improves AUAC and AUMSE across random seeds.

### Stability  
Interpolated propagation removes the pathological relevance collapse seen in GI.

---

## Repository Structure

```
├── attribution.py                   
├── xai_transformer.py                # Main attribution implementation
├── utils.py                         
├── plot_utils.py                    
├── paper_plots.ipynb                
├── baseline_detach_method/          
├── baseline_testing/                
├── GammaNet/                        
├── global_grid_search/              
└── global_grid_search_testing/      
```

---

# Presentation  
A detailed presentation walking through the high-level overview and theoretical contributions can be found here:

[View Presentation PDF](src/Theoretical%20Overview%20and%20presentation.pdf)

Alternatively, you can also view the slides on [Canva](https://www.canva.com/design/DAG5aimzPOU/MuIAkAPRCUYjrR8rdKgcXg/edit?utm_content=DAG5aimzPOU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

---

# References

- Ameen Ali, et al. **XAI for Transformers: Better Explanations through Conservative Propagation.** ICML 2022.  
  https://arxiv.org/abs/2202.07304  
- Our extended theoretical derivations and implementation.

---

# Citation

If you use this work or build upon it, please consider citing the Ali et al. paper and this repository.

```bibtex
@misc{goyal2025xai_interpolated_propagation,
  title        = {XAI for Transformers: Even Better Explanations through Interpolated Propagation},
  author       = {Goyal, Nikhil},
  year         = {2025},
  howpublished = {\url{https://github.com/nikhil-405/XAI_for_Transformers}},
  note         = {Extended implementation and analysis building on Ali et al. (2022), including Interpolated Propagation and GammaNet.}
}
```
