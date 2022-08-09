Decision Tree Robustness Validation using SAT
===
This is an extended implementation of 
```
Narodytska N, Ignatiev A, Pereira F, Marques-Silva J, Ras I. Learning Optimal Decision Trees with SAT. InIjcai 2018 Jul 13 (pp. 1362-1368).
```
We use the above paper to represent the decision trees with propositional logic and use this representation to find adversarial examples, to test for model robustness.

This is code is mainly based on the code written by `Chengxi Yang` at https://github.com/kamikaze0923/SAT-Decision-Tree

Inference
====
One extension in our code is to use the proposed representaion to find model predictions.

The following logic will be used to traverse the tree: If a feature has been used to make a decision along
that path ($u_{rj}$) and it is used to decide to go left or right
($d^0_{rj}$ and $d^1_{rj}$ depending on the value of the feature, $f_r$)
and it is a leaf ($v_i$) then the decision is the same as $c_0$ or $c_1$

$\bigwedge_{r=1}^{K}\left[\left(\left(d_{r j}^{0} \wedge f_{r}\right) \vee \neg u_{r j}\right) \vee\left(\left(d_{r j}^{1} \wedge \neg f_{r}\right) \vee \neg u_{r j}\right)\right] \wedge v_{j} \wedge c_{1 j} \leftrightarrow \hat{c}_{1 j}$

The same for $c_{0 j}$ and $\hat{c}_{0 j}$.

Demo
====
Use `demo` notebook to see an example of how to use the code.
