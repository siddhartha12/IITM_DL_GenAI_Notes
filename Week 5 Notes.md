# L5.1 - Optimization in DL Challenges
Major challenges
* Local Minima: Getting stuck 
* Saddle point: Getting stuck on a flat point that is a minimum on one side and max on the other
* Vanishing Gradients: Gradients becoming so small that learning effectvely stops

Saddle points are more common than minima in high dimensions
AT a zeor gradient point, the type of point depends on the hessian matrix (matrix of second derivatives)
* Local Minimum: All eigenvalues of the hessian are positive
* Saddle Point: Some eigenvalues are positive, and some are negative
In a high D space, its statistically much more likely to get a mix of positive and negative eigenvalues than for all of them to be positive

# L5.2 - Convex Optimization

optimization are nonconvex, why?
* Analytical tractability
* Algorithms design and testing
* Local properties

convex sets: where the line connecting any two points in the set lies entirely within the set

Linear combination of functions must be greater than the actual function

Properties of convex functions
* Local minima is global minima
* second derivative: if and only if its 2nd derivative is non negative 
	* above for a 1d functions
	* for multidimensional convex only if hessian matrix del^2(f(x)) is positive semidefinite for all x

constrained optimization: solns
* lagrangian
* penalties
* projections

# L5.3 - Grad Descent Algorithm
4 method:
* Match Gradient Descent - do a full pass over a dataset to get gradient information and update - takes long time, not very efficient but less noise
* Newtons method - uses 2nd order Hessian properties, but parameters and computational complexity explode
* Stochastic - do at every datapoint - high noise, but fast
* minibatch - a good middleground

# L5.4 - LR Scheduling
Why schedule?
* Magnitude - is too large optimizer diverges, too small converges too slowly
* Decay - As we approach a minimum, need to take smaller stps
* Warmup, parameters are random, large steps harmful, warmup with small but increase great

Common Schedule
* Piecewise constant - eg LV by 10x over every 30 epochs
* Exponential decay - neta_new = neta * exp^-lambda * t
* Polynomial decay - alpha
* Inverse Time Decay - simply scheduler that decreases learning rate inverse 
* Cosine annealing

Problem: Oscillations in GD when unequal parameters

Solution - 1: Momentum
* can use a velocity vector which is exponentially weighted moving average of past gradients

## Adagrad:
Problem: sparse features
* Common words exist, their vectors receive frequent updates, decaying rate appropriate for them
* rare words exist, their embedding receive very few updates, by the time more comes, global learning rate may have decayed preventing them from learning
Challenge: per parameter learning rates

Idea: Adagrad adapts the learning rate for each parameter based on the history of its gradients, it does this by dividing the global learning rate by the square root of the sum of all historical squared gradients for that parameter

RMSProp
Uses an exponentially decaying average of past sequred gradients to dampen oscillations and adapt learning rates

AdaDelta 
uses del(x) instead of differentiation

