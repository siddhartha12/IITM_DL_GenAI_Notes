# L6.1 - Introduction to Generative Models
A generative model is a type of statistical model that is capable of learning the underlying distribution of a dataset in order to generate new synthetic data samples that are similar to original data

Key features:
* core idea is to understand the process by which the data originated
* models the joint probability ditribution of features and labels
* can be used to generate new data samples that are not present in the original dataset
* Often learn a compressed and meaningful latent space representation of data

Capabilities
* Data Generation - create new realistic samples
* Understanding data - revela latent structures and dependencies within the data
* missing data imputation - fill in gaps in incomplete data
* anomaly detection - identify smaples that deviate from the learned distribution
* representation learning - discover meaningful low dimensional representations of the data

We take example of an explicit generative model on a 2 * 2 pixel grid, and we predict the rest of them based on the possibility of the combined probabilities, the events are not independent,  but once it gets bigger, the problem becomes impossible to solve

How model models address the bottleneck:
* Modern Generative models (GAN, VAE, diffusion models) do not explicitly learn or store the full joint probability table
* Instead they learn a function that can approximate this distribution or generate samples from it
* For in-painting, they use these learned functions to:
	* Sample iteratively: Start with random values for missing pixels and refine them
	* Direct predict: A neural network is trained to directly predict the missing pixels given in the context
	* Latent space manipulation: For VAEs and GANs, in-painting can involve encoding the partial image into a latent space, manipulating the latent representation and decoding it
* These methods avoid the explicit enumeration of all possible given pixel combinations, making in painting feasible for high resolution images

Generative vs discriminative classifiers
* Generative classifier: Naive-Bayes, Latent Dirichlet Allocation
Generative classifiers can be more robust to missing data and provide richer information but often require larger datasets

# L6.2 - Introduction to Latent space
Latent space:
* Lower dimensional representation that captures the essential underlying structure of high dimensional data

High dimensional data
* complex nonisy
* redundant features
* eg: 256  * 256  image

latent space
* Simplified, meaningful
* Captures core concepts
* A 50 dim vector representing

Example-1: Principal Component Analysis
* Core Idea: Find Orthogonal axes
* Latent space: Subspace spanned by first k principal components
* Application: Dimensionality reduction, data visualization, "Eigenfaces" algorithm used PCA to represent 

Key pre deep learning models
* Factor Analysis
	* Idea: Assumer observed data is a linear combination of unobserved factors plus noise, tried to find causal structure
	* Application: Psychology, finance (eg: finding market setiment)
* Latent Dirichlet Allocation 
	* Idea: A generative model for text, eaach document is a mix of topics, and each topic is a distribution over words
	* Latent space: Space of topics
	* Application: Topic modeling in large text corpora
* Matrix Factorization (SVD)
	* Decomposes a large ratings into smaller matrices of latent factors

## Latent Space: discovery and inference
Take a high dmensionsal data point x and find its low dimensional representation z in the latent space

Data(x) -> Encoder -> Latent Space(z)

Problem: Latent space was often unstructured, picking a random point z and decoding it would usually produce garbage not a valid new point, focus was on inference, not generation

## Latnet space: sampling from noise
new process:
Take a random vector of noise z from simple distribution and use a trained decode to generate a new data point x

Noise(z) -> decode/generator -> New Data (x_hat)

This was enabled by Variational Autoencoders (VAE) and Generative Adversarial Networks (GANs)

Deep generative models: Latent Space Modelling
* For high dimensional data, we cannot explicitly store P(x)
* Instead, these models learn a mapping function from a simple low dimensional latent space ( usually gaussian noise ) to complex data space
* The "Noise vector" (z in GAN/VAE) is a point sampled from this simple latent distribution
* the generator/decoder then transforms this noise vector into a coherent, high dimensional output (eg; an image)
* The "noise" is the source of variety. Different z vectors map to different generated outputs, allowing the model to produce diverse samples from the learned data manifold

## Toy vs Deep Models
toy - Rolling a loaded die loading of P(x) is in the roll
Deep model - You pick a random set of dna instructions (Z) and then an organism grows from it (The generator), different DNA leads to differnt organisms

Need for latent space/noise input
* Real images exist on a very low dimensional manifold within the vast pixel space. MOst random pixel configurations are just noise
* Problem: How do we randomly "find" a point on this manifold without explicitly mapping out the entire high dimensional probability
* Solution: Instead of trying to directly sample from P(X) in pixel space, let's learn a mapping from simpler, continuous distribution (latent noise z) to the data manifold


Intuition:
* Imagine the "space of all possible images", only a fraction are meaningful
* We can't pick random pixels and get a cat
* But we can pick a random point in a simpler, smaller "idea space" (Latent space Z)
* We then train a complex function (Generator/Decoder) to transform that random "idea" into a corresponding meaningful image
* Each unique z vectors acts as a unique seed or recipe for a specific image, ensuring variety while staying on the data manifold

Transition
* This concept was formalized with models like Autoencoders and extended to Variational Autoencoders
* GANs in 2014 provided an alternate way to leanr this
* diffusion models emerged recently

# L6.3 - Types of Generative Models
## Explicit density models
* define and learn a function for p(x)
* Allow for direct evaluation of the data likelihood p(x)

a) Tractable Density
* Density p(x) can be computed exactly
	* autoregressive models - model joint distributions as a product of conditionals, generate pixel by pixel
	* normalizing flows - transform simple distribution into a complex one via invertible functions

b) approximate desnity
* Density p(x) is intractible, so we optimize an approximation
* Variational Autoencoders: Maximize a lower bound on the likelihood ( ELBO )
* Diffusion / Score-based models - systematically add noise to data, then learn the reverse the process 

## Implicit density models
* Learn a process to sample from p(x) without ever defining the function itself
* Learn to generate data without ever defining the probability function p(x)
### GANs
The dominant family in the category
* Two player game between a generator and discriminator
* Generator: Creates fake data from random noise, tries to fool the discriminator
* Discriminator: Tries to distinguish real data from fake data
* Through this adversarial process, the generator learn the true data distribution implicitly
Examples: DCGan, StyleGAN


# L6.4 - Intro to Adversarial Networks
make neural networks compete against each other, composed of two neural networks
* generator - artist
* disctiminator - Art Critic

## Discriminator
its goal is to become an expert at distinguishing real data from fake data, it is trained to maximize the probability of making a correct classification

## Generator 
Create data so realistic that the discriminator cannot tell it's fake, it wants to fool the discriminator. Objective is to minimize the second term of discriminators objective

equivalent to training g to maximine log(D(z)) pushing discriminators output towards fake images to 1

## Combined minmax objective
full training objective pits the generator against the discriminator
min

![[Pasted image 20251220125403.png]]

Ideal output is to find a nash equilibrium

DCGAn was a stop forward
* Replace pooling layers
* use batch normalization
* remove fully connected layers
* activation functions
	* use rely except for output which should use tanh
	* use leakyrelu in discriminator

StyleGAN - by nvidia
incorporated style transfer techniques
* Mapping network
	* MLP maps input latent code into intermediate latent space w
	* disentangles latent space and reduces correlations
	* from w, multiple style vectors are produced via affine transformations
* Synthesis network
	* Responsible gneerating the image starting from a learned constant input
	* adaIN (adaptive initialization)

Training gan is challening
* Mode Collapse
	* output becomes less diverse
* non convergence
	* model parameters may oscilllate, become unstable, never converge
* Vanishing Gradients:
	* if discriminator gets too good too quickly
* Hyperparameters sensitivity
	* very sensitive to choice of hyperparameters

WGAn - wasserstein distance, pushes for diversity
Minibatch discrimination = disc sees batch of samples
progressive growth of GANs


