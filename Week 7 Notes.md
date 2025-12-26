# L7.1 - Introduction to Sequence Modelling
Intro to RNNs -> the precursors to modern GenAI
* Infer next term in a sequence

What is Sequential Data?
* Sequential data is a type of data where the order of the data points is significant. The sequence itself contains important information
* Each data point is dependent on the previous ones, This creates a temporal or spatial (when we use for vision) dependencies within the data (Cannot make iid)
Key characteristics
* Ordered: Data points have a specific position or index in a sentence
* Variable length: Sequences can have different lengths (eg. different sentence lengths)
* Dependencies: Values can depend on other values at different positions in the sequence
	* can be anywhere in the sequence, very far away also

How is it processed?
* Make the sentence subparts/atomic parts -> tokenization 
* each x is an rnn is an atomic RNN

Key concept
* Analysing this data involves understanding patterns, trends and contexts over time or space

# L7.2 - Mathematical Modelling of Sequences
Mathematical formulation:
* A sequence model aims to estimate the joint probability of an entire sequence of observations
* For a sequence x1, ... ,xn, model estimates P(x1, ... ,xn) - joint distribution
	* we are estimating the joint probability
* allows us to assess how plausible a sequence is, or to generate new similar sequences

Language modells
* When a sequence model is applied to text data, it is called a language model. Its goal is to estimate the probability of a sequence of words or characters

Definition
* Autoregressive models predict future values in a sequence based on past values
* This is achieved by factoring the joint probability of the sequence using the chain rule of probability:
	* `P(x1, ... , xn) = P(x1)pi(T, t=2)[P(xt|x1, ... ,xt-1)]
* In words
	* The probability of a sequence is the product of the conditional probabilities of each element, given the elements that came before it
* Basically we harness the chain rule - `P(x1, x2, x3) = p(x1)p(x2|x1)p(x3|x1, x2)`
* Why conditionals instead of joints? easier to model, size increases as you go

Markov Assumption:
* Handling very long sequences can be computational expensive because the model would need to consider the entire history
* To simplify this, we can use the markov assumption

First order Markov Assumption
* assumes that the current state xt only depends on the immediately preceding state xt-1
* probability simplifies to
	* `P(x1, ... , xt) = P(x1)pi(T, t=2)[P(xt, xt-1)]
* Also known as a bigram model in context of language processing

This can be extended to consider a fixed length window of previous states (eg: trigrams, etc)

Linear regression for sequence modelling
* follows an n gram markovian assumption
* regressino over last n points
* limitations
	* fail to capture complex non linear relationships
		* interaction between different past events
		* sudden shocks or spikes in the data
	* Lack of long term memory
		* by conditioning only a fixed window, the model is completely blind to any information that occured before that window. This is a major drawback when long range dependencies are important
	* Error Accumulation
		* for k step ahead predictions, the model must use its own previous predictions as input, creates a feedback loop where errors compounds over time

Why MLPs and CNNs are not ideal for sequences
* Fixed size vectors: Require inputs of a fixed length, forces to either truncate long sequences or pad short ones, both of which lead to loss of information
* no parameter sharing across time: An MLP learns separate weights for features at each position. It cannot generalize a pattern it learn at position 'i' to position 'j'. A CNN shares parameters spatially, but isn't designed for capturing temporal dependencies of arbitrary length
* no memory of context: 

# L7.3 - Languages modelling

finding a join probability of any sequence of words that occur, are natural language processing methods, we try to pose a language as a sequence problem.

Goal: Make neural networks understand words, they only understand numbers, first and most critical task is to convert raw text corpus into a numerical format that a model can process

Core idea: we will build a dictionary that maps every unique word or character to a unique integer index

Steps for convert to language model
1. Tokenization
	* Character tokens: split text into individual characters, this results in a small vocabulary but very long sequences
	* Word tokens: Split the text by spaces and punctuation. more common and results in a larger vocabulary but shorter meaningful sequences

2. Building the vocabulary: 
	* Corpus: total all the words that exist
	* Unique word tokens: { dictionary }
	* vocabulary: we create a mapping. It's also crucial to add a special token for any words that might appear later but are not in our current vocabulary
3. Creating final numerical sequence

Preparing data for neural models:
* To train a neural language model, we need to create feature-label pairs from out long text sequence, there are two common strategies - random sampling and sequential partitioning.
* This process is not truly supervised or unsupervised inherently, we frame it as supervised

1. Random Sampling
	* to introduce randomness, a random number of tokens are discarded from the start of the corpus at the beginning of each epoch
	* The rest of the corpus is then partitioned into multiple non overlapping subsequences of a fixed length t
	* for any given input subsequence X = (x1, ... , xn), the target sequence Y is the same sequence shiften by one token to the left
	* Each subsequence created this way as an independent example for training
2. Sequential Partioning
	* ![[Pasted image 20251225214749.png]]

How good is a language model?
* **Perplexity(PPL)** is the standard metric for evaluating language models, measures how "surprised" or "perplexed" a model is by a sequence of a text
* Lower is better

Mathematical definition of perplexity
* Exponentiation of average negative log-likelihood of the sequence

Intuition
* A perplexity of k means that, on average, the model is as uncertain about the next word as if it had to choose uniformly from k different words
	* if PPL = 100, model confused as someone randomly guessing from 100 words

# L7.4 - Recurrent Neural Networks
* General Sequence models - Variable Length Data
* Predict the next term in the sequence

Intro to RNNs
* markov models and n-grams have a fixed memory, they only look at the last n-1 tokens
* To remember more, the number of model parameters grows exponentially, making them impractical for capturing long range dependencies

## RNN:
Instead of conditioning the raw sequence of past tokens, an RNN maintains a compact summary of the entire history in a hidden state, ht
* Model approximates the true probability with the hidden state
	* `P(xt | x1, ... ,xt-1) = P(xt | ht-1)
	* ht-1 -> single vector representing the entire sequence until now
	* roll the sequence until ht
	* the new information is zipped into ht-1 for the current ht
* This state is updated at each time step using the current input xt and the previous state ht-1

core idea:
* An RNN processed sequences by iterating through its elements one by one
* it maintains a hidden state (or memory) that captures information about what it has seen so far
* The output at a given time step depends not only on current input but also on the hidden state from the previous time step

RNN Basics
* RNN Process sequential data by maintaining a hidden state that captures information seen so far
* At each time step, the network
	* takes a new input xt
	* updates its hidden state, ht
	* produces an output yt
	* ![[Pasted image 20251226115734.png]]
* It is an autoregressive model

Weights of a Single Recurrent Neuron: 
* A single recurrent neuron has three distinct sets of weights that manage its state and output:
	* w_xh: weight for current input x_t
	* w_hh: weight for its own previous hidden state, h_t-1. This is the recurrent or memory weight
	* w_hy: weight for producing the final output, y_t from its current hidden state, h_t
* For one neuron, the computation is:
	* `h_t = tanh(w_xh * x(t) + w_hh * h_(t-1) + b_h)
	* `y_t = w_hy * h_(t) + b_y`

Backpropagation - dependencies through time
* The hidden state h_t at any time step t depends on the state from the previous step, h_t-1
* This creates a long computational graph where the loss at the end of a sequence depends on computations from every single time step

Backpropagation through time (BPTT):
* BPTT is simply the application of standard backpropagation algorithm to the unrolled computational graph of an RNN, not a new algorithm but a new for the process
* ![[Pasted image 20251226121920.png]]

Chain rule in BPTT
* The gradient of the final loss L with respect to the hidden state at time t, del_L/del_h_t has two components
	* Gradient fro the output at the current time step y_t
	* The gradient from the enxt hidden state h_t+1, which was computed using h_t

The problem with long sequences
* For a sequence of length T, calculating the gradient for the initial weights involves a chain rule product of T matrices
	* ![[Pasted image 20251226122246.png]]
* This long product is the source of major numerical instability
* This depends on the eigenvalues of w
	* if eigenvalues of w is > 1, there is the problem of exploding gradients
		* become NaN or infinity
		* effect: model weights are updated by huge gradients
		* solution: gradient clipping: scaled down if the norm of the gradient exceeds a certain threshold
	* if < 1, then diminishing or vanishing gradients
		* gradients for early time steps becomes nearly zero
		* model cannot learn long range dependencies
		* sol: more complex RNN architectures

Solution: Truncated BPTT
* Instead of Backpropagating through the entire sequence, we detach the gradient history after a fixed number of steps (T)
	* The model still propagates its hidden state forward through the whole sequence, maintaining its long term memory
	* However, during the backward pass, the gradient calculation is cut off after T steps