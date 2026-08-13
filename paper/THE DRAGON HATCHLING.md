## Contents
- 1 Introduction
  - 1.1 Motivation
    - Reconciling Reasoning Function of the Brain with Language Models.
    - Towards scale-free foreseeable AI.
    - Introducing Axiomatic AI.
  - 1.2 Intuition of results: combining modus ponens reasoning with Hebbian learning
  - 1.3 Contribution of this work
    - Language Models as Local Graph Dynamics.
      - Claim 1 (informal overview of theoretical results for BDH) .
    - A tensor-friendly case of BDH: the BDH-GPU architecture.
      - Claim 2 (informal overview of theoretical results for BDH-GPU) .
      - Empirical Finding 1 (informal overview of empirical results of BDH-GPU) .
    - The bridge between the Transformer and Brain models.
    - Implications for learning dynamics of natural lifelong inference systems.
  - 1.4 Notation
    - State-space models.
    - Models as programs.
    - Graphs and their dynamical systems interpretation.
- 2 BDH: a language model architecture given by local distributed graph dynamics
  - 2.1 Formalism for local graph-based language models
    - Introduction to distributed graph systems.
    - Programmable rulesets and the interaction kernel.
      - Definition 1 (Interaction kernel, general form) .
    - Restricting the interaction kernel to spiking signals and graph systems.
    - Definition of the edge-reweighting kernel.
      - Definition 2 (edge-reweighting kernel) .
  - 2.2 Definition of BDH as a local edge-reweighting process (equations of reasoning)
    - Definition 3 .
    - Inference dynamics of BDH.
      - Observation 1 .
      - Proof.
    - Notes on training.
  - 2.3 Interpretation of attention as a micro-inductive bias of reasoning
  - 2.4 Interpretation of BDH as an oscillator network toy-model
    - Definition of the toy-model.
    - Effects captured by the toy-model.
  - 2.5 Expressing BDH using brain models
    - Observation 2 .
    - Empirical Finding 2 .
- 3 BDH-GPU: a tensor-friendly version of the BDH architecture
  - 3.1 Notation for BDH-GPU
    - Nonlinearities: ReLU and LayerNorm.
    - Activation vectors and parameter matrices.
  - 3.2 Definition of BDH-GPU as a state-space system
    - Definition 4 (inference dynamics of BDH-GPU) .
    - BDH-GPU as a language model.
    - State-space representation.
  - 3.3 Interpretation of BDH-GPU as a local interacting particle system
    - Observation 3 (local particle interaction ‘by mean-field’) .
  - 3.4 Expressing BDH-GPU using BDH: preserving parameter and state size
    - Observation 4 (BDH-Normfree is a special case of the BDH graph model) .
    - 3.4.1 Expressing matrices D x , D y , E {D_{x}},{D_{y}},E as graphs G x , G y {G_{x}},{G_{y}}
      - Observation 5 .
      - Claim 3 .
      - Proof.
    - 3.4.2 Expressing BDH-GPU attention on graphs: sparsification and trainability of G s {G_{s}}
      - Claim 4 .
      - Proof.
- 4 Implementation and scaling laws
  - 4.1 Implementation characteristics of BDH-GPU
    - Model scaling in neuron dimension n n .
    - Layers and heads.
    - Linear attention with state aligned to neurons.
    - Sparse positive activation.
  - 4.2 Comparison of BDH-GPU to GPT2-like Transformers
    - Architecture differences.
    - Transformer-like scaling laws.
    - FLOPS counts.
  - 4.3 Comparison of BDH-GPU to other sequence processing architectures
    - Transformers with Linear Attention.
    - Other types of Transformers.
    - Networks with sparse activation.
    - Oscillatory SSM’s.
- 5 Analysis: emergence of modularity and scale-free structure
  - 5.1 Background: modularity and scale-free property of systems
    - Importance of modularity for information propagation.
    - Scale-free property.
  - 5.2 BDH-GPU feed-forward network with the ‘ReLU-lowrank’ block
    - Definition of ReLU-lowrank.
    - Expressiveness of ReLU-lowrank in BDH-GPU and MLP in the Transformer.
  - 5.3 ReLU-lowrank as a signal propagation dynamics
    - Error of low-rank approximation (without ReLU).
    - Expressiveness of ReLU-lowrank for Markov chain propagation.
      - Claim 5 (propagating a Markov chain) .
      - Proof (sketch).
    - Propagation and reinforcement of signal.
  - 5.4 Modularity in BDH-GPU signal propagation
    - Claim 6 (selective neuron activation) .
    - Proof.
    - Observation 6 (in-cluster signal reinforcement) .
    - Supermodularity on input perturbation.
  - 5.5 Empirical findings: parameter distribution in ReLU-lowrank matrix products
    - Choice of prior of matrix parameter distributions.
      - Hypothesis 1 (right tail contains signal) .
    - Experimental setup.
    - Findings.
      - Empirical Finding 3 .
- 6 Analysis: linear attention, sparse positive activation, and monosemanticity
  - 6.1 Macro-expressiveness of attention in BDH-GPU
    - Basic properties of BDH-GPU attention.
    - State capacity vs. distinction capacity.
    - Expressiveness of linear attention in dimension n n .
      - Claim 7 (informal statement) .
    - Preparation of positive keys for Linear Attention.
      - Using LSH to move key vectors into the positive orthant.
      - Observation 7 .
      - Attention in the positive concept space of language and reasoning.
      - Observation 8 .
    - Natural support for long context.
  - 6.2 Micro-interpretation of attention in BDH-GPU
  - 6.3 Empirical findings: monosemantic synapses
  - 6.4 Empirical findings: sparse neuron activations
- 7 Playing with the Hatchling
  - 7.1 Model merging: concatenating two models
  - 7.2 Training without backpropagation through time
- 8 Conclusions
  - 8.1 Takeaways for model engineering
  - 8.2 Implications for brain science
    - How this work helps with axiomatization of learning theory in the brain.
  - 8.3 Societal impact
  - Acknowledgments
  - Author contributions
- Appendix A Connection between generalization of reasoning and computational expressiveness
  - Observation 9 .
- Appendix B Further description of experiments
  - B.1 Language translation task
  - B.2 BDH Scaling Experimental Details
  - B.3 BDH Monosemantic Synapse Experiment Details
  - B.4 BDH Merging Experiment Details
- Appendix C Omitted formal claims and proofs
  - C.1 Proof of Observation 1
    - Proof.
  - C.2 Formal statement of Claim 7 (linear attention)
    - Claim 8 .
    - Proof (sketch).
  - C.3 Proof of Claim 3
    - Proof.
    - Considerations of building linear circuits.
  - C.4 Formal statement of Claim 4
    - Claim 9 .
    - Proof.
- Appendix D Desirable properties of a local graph dynamics for language models
  - Hypothesis 2 .
  - [No Easy Simulation] ⋄ \diamond Computational models have irreducible dynamics.
  - [Particles Talk] ⋄ \diamond Latent concept spaces arise from outcomes of particle-particle interactions.
  - [Attention Deforms Pairwise Connections] ⋄ \diamond The interaction process X ⁡ ( i ) , Y ⁡ ( j ) → σ ⁡ ( i , j ) X(i),Y(j)\to\sigma(i,j) describes attention.
  - [Time Dictates Structure] ⋄ \diamond Inputs to reasoning problems are sequential, not graph-based.
- Appendix E BDH-GPU PyTorch code listing

## Abstract

Abstract The relationship between computing systems and the brain has served as motivation for pioneering theoreticians since John von Neumann and Alan Turing.
Uniform, scale-free biological networks, such as the brain, have powerful properties, including generalizing over time, which is the main barrier for Machine Learning on the path to Universal Reasoning Models. We introduce ‘Dragon Hatchling’ (BDH), a new Large Language Model architecture based on a scale-free biologically inspired network of n n locally-interacting neuron particles. BDH couples strong theoretical foundations and inherent interpretability without sacrificing Transformer-like performance. BDH is a practical, performant state-of-the-art
attention-based state space sequence learning architecture.
In addition to being a graph model, BDH admits a GPU-friendly formulation. It exhibits Transformer-like scaling laws: we find empirically that BDH rivals GPT2-architecture Transformer performance on language and translation tasks, at the same number of parameters (10M to 1B), for the same training data. BDH provides theoretical foundations for understanding model behavior in the limit of large size and reasoning time.
Our results, formalized as a chain of reductions of expressiveness in the framework of computational Complexity Theory and Distributed Computing, and combined with findings on the BDH model, show a macro-to-micro correspondence of function between the general attention mechanisms in state-of-the-art Language Models, and attention mechanisms observed in the brain. These attention mechanisms formally converge as closed-form local graph dynamics at neurons and synapses: “the equations of reasoning”. BDH can be represented as a brain model. It contains n n neurons, organized as an excitatory circuit and an inhibitory circuit with integrate-and-fire thresholding of input signals at neurons. The working memory of BDH during inference entirely relies on synaptic plasticity with Hebbian learning using spiking neurons, at potentiation scales of minutes for the brain (up to hundreds of tokens). We confirm empirically that specific, individual synapses strengthen connection whenever BDH hears or reasons about a specific concept while processing language inputs. The neuron interaction network of BDH is a graph of high modularity with heavy-tailed degree distribution. The BDH model is biologically plausible, explaining one possible mechanism which human neurons could use to achieve speech. BDH is designed for interpretability. Activation vectors of BDH are sparse and positive. We demonstrate monosemanticity in BDH on language tasks, including representation of concept abstractions, which happens even for small models, below 100M-parameter scale. Interpretability of state, which goes beyond interpretability of neurons and model parameters, is an inherent feature of the BDH architecture. We believe BDH opens the door to a new theory of “Thermodynamic Limit” behavior for language and reasoning models, with the ultimate goal of Probably Approximately Correct (PAC)-like bounds for generalization of reasoning over time. ⊳ \rhd Technical blog entry: https://pathway.com/research/bdh . ⊳ \rhd Code listings: https://github.com/pathwaycom/bdh .

## 1 Introduction

Long reasoning and long context inference pose a severe challenge of generalization across scales of time. From vibe coding to market research, users of Language Models and agentic systems are increasingly relying on defining tasks through informal prompts, which the language model is expected to follow over long sequences of actions or decisions, like a reasonable human actor would. Implicitly, most users expect machines to follow the generalization patterns of human reasoning, i.e., to generalize reasoning in the same way as humans do. The complexity of tasks attempted in this way has gone from the equivalent of hours of human work for a single prompt, to weeks . However, experimental evidence suggests that the Transformer and other state-of-the-art architectures do not systematically generalize chain-of-thought (CoT) reasoning to scenarios longer than the ones seen during training .

Chain-of-Thought reasoning models can be considered through the lens of computational complexity theory. For a Language Model to generalize human reasoning on a given class of tasks, we expect this model to be able to emulate the corresponding reasoning function of the human brain efficiently.(^1^1 1 We provide a more formal explanation of this point in Appendix [A](#A1).)
While the Transformer with Chain-of-Thought is Turing-complete and can efficiently emulate certain restricted classes of formal languages , this does not in itself provide a satisfactory answer as to how it emulates human reasoning. The human brain is an extremely complex graph-based distributed computing system with $n\approx 8\cdot 10^{10}$ neurons, and $m>10^{14}$ neuron connections (synapses), of which a certain percentage is actively used. The direct simulation of such a distributed system by a Language Model through generic Turing-machine reductions would require billions of CoT tokens of the Language Model to represent a single step of reasoning in the brain. So, do Transformer-like models actually relate to brain function?

Such a relationship should follow more closely from a tighter, more direct simulation. Finding such a connection between Language Models and human brain function has, so far, proved elusive. Indeed, when comparing a tensor-based Language Model based on feed-forward network blocks and attention, to a uniform, scale-free graph-based distributed system, such as the brain, the two may, at first glance, appear very dissimilar.

This apparent dissimilarity of structure between Language Models and brain structure has been one of the main causes of concern in attempts to reconcile Computation and the Brain , as well as a cause of concern regarding the difficulty to foresee the behavior of autonomous AI systems.

In this paper, we show the link between the Transformer and Brain models.

### 1.1 Motivation

The development of Artificial Intelligence and the understanding of Neural Science have gone hand in hand since the 1940’s, both being efforts to understand the “mystery of intelligence”. The relationship between computing systems and the brain served as motivation for the pioneering theoreticians such as John von Neumann (), Alan Turing (), Goeff Hinton (), Warren McCulloch and Walter Pitts (), and Horace Barlow ().

Since then, milestones in Machine Learning around Artificial Neural Networks — using backpropagation with SGD , followed by Deep Learning , and the Attention mechanism  — have split the “mystery of how intelligence works” into two. First, we still have no clear explanation for the micro-to-macro correspondence of the reasoning function of the brain. Second, we do not understand the correspondence between the artificial and natural systems — notably, how effects observed in the brain (emergent network; sparse activations; oscillatory phenomena; unknown relationship to backpropagation mechanisms) map into those which appear in systems based on dense tensors, trained using gradient back-propagation over time.

##### Reconciling Reasoning Function of the Brain with Language Models.

There is a seemingly deep divide between state-of-the-art language models, like the Transformer, and natural distributed systems with local graph dynamics, like those of the brain. Specifically, for the brain, we do not understand how the reasoning function emerges from neuronal dynamics at the microscale. For the Transformer, the interpretation of function is given at the level of vectors, but not at the level of particle dynamics or a uniform distributed computing system.

Language and reasoning are the key areas of higher-order brain function for which we do not yet have a complete understanding. Many other areas of brain function have been explained through analogies to Machine Learning architectures.
For example, the visual cortex is becoming well-understood, especially in its peripheral layers, and the observed inference dynamics are shown to have a correspondence to known Deep Learning architectures . The use of sparse coding by the brain was considered in the context of processing visual cues , as well as for the olfactory systems . By contrast, higher-order cognitive functions of the association cortex of the human brain, such as language and reasoning, are among the least understood. A number of models provide partial explanations and have been verified at small scales. Some of the first attempts include explaining context-dependent computation in the prefrontal cortex using population dynamics of an RNN . Later approaches include the Tolman-Eichenbaum Machine , as well as a number of more recent works . One of the main stumbling blocks concerns going from spiking activation patterns at neurons, and localized attention effects at synapses, to a higher-order function, serving a reasoning purpose, efficiently organized at a scale of millions to billions of neurons.

Conversely, for Language Models architectures such as the Transformer, we miss a compact micro-interpretation as a distributed system. The expressiveness of the Transformer has been approximated using approaches from centralized computing and Complexity Theory, rather than from distributed systems. In the centralized perspective, a language model can be seen as a transformation function from inputs into outputs. The computational expressiveness of the Transformer architecture may then be approximated through frameworks based on RASP, such as RASP-L  or C-RASP . RASP-L provides a very convenient heuristic for estimating Transformer expressiveness at the rather coarse level of vector operations, while C-RASP provides a more specialized lower-bound on expressiveness, capturing a class of formulas of temporal counting logic. Both frameworks have been used to suggest theoretical models of task length generalization.
This type of expressiveness techniques, however, do not lead to a uniform asymptotic model for the behavior of the Transformer, whether in GPT2 architecture or simplified. The scaling of the Transformer in its different dimensions, and the need to manipulate context length, complicate this goal.

The lack of such a uniform model also makes it hard to compare the capabilities of the Transformer to the capabilities of the brain at the level of correspondence of structure. Generally, the temporal behavior of a state-space system is reflected in its structure(^2^2 2 For a linear system, temporal behavior would be a direct consequence of the spectral properties of the system. The considered systems dynamics are not linear.).

Understanding whether it is possible to show alignment of the temporal behavior of two systems, which do not display any structural correspondence, and without a clear idea of how the weight tensors and state representation of one system ‘embed’ into the graph structure and state representation of the other system, is an awkward task.

This brings us naturally to our motivational objective: Can we create Machine Learning models which are closer to the desirable properties of natural (human) reasoning systems, and which exhibit the same types of limit and scaling behavior as such natural systems?

##### Towards scale-free foreseeable AI.

Ensuring correct scaling behavior of inference over time is of paramount importance for the deployment of AI whose reasoning or actions are not subject to strict human supervision. Most reasoning models and AI agentic systems admit limit objects (i.e., extensions to infinite time and infinite size) which are Turing-complete (cf. e.g. ). This means that they should be treated like computer programs — and should be approached by the users with the same standards of care, as a computer program of unknown origin and unknown purpose.

An AI model can malfunction when allowed to run for a long time autonomously, i.e., without human validation of actions and reasoning outcomes. The most painful of all consequences, perhaps, is the concept of a failed generalization of reasoning (a malfunction with respect to the original task objective) over time, leading to a grotesque effect known as the “Paperclip Factory” .

Can the risk of such unsuccessful generalization be bounded?

There are at least two scenarios in which a black-box model $M$ cannot be considered to have undergone previous empirical validation, and consequently cannot be used in higher-risk autonomous AI use cases.

- 1.
Length-generalization scenario: Model $M$ is expected to act autonomously on a task which is longer than tasks forming part of its validation set.
- 2.
Model scaling scenario: Model $M$ is not exactly the same closed system as the one which was tested during validation. For example, suppose that models $M_{1}$ and $M_{2}$ were tested individually on smaller tasks, and let $M$ be an agentic system composed of instances of $M_{1}$ and $M_{2}$ which communicate with exchange messages with each other during inference.

A natural way of avoiding both difficulties consists in studying systems which are scale-free with respect to size and time, and admit a form of uniform “thermodynamic limit” behavior. The limit behavior of computational systems at criticality naturally connects the size of the system with the probable duration of its operation, with the connection usually taking polynomial form (cf. e.g.  for examples of graph-based interacting particle systems for which rigorous results have been obtained in this direction). Consider a model $M_{n}$ with architecture $\mathcal{A}$, parameterized by its size $n$ (with the interpretation of the number of uniform particles), and sampled from some space of $n$-neuron models in architecture $\mathcal{A}$ in some space equipped with a probability measure, $M_{n}\sim\mathcal{P}_{\mathcal{A}}(n)$. Informally, if the limit object $\mathcal{P}_{\mathcal{A}}:=\lim_{n\to\infty}\mathcal{P}_{\mathcal{A}}(n)$ exists (under an appropriate, well-defined sense of uniformity of limit) then models $M_{n}$, for $n$ sufficiently large, will admit in the limit asymptotic properties, which can be used to characterize their behavior over time.

The existence of such a limit theory means that we can characterize, with bounded probability of error, the behavior of a family of large models, having $O(n)$ parameters, while relying on a theory which is independent of the specific structure and size of the specific model. In this way, the limit behavior of a system of a very large number of interacting uniform particles over time becomes (stochastically) foreseeable in the sense of its adherence to expected behavior, which can be extrapolated from observations at shorter time scales. Thus, small tests may be conceived in order to provide validation for a scale-free system at long time scales.

##### Introducing Axiomatic AI.

Axiomatic systems are those in which micro-foundations and the macro-description which arises from them are consistent and well-understood. The need for axiomatic understanding was highlighted by David Hilbert (), and has become the foundation in Statistical Physics (e.g. thermodynamics, fluid dynamics, spin glass theory), cellular mechanisms, Social Networks Science, and reconciliation of Microeconomics and Macroeconomics through a Network Economics perspective.

This paper brings a micro-foundational understanding to Language Model inference, to the mechanisms of in-context learning, and Chain-of-Thought reasoning dynamics.

The considerations in this work naturally support a shift of perspective from *Interpretable AI*, which gives an approximate understanding of what the model is doing now (without necessarily telling us what its current actions are going to lead to over longer time scales), to *Axiomatic AI*, where we also understand the micro-foundations of how the model can be expected to behave subsequently over time.

### 1.2 Intuition of results: combining modus ponens reasoning with Hebbian learning

In this section we provide the reader with some of the main intuitions behind this work which, we hope, will help to navigate the remaining, more formal parts of this paper with ease.

While there are many formal deductive systems in logic, they predominantly rely on the modus ponens inference rule. Applied to a rule-based reasoning system, it takes the following form:

If we know that the $i$-th fact is true, and our ruleset $\sigma$ indicates that the $i$-th fact implies the $j$-th fact, then we know that the $j$-th fact is true as well. In an approximate reasoning system, the strength of the rule $\sigma(i,j)$ indicates how the belief $X(i)$ of the system affects its belief $A(j)$. We could write:

$$ $X(i),\sigma(i,j)\xrightarrow{}A(j),$ (1) $$

to indicate that if $X(i)$ is a weighted belief, it contributes $X(i)\sigma(i,j)$ to the system’s belief $A(j)$.

Practical logical inference systems differ in strategies employed for rule selection, with the most advanced ones allowing direct manipulation of the ruleset, effectively resulting in a form of program evolution during inference(^3^3 3 The authors’ personal experience with writing efficient Prolog programs confirms that such direct ruleset management is often a necessary pragmatic evil, guiding the inference system in the right direction.). For an approximate reasoning system, such a heuristic could manipulate the strength of rules, modulating the impact of belief $X(i)$ on the system’s belief $A(j)$.

Hebbian learning , often presented as the mnemonic “*Neurons that fire together wire together*”, can be seen as a heuristic for ruleset manipulation. It postulates that synaptic connections are strengthened when the activity of one neuron, $Y(i)$, led to the firing of another neuron, $X(j)$. In the context of an adaptive, approximate inference system, the Hebbian heuristic means that if during the course of operation a fact $i$ contributed some evidence for $j$, the system increases the significance of the implication $\sigma(i,j)$. We could write this rule as:

$$ $Y(i),X(j)\xrightarrow{}\sigma(i,j),$ (2) $$

with the interpretation that co-presence (or a spike) of $Y(i)$ followed by $X(j)$ increases $\sigma(i,j)$ by $Y(i)X(j)$.

The relations ([1](#S1.E1)) and ([2](#S1.E2)), over a set of $n$ facts, may form the basis of a simple approximate reasoning system that adapts its operation to the problem at hand. Starting with some initial connections between facts, the system applies the rules to discover new facts, at the same time reweighting the ruleset in a way that strengthens the connections between the initial and derived facts. Effectively, should the system be rerun with the new ruleset, it would arrive at similar conclusions faster.

Suppose now that the reasoning system is equipped with two sets of rules: a fixed set $G$ and an evolving set $\sigma$. From a machine learning perspective, the fixed ruleset $G$ can be seen as model weights in Deep Learning terminology, learned using e.g. error backpropagation on a training set. On the other hand, the evolving ruleset can be seen as the temporal state of the reasoning system, sometimes called “fast weights” . Fast-weights systems have a favorable ratio of state size to parameter count. A system with $n$ facts has $m=O(n^{2})$ trainable parameters (expressed using one or more $n\times n$ matrices). A classical recurrent neural net, such as the LSTM , treats individual fact (neuron) activations as its state, thus maintaining only $O(n)$ state variables. On the other hand, the evolving set of fast-weights $\sigma$ has $m=O(n^{2})$ state entries. We believe this 1-1 ratio of trainable parameter to state size is important in designing practical reasoning systems and may justify the success of the Transformer and state-space sequence processing models.

Now, bearing in mind that the trainable parameters and state have comparable size $m$, we can adjust the ratio between this value $m$ and the size $n$ of the fact base. This will happen through a choice of sparsity for the $n\times n$ matrices carrying parameters and state, resulting in a specific relationship of the two values, $n\ll m\ll n^{2}$. In this way, our system gets a natural interpretation in terms of graphs on $n$ nodes and $m$ edges, with the graph edges tasked with their first two roles: carrying state, and, carrying trainable parameters. Finally, we will give our system an interpretation of a dynamical system with distributed (localized) dynamics, and we will task our edges with their third crucial role: mediating in communication between nodes of the system. In this way, through assimilation of edges to natural function in the brain, we will refer to the $m$ edges as *synapses* connecting a set of $n$ *neurons* into a distributed graph-based system.

In the following Section [2](#S2), we will introduce BDH, a reasoning system that formalizes and combines relations ([1](#S1.E1)) and ([2](#S1.E2)) with dynamics involving fixed rules. The BDH system:

- 1.
is a reasoning system, efficiently using the modus ponens reasoning rule with heuristic rule reweighting, based on ([1](#S1.E1)) and ([2](#S1.E2)),
- 2.
can be implemented with local graph dynamics, making it suitable for brain-like execution model, and amenable to a principled, axiomatic description,
- 3.
contains a set of fixed connections (parameters), and a set of dynamically adjusted connections ($\sigma$), which can be seen as its dynamic state updated with a Hebbian learning rule,
- 4.
admits as its special case BDH-GPU, a GPU-efficient reasoning model architecture, introduced in Section [3](#S3) and experimentally validated at scale in Section [7](#S7) in direct comparison to state-of-the-art GPT2-like Transformers.

### 1.3 Contribution of this work

The focus of this paper is in explaining the dynamics of the primary function of language and reasoning models: inference. We provide a description of a language model architecture which is directly comparable to the Transformer, and admits a clear and interpretable local interpretation of its inference dynamics as a programmable interacting particle system.

##### Language Models as Local Graph Dynamics.

In Section [2](#S2), we introduce a graph-based model architecture called *BDH*, where all model parameters are represented as topology and weights of the communication graph, and model state during inference is represented as edge-reweighting applied to this graph topology.

###### Claim 1 (informal overview of theoretical results for BDH) .

We introduce a state-space Machine Learning architecture called BDH, formed by a system of $n$ particles called neurons which communicate in a way governed by the weights and topology of the system graph, representing a “communication by wire” network.

- •
The inference dynamics of BDH, treated as a distributed system, can be represented as execution of local rulesets for $n$ particles with programmable interactions, with particles acting as nodes of the interaction graph and scalar state variables located on its edges (cf. Section [2.2](#S2.SS2)).
- •
The local kernel of BDH can be naturally expressed (emulated) by a graph-based Spiking Neural Network system capable of Hebbian learning dynamics, an Excitatory circuit, and an Inhibitory circuit on an $n$-neuron system described by a neuron interaction graph (cf. Section [2.5](#S2.SS5)).

In order to train BDH efficiently and analyze its performance, we restrict it, making this restriction the core of a GPU-friendly architecture called *BDH-GPU *. This restriction is obtained by treating the communication of the $n$ particles as proceeding through a mean-field (“radio network”), rather than a graph (“communication by wire”), cf. Fig. [3](#S3.F3) for an explanation of how the state-space equations of BDH-GPU are obtained from BDH.

This allows us to train a mathematically equivalent model, while localizing its state in short vectors at neurons, not at connections (synapses) of the system.

###### Claim 1 (informal overview of theoretical results for BDH) .

##### A tensor-friendly case of BDH: the BDH-GPU architecture.

The BDH-GPU architecture, like the Transformer, crucially relies on an attention mechanism, and is amenable to token-parallel training on GPU for next token prediction tasks. Unlike the Transformer, activation vectors of BDH-GPU appear in a very high dimension $n$, are positive by design, and turn out to be sparse.

###### Claim 2 (informal overview of theoretical results for BDH-GPU) .

We introduce a Machine Learning architecture called BDH-GPU, parameterized by a single (very large) scaling parameter $n$ and a second parameter $d$, $\log n<d\ll n$ ($d=256$ in practice), such that:

- •
A model in BDH-GPU $(n,d)$ has $(3+o(1))nd$ parameters, and admits a precise interpretation as a state-space system following the local dynamics of a $n$-particle system in an interaction field subject to equations of state ([8](#S3.E8)). This system is described by $O(d)$ parameters per particle, whose interaction field has mean field interpretation, which in a computational view corresponds to a particle communication network realized by means of “noisy radio broadcast”.
- •
BDH-GPU is a special case of BDH in the sense that, for any BDH-GPU model with $n$ particles, there exists a BDH model with $n$ particles with the same inference behavior and the same size $O(nd)$ of trainable parameters, with the two models being formally equivalent up to placement of Layer Norms (cf. Claims [3](#Thmclaim3) and [4](#Thmclaim4)).
- •
The BDH-GPU architecture relies on a combination of two blocks: a specific kind of *ReLU-lowrank* feed-forward network, and a *linear attention* mechanism, which both operate in the same neuron dimension $n$, using positive activation vectors.
- •
The mechanisms of BDH-GPU, considered at the macro-level of activation vectors in $R^{n}$, can be compared to those of the Transformer (cf. Section [6.1](#S6.SS1), Section [5.2](#S5.SS2)). This justifies the applicability of the frameworks of approximate macro-expressiveness, based on RASP  and designed for the Transformer, to BDH-GPU.
- •
The micro-interpretation of BDH-GPU mechanisms as neuron-neuron interaction dynamics: (1) explains mechanisms of in-cluster communication of neurons and the spontaneous emergence of graph structure with high Newman modularity in the neuron-neuron communication network (cf. Section [5](#S5)), and (2) provides a strict correspondence between the macro-mechanism of in-context inference based on attention and the local representation of state on individual neuron-neuron pairs (synapses) with state update dynamics based on sporadic updates to synaptic edge weight (cf. Section [6](#S6)).

The above results are complemented by empirical findings.

###### Empirical Finding 1 (informal overview of empirical results of BDH-GPU) .

BDH-GPU is represented as a tensor-based architecture and can be trained with standard back-propagation methods (cf. Section [3](#S3)).

- •
The BDH-GPU architecture is shown to follow scaling laws (parameters vs. loss) of optimized Transformers in the GPT architecture, at parameter scales between 10M to 1B, on all next token prediction tasks we tested, including tasks of language and translation reminiscent of those in the original benchmark set for the Transformer architecture (cf. Section [4.2](#S4.SS2)).
- •
An emergent network reflecting the associated BDH graph dynamics can be read out directly from the parameter matrices of a trained BDH-GPU model, showing emergence of graph structure (cf. Section [5.5](#S5.SS5)).
- •
The positive activations of BDH-GPU exhibit sparsity (at about 5% level) in the $y$ vectors of its state space dynamics, with sparsity levels reflecting the amount of activity being performed by BDH-GPU for a given token (cf. Section [6.2](#S6.SS2)).
- •
In-context state of BDH-GPU attention is shown to localize on the same synapses (neuron-neuron links) consistently across multiple prompts, allowing for some basic features, the interpretation of the current in-context state based on the reading of state of an individual synapse associated with such a feature (cf. Section [6.3](#S6.SS3)).

A more detailed discussion of the training approach is provided in Appendix [B.2](#A2.SS2), while the code listing for BDH-GPU is provided in Appendix [E](#A5). For the purposes of our experiments, we did not apply any specific training method which would be known to guide the system towards any of the observed emergent properties. (In particular, L1-regularization was disabled.) The observed emergent effects follow naturally from the design choices of the BDH and BDH-GPU architectures, and are largely attributable to the combination of: the choice of model dimensions with comparable model-to-state ratio, reliance on linear attention in high dimension, reliance on ReLU thresholds for ensuring that activation vectors are positive (trivially) and sparse (an effect empirically noted in ).

We also remark that the BDH-GPU architecture allows for the uniform asymptotic scaling of the model in one dimension, $n$. For example, a composition of models, obtained by concatenation, is also model in the same architecture, with a larger value of $n$ (cf. Section [7.1](#S7.SS1) for an empirical study of this effect for practical translation tasks).
Historically, a link has been established between infinitely wide feedforward networks and Gaussian Processes . BDH allows the study of limit behavior of reasoning models.

> *With BDH and BDH-GPU, we show that Language Models can be amenable to a particle-based interpretation. In fact, two micro-foundations — particle-based behavior and logic-programming behavior of a reasoning system — fuse together in these architectures.*

###### Claim 2 (informal overview of theoretical results for BDH-GPU) .

###### Empirical Finding 1 (informal overview of empirical results of BDH-GPU) .

##### The bridge between the Transformer and Brain models.

The inference dynamics of BDH and BDH-GPU act as a natural bridge between Transformer, and neuromorphic models of the brain and its subsystems. We illustrate this in Fig. [1](#S1.F1).

Figure: Figure 1: General overview of architectures and their relationships: the inference dynamics of BDH and BDH-GPU act as a natural bridge between Transformer and models of the brain. The two main inference mechanisms of a reasoning architecture, attention and the feed-forward network, are defined at a macro-level through tensor operations for the Transformer, and at the micro-level of neuron interactions through local graph dynamics for Brain models. The new BDH-GPU architecture is naturally defined both at the level of vectors and of particle dynamics of neurons and synapses, acting as a bridge between these two approaches. See also Table [3](#S8.T3) at the end of the paper for a more detailed comparison of architecture properties.

##### Implications for learning dynamics of natural lifelong inference systems.

A lifelong learning system progresses in time, performing extremely rapid inference, combined with several training mechanisms at different time scales.

In this work, we provide and validate at scale a plausible explanation of what the *predominant* dynamics of such a system could look like, taking the system from ‘split-second’ scale, to the scale of inference during ‘minutes’, considering the flow of time at the natural rate of thought and language for humans.

A complementary discussion of learning dynamics would aim to provide an explanation of how to take such a lifelong inference system from the scale of ‘minutes’ into even longer timescales. This would concern the slower transfer of “fast-weights”-like inference state to long-term memory, starting at the order of $10^{3}$—$10^{4}$ tokens, and taking into account feedback signals. In this work, we do not provide a direct answer as to how the brain actually handles this effect at longer timescales. However, a constructive way to resolve this problem seems to be less challenging, once the local inference dynamics of the brain are better understood (we come back to this in the Conclusions). The modeling approach provided in Section [2.5](#S2.SS5) is proposed as a suitable framework for such a study.

### 1.4 Notation

##### State-space models.

For describing inference dynamics of any system, we will use state-space notation, and consider a state-space system composed of two parts: a set of *model parameters* $M$ which does not change during inference, and a *state* $\sigma(t)$ which changes during inference. The model performs inference following state-space equation $\sigma(t+1):=\mathcal{A}(M,\sigma(t),a_{t})$, where $a_{t}$ is a possible external input to the system at time $t$ (such as a language token), $t=0,1,2,\ldots$, and $\mathcal{A}$ is referred to as the *architecture* $\mathcal{A}$ which drives its progress. During inference without external input, usually autoregressive inference, we will shorten this to $\sigma(t):=\mathcal{A}^{t}(M,\sigma_{0})$.

##### Models as programs.

In settings that are of interest to us (inference with combining multiple facts, reasoning), we opt for terminology from computing. $M$ has the interpretation of a computer program code, $\mathcal{A}$ has the interpretation of a computational machine architecture which runs it, and $\sigma$ has the interpretation of the variable state of the program. We will use the terms ‘model $M$’ and ‘program $M$’ interchangeably.

##### Graphs and their dynamical systems interpretation.

For a square matrix with non-negative coefficients, $H\in(\mathbb{R}^{+})^{n,n}$, $n\in\mathbb{N}$, we will consider two more equivalent representations. In one, we will treat $H$ as a graph defined on some nodeset $V$, with $V=|n|$. Formally, we can take $V=\{e_{1},\ldots,e_{n}\}$, where $e_{i}=(0,\ldots,0,1,0\ldots,0)\in\mathbb{R}^{n\times 1}$ with $1$ on the $i$-th position, forming an orthonormal basis. Non-zero entries of $H$ are referred to as *edges*. By an overloading of notation, we will write $H(i,j):={e_{j}}^{T}H{e_{i}}\geq 0$, to represent the node affinity function, or *edge weight*, from $i$ to $j$. We define the *edge set* $E(H):=\{(i,j)\in V\times V:H(i,j)>0\}$.

In discussions of graph-based model architectures, we will take the standard interpretation of graphs from a linear dynamical systems perspective, applied to positive vectors. When $v\in(\mathbb{R}^{+})^{n\times 1}$ is a non-negative vector, $Hv\in(\mathbb{R}^{+})^{n}$ has the interpretation of a linear transformation of $v$. If $H$ satisfies the condition of stochasticity (column-normalization to $1$), then $v\mapsto Hv$ is a Markov chain transition, with $\|Hv\|_{1}=\|v\|_{1}$. From a distributed systems perspective, transitions of stochastic matrices can be represented either through the direct simulation of (probabilities) of such a Markov chain, or described by the token dynamics of an extremely simple stochastic token distribution scheme in which a token located at node $e_{i}$ goes to node $e_{j}$ with probability $H(i,j)$. If $H$ is not stochastic, the operation $v\mapsto Hv$ additionally necessitates the suppression of a fraction of tokens, or the multiplication of tokens, at each step at each node, depending on the column-normalization of a given node.(^4^4 4 We provide a graph distributed systems interpretation only for dynamics on graphs with non-negative matrix entries (positive-weight edges). Negative-weight edges are hard to represent using natural local dynamics based on token distribution or spiking models.)

For two graphs $H_{1},H_{2}\in\mathbb{R}^{n\times n}$, the graph $H=H_{2}H_{1}$ is obtained through (linear algebraic) matrix multiplication, and in a distributed system, the corresponding transition $v\mapsto Hv$ is obtained with two steps of token dynamics, one following graph $H_{1}$, the next following graph $H_{2}$.

Representing $m$ edge-weights of a sparse $n$-node graph with $b$ bits of numerical precision per parameter is possible with $O(m(b+\log n))$ bits of information, which corresponds to $O(m(1+\frac{\log n}{b}))$ parameters. For the sake of simplicity, we will assume in asymptotics that the second term of the sum does not dominate (i.e., $\log n=O(b)$), and so we simply say that we represent the graph with $O(m)$ parameters.

## 2 BDH: a language model architecture given by local distributed graph dynamics

### 2.1 Formalism for local graph-based language models

We consider model architectures $\mathcal{A}$ which correspond to models of graph-based distributed computing (cf. ). A specific model $M$ in architecture $\mathcal{A}$ corresponds to the weights and topology of the communication graph or graphs used by such a system.

##### Introduction to distributed graph systems.

The distributed system architecture $\mathcal{A}$, representing the model architecture, is defined through a *scheduler*, and a local dynamics (*kernel* $K(\mathcal{A})$) describing the local computations to be performed at each node of the system, and, communication between pairs of nodes connected by edges of the graph representing a given model $M$.

We will generally accept that computations are performed only at $n$ neuron nodes (particles), whereas state variables of the system may appear both on nodes and edges. We will, for simplicity of analysis, consider systems governed by a *synchronous scheduler*, which in successive rounds, acts in two sub-rounds:

- 1.
Computation: computations of the kernel of $\mathcal{A}$ are run at all neuron nodes independently.
- 2.
Communication “over wire”: each neuron node sends specified ‘output variables’ to specified ‘input variables’ of its neighboring neurons.

We expect the scheduler to follow the same communication pattern between neurons over time in a uniform way. In order to avoid artificial constructions of cyclic time-counters at nodes, we will define the architecture kernel through a short sequence of kernels, with the scheduler executing them in successive rounds in round-robin manner. Specifically, when $\mathcal{A}$ is BDH, we will have a sequence of four kernels, $K(\mathcal{A})=(K_{1}(\mathcal{A}),K_{2}(\mathcal{A}),K_{3}(\mathcal{A}),K_{4}(\mathcal{A}))$, with $K_{i}(\mathcal{A})$ being executed in every round $r$ such that $r\equiv i\textrm{\, mod \, }4$.

##### Programmable rulesets and the interaction kernel.

We recall from Section [1.4](#S1.SS4) that a model architecture $\mathcal{A}$ has the interpretation of a computational machine architecture, and models $M$ have the interpretation of programs in architecture $\mathcal{A}$. We also recall that a graph-based model $M$ is defined through a set of parameters which represent the topology and weights of the communication graph of the system.

The above considerations lead directly to the following observation: *The graph of the communication network, which is used for communication between sites by the distributed system architecture $\mathcal{A}$ during reasoning and language inference, has the interpretation of a (trainable, rule-based) program.* Consequently, we embed the subsequent definition of BDH in a kernel formalism, given through a form of *programmable rulesets*, using two-particle interaction rules on a graph.(^5^5 5 We refer the reader to Appendix [D](#A4) for a more principled background discussion, guiding the appropriate choice of formalism for rule-based local interaction.)

The rulesets which we will use to define BDH will closely resemble rulesets (protocols) known from evolutionary and population dynamics  and chemical reaction networks , however, they will be restricted to a special class of interactions.

We start by presenting the more general form of this *interaction kernel*. We then explain how such a kernel can be restricted, allowing it to be naturally implemented using a local graph-based distributed system (in particular, one relying spiking dynamics), while remaining sufficiently expressive to describe an attention-based language model. The resulting restriction will be called the *edge-reweighting kernel*.

###### Definition 1 (Interaction kernel, general form) .

A system with $z$ species, $z\in\mathbb{N}$, and state $(q_{1},\ldots,q_{z})\in Q$, $q_{i}\in R^{+}$, performs the *interaction kernel with a ruleset (protocol) $P$* given by a set of transition rates called *rule weights*, $P=((r_{ijk}\in R^{+})_{i,j,k\in\{1\ldots,z\}},(d_{k}\in R^{+})_{k\in\{1\ldots,z\}})$, producing the following transition from a state $(q_{1},\ldots,q_{z})\in Q$ to a state $(q^{\prime}_{1},\ldots,q^{\prime}_{z})\in Q$:

$$ $q_{k}^{\prime}:=(1-d_{k})q_{k}+\sum_{i,j}r_{ijk}q_{i}q_{j}$ (3) $$

We will describe such a ruleset $P$ using the notational form:

$$ $P=(\{``q_{i},q_{j}\xrightarrow{r_{ijk}}q_{k}"\}_{i,j,k\in\{1\ldots,z\}},\{``q_{k}\downarrow_{d_{k}}\!\!"\}_{k\in\{1\ldots,z\}}).$ $$

As a matter of convention, omitted rules correspond to $r_{ijk}=0$ (respectively, $d_{k}=0$), while rules with no rate value stated next the pointer correspond to $r_{ijk}=1$ (respectively, $d_{k}=1$). If $q_{j}$ is omitted from notation on the left-hand side, we assume $q_{j}=1$.

Equation ([3](#S2.E3)) captures the dynamics of the following differential equation: $\frac{dq_{k}}{dt}=-d_{k}q_{k}+\sum_{i,j}r_{ijk}q_{i}q_{j}$. Assuming $q_{i},q_{j},r_{ijk}\in[0,1]$, the expression $r_{ijk}q_{i}q_{j}$ has the interpretation of a population dynamics or chemical process of the form “$i$ and $j$ give $k$”, with this processes happening at rate $r_{ijk}$, assuming $q_{i},q_{j},q_{k}$ have the interpretation of concentrations of species $i,j,k$. The formalism we use here assumes non-normalized state variables.

We will subsequently use a restriction of the interaction kernel to graph-based systems, which we call the *edge-reweighting kernel*, to describe BDH.

###### Definition 1 (Interaction kernel, general form) .

##### Restricting the interaction kernel to spiking signals and graph systems.

First, we observe that rules of the form used in the interaction kernel from Definition [1](#Thmdefinition1) are extremely easy to implement in systems which rely on stochastic 0/1-valued signals. When $\hat{q}_{i}$ and $\hat{q}_{j}$ are independent random variables in $\{0,1\}$, with $\Pr[\hat{q}_{i}=1]=q_{i}$ and $\Pr[\hat{q}_{j}=1]=q_{j}$, then $q_{i},q_{j}\xrightarrow{}q_{k}$ is expressible as the “AND gate” of probability: the random variable $\delta\hat{q}_{k}:=q_{i}q_{j}\in\{0,1\}$ gives the same expected contribution $\mathbb{E}\delta\hat{q}_{k}=q_{i}q_{j}$ as the considered rule.

We now consider the restriction of interaction kernels to the case of graph systems. In the general formalism, $k$ can be arbitrary with respect to $i$ and $j$. By contrast, consider graph systems, which describe binary relations between nodes, and not (directly) three-point relations. To resolve this, we will require that $i$, $j$, and $k$ have the interpretation of two nodes of a graph and an edge which is incident to them.

For an anchoring in the literature of dynamical systems, we note that already systems following an interaction kernel with a strongly constrained $k$ of the form $k\in\{i,j\}$, exhibit powerful nonlinearities: with such a restriction on $k$, Equation ([3](#S2.E3)) describes the class of evolutionary systems following the equations of *replicator dynamics* , also equivalently known as a non-normalized form of the fundamental Lotka-Volterra predator-prey dynamics. Replicator dynamics can naturally be represented as graph systems whose parameters are defined on *on edges of the graph*, but whose state is updated on *on nodes of the graph*. By contrast, when defining dynamics for reasoning in the current work, we will also need to capture a more powerful class of graph-based systems, where, crucially, state is larger than the number of neuron nodes, appearing on neuron-neuron edges (synapses).

We are now ready to describe a restriction of the interaction kernel from Definition [1](#Thmdefinition1) to the case of node-edge-node interaction rulesets in a graph: the *edge-reweighting kernel*.

##### Definition of the edge-reweighting kernel.

We consider a graph system with $n$ *nodes*, indexed $V=\{1,\ldots,n\}$. Additionally, a subset $E$ of pairs of indexes $(i,j)$, for $i,j\in\{1,\ldots,n\}$ forms the *edges* of the system.

The system has state variables associated (uniformly) with nodes and edges, which we denote with capital letters, e.g., $X(i)$, for $i\in V$ or $Z(i,j)$, for $(i,j)\in E$.

###### Definition 2 (edge-reweighting kernel) .

A distributed system follows the *edge-reweighting kernel* if its dynamics are given by the interaction kernel (Definition [1](#Thmdefinition1)) with a set of non-negative state variables, defined on the set of nodes $V$ and set of edges $E$ of a graph, such that each local rule with non-zero rate is either a *computational rule* involving only state variables on a single node $i\in V$, or a *communication rule* for an edge $(i,j)\in E$, involving state variables from the nodes $i,j$ and edge $(i,j)$.

For context, we remark that, in comparison to the strictly simpler dynamics of node-reweighting governed by graph-based replicator dynamics equations, dynamical systems based on the edge-reweighting kernel given by Definition [2](#Thmdefinition2) are rather elusive to study. We credit the seminal work of Algorithms theory [Fig. 1, Thm 3.2] as the first rigorous study of local edge-reweighting graph dynamics, combining fast-paced linear kernels on nodes with a slower-paced edge-reweighting process, in order to refine (‘focus’) electrical flows on graphs towards a sharper form of cost optimality.(^6^6 6 The graph dynamics used in this approach are naturally phrased in distributed computing parlance, see .) The BDH dynamics that we will introduce here rely on fundamentally different nonlinearities in the process, and will have the interpretation of guiding the system from premises defined at a subset of nodes, towards search targets at nodes representing a desired outcome, through reasoning inference rules with tunable weights set on edges.

In the following Subsection, we will use the introduced formalism to define BDH as an edge-reweighting kernel on the union of edges of several graphs (${G_{x}}^{\mathfrak{e}},{G_{x}}^{\mathfrak{i}},{G_{y}}^{\mathfrak{e}},{G_{y}}^{\mathfrak{i}},{G_{s}}$) with the same set of $n$ nodes.

###### Definition 2 (edge-reweighting kernel) .

### 2.2 Definition of BDH as a local edge-reweighting process (equations of reasoning)

Bearing in mind the discussion of graph dynamics suitable for the case of language inference, and specifically the definition of the edge-reweighting kernel (Definition [2](#Thmdefinition2)), we are now ready to formalize the state-space dynamics of Equation ([6](#S3.E6)) as a local graph dynamics.

###### Definition 3 .

The BDH model with $n$ neurons, with parameters expressed through graphs ${G_{x}}^{\mathfrak{e}},{G_{x}}^{\mathfrak{i}},{G_{y}}^{\mathfrak{e}},{G_{y}}^{\mathfrak{i}},{G_{s}}$ is represented as the ruleset of the edge-reweighting kernel, with $O(n+|E({G_{s}})|)$ state variables, with rule amplitudes given by “the equations of reasoning” in Table [1](#S2.T1).

###### Definition 3 .

##### Inference dynamics of BDH.

The BDH dynamics rely on rapid pulse dynamics with state variables $X(i)$, $Y(i)$, $A(i)$, defined on the $n$ neuron sites of the system, and fast-weight-like state variables $\sigma(i,j)$, defined on a subset of edges of the system, $(i,j)\in E({G_{s}})$. The full implementation of BDH shown in Table [1](#S2.T1)(b) also includes auxiliary state variables $X^{\mathfrak{e}}(i)$, $X^{\mathfrak{i}}(i)$, $Y^{\mathfrak{e}}(i)$, $Y^{\mathfrak{i}}(i)$ which are used as temporary counters, for integration of excitatory and inhibitory signals received by neurons. The dynamics also rely on a set of damping hyperparameters on state, $u>0$, which may in full generality be defined separately as $u(i,j)$ for each edge $(i,j)\in E({G_{s}})$.

Inference with BDH is performed as follows. For some parameter $L$ (e.g. $L=8$ in most of this paper), which would correspond to the number of layers in a Transformer-like system, the system scheduler proceeds through rules in round-robin manner, ingesting new tokens every $4L$ rounds and retrieving results $4L$ rounds later. During round $4l+k$, for $0\leq l<L$, the system performs rules from the $k$-th column of Table [1](#S2.T1), with each such round consisting of a communication step on edges and a local computation step on nodes.

The state-space dynamics of BDH can be rewritten in vector-tensor form, equivalent to the local dynamics of the interaction kernel given in Table [1](#S2.T1). This representation is given by Equation ([6](#S3.E6)) in the following Section.

###### Observation 1 .

The BDH-Graph protocol for the interaction kernel, given for any time round $T=4Lt+(4l+k)$, $0\leq l<L$, $k=\{0,1,2,3\}$ by the ruleset in Table [1](#S2.T1) is equivalent to the state-space dynamics over time $t$ and layers $l$, given by Equation ([6](#S3.E6)).

###### Proof.

For completeness, a detailed explanation of the equivalence is provided in Appendix [C.1](#A3.SS1).
∎

The variables $X(i)$, $Y(i)$, $A(i)$, defined for each of the $n$ nodes of the system, are updated in successive rounds. The state variables $\sigma$ defined on edges are assumed to be distinct over $l$ as $\sigma_{l}$, for $0\leq l<L$; this distinction serves to facilitate interpretation and to strike a balance between the number of parameters and the size of state of the system (assuming a single state matrix $\sigma$, uniform across $l$, does not fundamentally change the operation and scaling laws of the architecture).

Table: Table 1: The “equations of reasoning”: State-space dynamics of the BDH language model expressed through local graph dynamics with the edge reweighting kernel (Definition [2](#Thmdefinition2)). The rules are executed for a distributed system of $n$ neurons performing steps of parallel computation and communication during inference. Model parameters are expressed through the weights of edges of graphs ${G_{x}}^{\mathfrak{e}},{G_{x}}^{\mathfrak{i}},{G_{y}}^{\mathfrak{e}},{G_{x}}^{\mathfrak{i}},{G_{s}}$, and BDH model training is equivalent to defining rule probability amplitudes ${G_{x}}^{\mathfrak{e}}(i,j),{G_{x}}^{\mathfrak{i}}(i,j),{G_{y}}^{\mathfrak{e}}(i,j),{G_{y}}^{\mathfrak{i}}(i,j),{G_{s}}(i,j)\geq 0$ for pairs of neurons $i,j\in\{1,\ldots,n\}$ connected by the edges of these graphs. State is encoded in variables $\sigma(i,j)$ at synapses, representing edges of graph $G_{s}$. The system proceeds in parallel rounds, with new tokens arriving into the system encoded through variables $X(i)$ at neurons and introduced every $4L$ rounds, where $L$ is a parameter of the model (e.g., $L=8$). The set of rules being executed (for each round modulo $4L$) is given in the table. The readout of the system also happens through variables $X(i)$ at the end of each $4L$ rounds. (a) Set of rules for the simplified version of the BDH model with no neuron inhibitory circuits and no thresholding (${G_{x}}^{\mathfrak{i}}={G_{y}}^{\mathfrak{i}}=0$), capturing the general form of the communication structure and synaptic attention of the model. (b) Set of rules for the general case of BDH, including inhibitory circuits ${G_{x}}^{\mathfrak{i}}$, ${G_{y}}^{\mathfrak{i}}$. An execution of the provided rules is equivalent to the state-space dynamics given by Equation ([6](#S3.E6)).

In the representation in Table [1](#S2.T1) we do not impose how the local thresholding operation within some neuron $i$, of the form $A(i)\ ,B(i)\dashrightarrow\left(A(i)-B(i)\right)^{+}$, should be performed. We leave this as a computational primitive, which can be realized based on approximate counting or a comparator. The way natural neurons achieve thresholding to determine whether input signal excitation outweighs inhibition relies on time-integration of impulses. For realizations in other types of distributed systems and population protocols, we refer the reader to the literature on thresholding and Majority Protocols, cf. e.g. .

The definition of the protocol does not specify how variable $X(i)$ should be reset when the scheduler passes from layer $L$ of one input token to layer $0$ for the next input token. As with the definition of state-space equations in Section [3](#S3), we leave this open to allow the dynamics to work both with externally provided input (for next-token prediction), or in a self-feedback loop (for autoregressive operation).

###### Observation 1 .

###### Proof.

##### Notes on training.

Direct training of the BDH model would be performed by selecting the edges of the considered graphs, and then setting rule weights ${G_{x}}^{\mathfrak{e}}(i,j),{G_{x}}^{\mathfrak{i}}(i,j),{G_{y}}^{\mathfrak{e}}(i,j),{G_{y}}^{\mathfrak{i}}(i,j),{G_{s}}(i,j)\geq 0$ for pairs of neurons $i,j\in\{1,\ldots,n\}$ connected by the edges of these graphs.

In what follows, we will train a tensor-friendly special case of BDH, called BDH-GPU, relying on an implicit (generally more efficient) representation of the considered graph parameter weights, using a low-rank product representation for the matrices of these graphs. This representation is reminiscent of the hub-labeling graph representation technique, but is directly suitable for describing and evolving high-conductance scale-free networks. The appropriate architecture is introduced in Section [3](#S3).

### 2.3 Interpretation of attention as a micro-inductive bias of reasoning

Rule weights in the edge-reweighting kernel have the interpretation of micro-programs, governed by rules of transformation of state variables of the form $A(i),B(j)\to\sigma(i,j)$ and $A(i),\sigma(i,j)\to C(j)$, defined on edges between nodes $i,j$ of some $n$-node graph.

This formalism can be seen as running an enormous circuit with a form of universal gates given by the transition rules, over a structure of computational elements at nodes, and memory elements on edges of a graph.

While the local rulesets have the form of a rule-based micro-assembly, we leave open the extent to which they should be considered to have an interpretation of programming in logic (as would be the case, e.g., for C-RASP ). The natural interpretation of $\sigma(i,j)>0$ is a positive bias associated with the neuron pair $(i,j)$, $i,j\in\{1,\ldots,n\}$, which follows from past context. This can be considered by phrasing the local rules of the system in a framework of logic inference; we do so informally, omitting discussion of layers.

> If past context $(x_{\tau}:\tau<t)$ implies that implication $i\to j$ has weight $\sigma_{t-1}(i,j)$, and if the current state at time $t$ implies that $i$ follows from this state with weight $x_{t}(i)$, then the current state at time $t$ implies that $j$ follows from this state with weight $x_{t}(i)\sigma_{t-1}(i,j)$.

The above is intentionally phrased to resemble the logical axiom $(X\to(i\to j))\to((X\to i)\to(X\to j))$, which is perhaps most prevalent across different formalizations of axiomatic logic, with an application of modus ponens as an inference rule. The inference system of the considered model uses state and model weights to devise its own heuristic for the order of evaluation, i.e., to consider which facts appear to be most plausible to be evaluated next, and to evaluate them in an order based on what follows most strongly from context. In a way consistent with what we expect from informal reasoning in language, the considered weights have a more direct interpretation of an increment of utility associated with a given inference.(^7^7 7 Here, the term *utility* is understood in the sense of evolutionary game theory, as applied to the population of neurons, considering the standard interpretation of replicator dynamics, as applied in the ruleset from Table [1](#S2.T1). Neurons which win in the natural selection process are added to the activation $Y$.) In the setting of argumentation, this utility-based approach could, for example, guide the inference process from a pair of known concepts in context, a source and a target, to an intermediate concept likely to be a common-neighbor shortcut lying on a logical path between this source and target (cf. Section [5.3](#S5.SS3) for a discussion of how this type of mechanism is enforced in the feed-forward network of BDH-GPU).

The considered micro-foundational interpretation of attention, defined at the level of individual neurons (or logical variables), does not contradict the way in which Transformer attention is often regarded at the coarser level of vectors through key-query lookup intuitions. At the same time, it highlights that an attention state entry $\sigma(i,j)$ (and similarly, a model edge weight leading from $i$ to $j$) does not have the interpretation of a logical value (i.e., something that is true or false), but an inductive bias associated with how likely the system is to consider the implication ‘$i\to j$’ in its next steps of reasoning, when proposing its next conclusions or next ideas for consideration.

Chains of implications in BDH guide activations along paths in the system graphs ${G_{x}}^{\mathfrak{e}},{G_{y}}^{\mathfrak{e}},{\boldsymbol{\sigma}}$. For the latter, attention allows specific implications to enter into paths of thought once the corresponding synapses are open in state ${\boldsymbol{\sigma}}$.

### 2.4 Interpretation of BDH as an oscillator network toy-model

Whereas the interpretation from Subsection [2.3](#S2.SS3) focuses on properties which fallow from the computational function (purpose) of the system, here we outline an interpretation of the behavior of BDH considered purely as a dynamical system.

##### Definition of the toy-model.

We will consider the toy-model of an $n$-particle system shown in Fig. [2](#S2.F2) as an illustration of the general form of dynamics of the state-space equation ([6](#S3.E6)) of BDH. We draw the $n$ particles in a circle.(^8^8 8 This is a direct tribute to the Kuromato coupled oscillators model; the crucial difference being that in BDH, the elements of state with an interpretation similar to oscillators appear on connections between nodes, not nodes.)

Figure: Figure 2: The ‘physical system’ representation of BDH as a physical graph toy-model.

The particles are connected with each other by state elements, represented in Fig. [2](#S2.F2) as elastic connectors. The topology of these pairwise connections is given by graph ${G_{s}}$, and may in general be dense.

The signal displays dynamics of state ${\boldsymbol{\rho}}$ through tension on connectors, which evolves at a slower time scale, and a more pulse-like activation dynamics $x,y$ (on nodes), appearing and vanishing regularly, at a rapid time scale.

The slower state dynamics represent, in the first order, oscillation or relaxation of the system of elastic connectors. Once an elastic connector between particles $i$ and $j$ has had its endpoints displaced through state $x$ and $y$, respectively, a tension appears on this connector, which causes its displacement ${\boldsymbol{\sigma}}(i,j)$ that relaxes over time (damping variant, corresponding to ALiBi), and/or acts as a spring element (oscillator variant, a simplified illustration of RoPE). Initially, ${\boldsymbol{\sigma}}(i,j)=0$.

The faster dynamics represent the node dynamics of particles. Over time, pulse displacements $x(i)$ happen at nodes, as a result of either previous behavior of the system, or perturbation by an external forcing field (in reality this field would be language input). A node $i$ with displacement $x(i)$ may, due to the aggregated action of tension of *elastic connectors* ${\boldsymbol{\sigma}}(i,\cdot)$ adjacent to it, activate a system of *prods* ${G_{y}}$ adjacent to it, perturbing nodes it hits in this way. If another node $j$ is prodded sufficiently hard, it may cause it to activate a perturbation $y(j)$. The perturbation $y(j)$ of a node $j$ will, in the next step, propagate again to those other nodes $i^{\prime}$, which are connected to $j$ by a system of *wires* (${G_{x}}$). If the aggregated pull of wires on a node $i^{\prime}$ is sufficiently strong, this modifies its pulse displacement $x(i^{\prime})$. The pulse activation $y(j^{\prime})$ of some node $j^{\prime}$, directly followed by pulse activation $x(i^{\prime})$ of node $i^{\prime}$, results in an increase in the tension on the connector $(i,j)$, adding to the value of the tension ${\boldsymbol{\sigma}}(i^{\prime},j^{\prime})$. All pulse activations $y$ subside, and the pulses propagate, consequently altering the slow state ${\boldsymbol{\sigma}}$.

In general, ${\boldsymbol{\sigma}}(i^{\prime},j^{\prime})$ is triggered simply by the temporal connection between the pulse $y(j^{\prime})$ activating, followed by the pulse $x(i^{\prime})$ activating immediately afterwards, even if there was no direct causality between the two (although $y(j^{\prime})$ contributed to pulse $x(i^{\prime})$ happening if $(j^{\prime},i^{\prime})\in{G_{x}}$). An appropriate correspondence of the graphs, ${G_{s}}\subseteq{G_{x}}$, would bring the system close to an observed causal effect on the activated synapse.

The above description of the pulse dynamics was given from the perspective of nodes. From the perspective of connectors, an existing tension on some connector ${\boldsymbol{\sigma}}(i,k)$ propagates through prods ${G_{y}}$ to some nodes $j$, then through wires ${G_{x}}$ to some nodes $i^{\prime}$, and this finally contributes to tensions on other connectors ${\boldsymbol{\sigma}}(i^{\prime},j^{\prime})$. This propagation of state thus happens to 3-hop neighbors, through $i$, $j$, $i^{\prime}$.

During training, the behavior of the system may, in even longer time scales, result in the propagation of changes of connection weight and structures to graphs ${G_{x}}$ and ${G_{y}}$, as well as (optionally) ${G_{s}}$.

##### Effects captured by the toy-model.

We have described a small local graph kernel, with 3-hop locality, capturing the two key effects of the local graph kernel.

The first effect is the graph form of communication pattern between nodes, and thresholding of updates. (We have omitted direct mention of inhibition from discussion of the toy-model, but it is direct to include.)

The second effect is the placement of attention state on node connections, its update patterns, and the dynamics of its relaxation over time.

We intentionally convey the interpretation of node pulses as a differential (gradient) of state on node connections. This interpretation is consistent with our empirical study from Section [7](#S7). It is worth considering once every how many steps of the operation of the toy-model, a single element of state ${\boldsymbol{\sigma}}(i,j)$ is updated. This depends directly on the sparsity of the pulse signals $y(i)$, $x(j)$; at least one of them is, generally, sparse. If the pulses where to happen very seldom for such a pair $(i,j)$, state updates are essentially a “second-order” correction effect. By adjusting the frequency of updates, the system can be made to operate exactly at the critical point where this pulse dynamics ceases to be a second-order correction of state ${\boldsymbol{\sigma}}(i,j)$, giving the random variable describing the time between updates of a connection pair ${\boldsymbol{\sigma}}(i,j)$ a heavy power-law-like tail distribution (possibly with different distribution parameters for different pairs $(i,j)$).

In the description of state dynamics, we noted the hop-distance of 3 in the forward propagation of changes to state. Bearing this in mind is helpful when considering how a gradient backpropagation mechanism would follow dependencies between changes of state if such a system were to have its graph weights altered through backpropagation.

Finally, let us clarify the specific choice of kernel we made for BDH. We found it to work well, and we knew how to train BDH models which implement it on GPU (which we will call BDH-GPU). This, with current hardware, made it $10^{2}-10^{5}$ times more cost- and time-effective to train models and analyze outcomes than kernels, for which we only knew how to train on CPU. Nonetheless, the question of finding optimal kernels according to different criteria (e.g.: minimality of kernel, best training rate per token, closeness to brain function based on known evidence from brain studies), is an extremely pertinent foundational problem. The problem can be phrased in a “closed-ended” way, leaving a finite number of possibilities to be checked, at least when considering small graph kernels. Some kernels may also prove to have superior learning capabilities to the Transformer (and BDH), and if this quality difference is overwhelming, they may eventually prove commercially viable.

In the following, we formalize the choice of kernel for BDH, and also provide a framework to describe other kernels capturing the same effects of graph communication and synaptic attention.

### 2.5 Expressing BDH using brain models

The results we obtain for BDH provide direct corollaries on the expressiveness of brain models which are capable of emulating the local graph kernels of BDH. Specifically, a distributed system, which is able to efficiently emulate the local kernels of BDH, has sufficient expressiveness to perform language inference and reasoning at least to the same extent as BDH.

###### Observation 2 .

The local ruleset of BDH (Table [1](#S2.T1)) can be expressed through a combination of simple mechanisms: neuron activation with positive state variables, Hebbian learning, and communication through excitatory and inhibitory circuits with thresholding.
∎

We note that in the description of the rulesets in Table [1](#S2.T1), Round ($4l+2$) and ($4l+3$) directly describe the use of excitatory and inhibitory circuits with integrate-and-fire thresholding at neurons. Round ($4l+2$) additionally includes a form of competition effect between neurons, realized fully locally at a neurons using the multiplication effect of replicator dynamics. The communication rule of Round ($4l+1$) involves the potentiation of a synapse based on activations of neurons at its endpoints. As was discussed in Subsection [2.1](#S2.SS1), the natural mechanism for implementing increase in synaptic strength is through spiking dynamics, where the execution of the communication rule of Round ($4l+1$) is a stochastic AND-gate on signals. Finally, Round ($4l$) describes the long-term effects of using a strengthened synapse for transmission of signals, and its strength decrease.

We can use the framework of expressiveness, as captured in Observation [2](#Thmobservation2), to shed light on the capabilities of natural systems through their ability to emulate artificial ones. Specifically, if a natural system A can plausibly emulate some artificial system B by using the resources it has at its disposal, and artificial system B is able to solve a problem P, this can be used to explain: (1) why the natural system A is sufficiently powerful to solve problem P, and (2) plausibly, that the purpose for which system A is equipped with certain mechanisms includes solving problem P, if such mechanisms prove useful in the emulation of B.

The experimental validation of the performance of BDH architecture at Transformer level (Section [4.2](#S4.SS2)) confirms that BDH is sufficient to provide language and reasoning function at scale. We can thus make the following statement.

###### Empirical Finding 2 .

The *Hebbian learning mechanism* is plausibly needed, and in combination with neural circuits, sufficient, for performing the *reasoning* function at the scale of the brain. This includes performing language function with attention, and performing thought processes, at a time scale of minutes.

In view of our results, Hebbian learning can be seen as a form of unsupervised learning over time, expressed through graph edge reweighting, to perform reasoning and language inference using the attention mechanism. This type of result can be compared to an analogous interpretation for Hebbian learning in the context of vision, as pioneered in . With the setting of language and chain-of-thought reasoning, we are able to directly capture effects of time in the brain.

Given the interpretation of neuron activations as carrying the necessary gradients of synaptic state (Section [2.4](#S2.SS4)), the problem of supervised learning (i.e., taking into account feedback signals) plausibly becomes deferred to a selective transfer and re-encoding of gradients from state into weights, at longer time scales. We return to a discussion of this point in the Conclusions, bearing in mind the fact that the general difficulty of the problem is now reduced through restrictions on the considered edge-reweighting kernel, and the relative rarity of synapse activation events.

Our work also suggests a framework for further discussion of reasoning function, with an anchoring point for this type of investigation in the time-scale of ‘split-seconds’ to ‘minutes’. The question of shorter time scales is then one of designing more precise communication and computational primitives for spiking neurons and synaptic plasticity, which can be used to perform primitives for individual rules of graph kernels for the inference dynamics.(^9^9 9 While we do not provide direct explanations for effects at shorter time scales and scheduler primitives, we note the type of kernels we rely on are well understood in terms of the ability to work with asynchronous schedulers, and obtaining emergence of synchronization.) The question of longer time scales, and the changes to model structure that follow in a learning process, naturally follows any explanation of unsupervised (Hebbian) learning from the shorter time scale that is considered here, as a mechanism of transfer from state to weights; we come back to this point in the Conclusions.

###### Observation 2 .

###### Empirical Finding 2 .

## 3 BDH-GPU: a tensor-friendly version of the BDH architecture

We will now introduce BDH-GPU, a variant of the BDH reasoning system, expressed in the language of tensor operations typical for Deep Learning models. BDH-GPU provides a GPU-compatible implementation of BDH. BDH-GPU can be easily implemented in PyTorch, a didactic code listing is provided in Appendix [E](#A5)). Furthermore, BDH-GPU can be trained on large text datasets using error backpropagation, and has been shown experimentally to match performance of GPT-based LLMs.

The main steps towards the efficient implementation of BDH-GPU on GPU are:

- 1.
Express graphs $G_{x}$ and $G_{y}$ a low-rank factorizations of their transition matrices, followed by ReLU nonlinearities (we explore graph properties of these approximations in Section [5](#S5)). We never materialize these matrices, but maintain instead a low dimensional state per each neuron.
- 2.
Never materialize the $\sigma$ state matrix, preferring instead to access it using a linear attention operation over low-rank representation of values (we explore the properties of this attention mechanism in Section [6](#S6)).
- 3.
Normalize all state variables using LayerNorm .

We will refer to the architecture in the final intermediate step, before the introduction of LayerNorm, as BDH-Normfree.

### 3.1 Notation for BDH-GPU

We consider the $\textrm{BDH-GPU}(n,d)$ architecture parameterized by positive integers $n,d$. The system scales in dimension $n$ — the number of particles. In what follows, we will use the terms *particle* and *neuron* interchangeably. Dimension $d$ is a measure of the number of parameters per neuron required to represent the interaction of this neuron with the particle interaction field or interaction graph. For asymptotic analysis, we assume that $n\to+\infty$ is the basis for all asymptotics, and $n\gg d>C\log n$ holds for some sufficiently large constant $C>0$. For the tensor representation of the model, which is the primary one for implementation and empirical studies here in this paper, vectors in $R^{d}$ have an interpretation as (fuzzy) addresses of a virtual memory space of size $n$, hence the assumption $d=\Omega(\log n)$ cannot be dispensed with while using natural (linear-algebraic) arithmetic on real numbers. We later show how to avoid this assumption in graph-based models, by using uniform local graph kernels of smaller degree with a graph communication structure.

###### Nonlinearities: ReLU and LayerNorm.

In what follows, we assume that a one-dimensional vector is denoted by a lower-case letter, e.g., $z$, with $z\in R^{n\times 1}\cong R^{n}$ unless otherwise stated. Vectors which appear in dimension $d$ are named with an asterisk, e.g., as $z^{*}\in R^{d\times 1}$. We denote the *ReLU operation* $\left(z\right)^{+}:=\max_{i\in{1,\ldots,n}}\{0,z_{i}\}$.

We further define *LayerNorm* of a vector $z^{*}\in R^{d\times 1}$ in a uniform non-parametric way, $\mathsf{LN}\left(z^{*}\right)=\frac{z^{*}-\mathbf{1}\mathbb{E}_{d}z^{*}}{\sigma_{d}z^{*}}$, where $\mathbb{E}_{d}$ and $\sigma_{d}$ are estimators of mean and standard deviation in dimension $d$, respectively.

###### Activation vectors and parameter matrices.

In vectors representing activations, each scalar element (element of $R$) of the activation vector has the interpretation of a ‘scalar’ activation state of a single particle. Throughout this text, $R$ is generally assumed be the field of real numbers $R:=\mathbb{R}$, and scalars are represented by a fixed-precision floating point number in experiments.(^10^10 10 When only asymptotic analysis is the object, it is sometimes convenient to consider $R:=\mathbb{R}^{k}$ for some $k=2,3,\ldots$. Specifically, considering $R:=\mathbb{R}^{2}$ allows $SO(2)$ rotations on $\mathbb{R}^{2}$ to be expressed as ‘scalar’ ones on $R$, thus making the $\mathbb{R}^{2n\times 2n}$ RoPE block-diagonal matrix of a diagonal matrix in $R^{n\times n}$ . This provides a consistent formalism for ALiBi , RoPE, and extensions such as LieRE as diagonal (communication-free) operations. In all cases, the application of ReLU $\left(\cdot\right)^{+}$ to a scalar remains coordinate-wise in $\mathbb{R}$.)

By convention, in discussions of parameters, matrices denoted ${G_{x}},{G_{y}},{G_{s}}\in R^{n\times n}$ will represent neuron-neuron interaction, encoders $E\in R^{d\times n}$ reduce dimensionality of activation vectors (e.g., $a^{*}=Ez$ for $z\in R^{n}$), and decoders $D\in R^{n\times d}$ lift them back into $R^{n}$ (e.g., $z^{\prime}=Da^{*}$).

Depending on the architecture variant considered, the state will either have the interpretation of a neuron-neuron correlation matrix ${\boldsymbol{\sigma}}\in R^{n\times n}$, or a compressed form with reduced dimensionality, ${\boldsymbol{\rho}}=E{\boldsymbol{\sigma}}\in R^{n\times d}$.

### 3.2 Definition of BDH-GPU as a state-space system

We now define the main architecture of this paper in its tensor flavor, called BDH-GPU.

###### Definition 4 (inference dynamics of BDH-GPU) .

A BDH-GPU state-space system $\textrm{BDH-GPU}(n,d)$, given by three parameter matrices: $E\in R^{d\times n}$ and ${D_{x}},{D_{y}}\in R^{n\times d}$, performs iteration over time $t=0,1,2\ldots$ and layers $l=1,2\ldots L$, governed for any time $t$ by the following recurrence:

$$ $\displaystyle\begin{split}{x_{t,l}}&:=x_{t,l-1}+\left({D_{x}}{v^{*}_{t,l-1}}\right)^{+}\\ {a^{*}_{t,l}}&:=\sum_{\tau<t}{v^{*}_{\tau,l-1}}{x_{\tau,l}}^{T}U^{t-\tau}{x_{t,l}}\\ {y_{t,l}}&:=\left({D_{y}}\mathsf{LN}\left({a^{*}_{t,l}}\right)\right)^{+}\odot{x_{t,l}}\quad\\ {v^{*}_{t,l}}&:=\mathsf{LN}\left(E{y_{t,l}}\right)\end{split}$ (4) $$

where inputs to the system are provided through the boundary condition $v^{*}_{\tau,0}$ in layer $0$, for $\tau=0,1,2\ldots t$.

Here, $U\in R^{n\times n}$ is a diagonal or block-diagonal matrix representing local rotation or damping of state (such as ALiBi or RoPE), $L\in\mathbb{N}$ is the number of layers.

###### Definition 4 (inference dynamics of BDH-GPU) .

##### BDH-GPU as a language model.

BDH-GPU is intended to be used as a language model, processing one token per time step, in which case the input $v^{*}_{t,0}$, for $t\in\mathbb{N}$, is obtained using some (linear) encoding function from the token alphabet $\Omega$, $f_{e}:\Omega\to R^{d}$, as applied to the $t$-th input tokens. Similarly, the logits of the $t$-th output token are extracted using some decoding function applied to outputs of the $L$-th layer $v^{*}_{t,L}$, using a (linear) token decoder function $f_{d}:R^{d}\to\Omega$. The source of language tokens may be external, as is the case for next token prediction tasks, or auto-regressive.

For training, we assume that a model $M$ trained in the $\textrm{BDH-GPU}(n,d)$ architecture has the trainable parameter set $M=(E,{D_{x}},{D_{y}},f_{e},f_{d})$, with all parameters trained together. The model has $3nd+2\Omega d=(3+o(1))nd$ parameters, i.e., the scalable part of the model is concentrated in the total of $3nd$ parameters of the matrices $(E,{D_{x}},{D_{y}})$.

##### State-space representation.

The notation of Definition [4](#Thmdefinition4) is chosen so as to exhibit its direct applicability in a Transformer-like token-parallel training framework. Vector $v^{*}_{\tau,l-1}$ has the interpretation of attention ‘value’ inputs at time $\tau$ in layer $l$. Vector $a^{*}_{t,l}$ represents the result of a linear attention mechanism for time $t$ in layer $l$.

Denoting in ([4](#S3.E4)) the model’s attention state as

$$ ${\boldsymbol{\rho}}_{t-1,l}=\sum_{\tau<t}{v^{*}_{\tau,l-1}}{x_{\tau,l}}^{T}U^{t-\tau}$ (5) $$

we obtain the equivalent but more compact form of representing the inference dynamics of BDH-GPU as a state-space model, presented in Fig. [3](#S3.F3), Eq. ([8](#S3.E8)).

Figure: Figure 3: State-space equations of the model architectures introduced in this paper. All architectures refer to a set of $n$ interacting particles (neurons), with activation vectors $x_{t,l}\in(R^{+})^{n}$. Vector $y_{t,l}\in(R^{+})^{n}$, $y_{t,l}$ is (typically) sparse in the sense of $\|y_{t,l}\|_{0}$. Variables ${\boldsymbol{\rho}}_{t,l}\in R^{n\times d}$ or ${\boldsymbol{\sigma}}_{t,l}\in R^{n\times n}$ represent hidden state of the system. $\diamond$ The graph-based BDH dynamics equation ([6](#S3.E6)), equivalent to the ruleset from Table [1](#S2.T1), serves as a starting point for development of architectures represented as local graph kernels in a distributed computing system. $\diamond$ The simplified BDH-Normfree equation ([7](#S3.E7)) is a special case of BDH. Up to lack of LayerNorms, it approximates the inference dynamics of BDH-GPU, with the correspondence ${\boldsymbol{\rho}}_{t,l}=E{\boldsymbol{\sigma}}_{t,l}$. $\diamond$ The tensor-based BDH-GPU architecture is described by equations ([8](#S3.E8)) (mathematically equivalent to Definition [4](#Thmdefinition4), Eq. ([4](#S3.E4)) and ([5](#S3.E5))) and is the primary point of reference for all model training and all empirical results presented in this study. For a discussion of extensions to BDH-GPU such as heads, see Subsection [4.1](#S4.SS1). A complete code listing for BDH-GPU is provided in Appendix [E](#A5).

Figure: Figure 4: Scaling of BDH-GPU architecture in dimension $n$. The other parameters can be considered fixed during scaling. For example, with choice of $d=256$ for low-rank dimension, $k=2$ for neuron pairing with RoPE, and $h=1$ for a single-head architecture, the model scales linearly in dimension $n$ in chunks of $dhk=256\cdot 2\cdot 1=512$ parameters.

In what follows, we will perform analysis focusing on the state-space representation of the architecture given by Eq. ([8](#S3.E8)).

### 3.3 Interpretation of BDH-GPU as a local interacting particle system

The BDH-GPU dynamics equation ([8](#S3.E8)) has the interpretation of a $n$-particle system, with the state ${\boldsymbol{\rho}}_{t}(i)$ of the $i$-th particle, $i=1,\ldots,n$, given at the end of time $t$ by a vector in $R^{d}$ for each layer:

$$ ${\boldsymbol{\rho}}_{i}(t):=({\boldsymbol{\rho}}_{t,l\ (i,\cdot)}:l\in(1,\ldots L)).$ $$

Overall, as we will see directly, the way particle $i$ interacts with other particles at time $t$ is described by the following tuple $Z_{i}$:

$$ $Z_{i}(t):=({\boldsymbol{\rho}}_{i}(t),E_{(i,\cdot)},{D_{x}}_{\,(\cdot,i)},{D_{y}}_{\,(\cdot,i)}).$ $$

Here, ${\boldsymbol{\rho}}_{i}(t)$ represents the in-context state associated with particle $i$ (initialized as $\mathbf{0}$ at the start of inference), while the other three vectors of length $d$ associated with this particle are trainable, but do not change during inference.

The system scales in dimension $n$ and is completely uniform in this dimension, excepting following effect. Let $k$ denote the size of largest block in the block-diagonal matrix $U$; then particles, are bound by this effect into non-uniform $k$-tuples when $k>1$ ($k=1$ when $U$ is the ALiBi matrix, and $k=2$ when $U$ is the RoPE matrix). Thus the system, in general, scales in the dimension of $n$ uniformly, in chunks of $k$ particles (see Fig. [4](#S3.F4)).

The interaction between particles is, intuitively, local. To be able to proceed with discussion with rigor and without complicating notation, we assume for the analysis that $k=1$. We also drop LayerNorms from the equations of inference dynamics. (Models generally do not train following BDH-GPU without any LayerNorm, but we observed empirically that there is some flexibility as to where these LayerNorms are placed; they can also be moved to the neuron dimension $n$, and they are parameter-free.) The dynamics without LayerNorm are represented under the name BDH-Normfree in Fig. [3](#S3.F3).

We have the following.

###### Observation 3 (local particle interaction ‘by mean-field’) .

The BDH-Normfree dynamics have the interpretation of a mean-field interaction between particles, fully characterized at any time by $O(dL)$ parameters of particle in state, and $O(d)$ parameters in particle representation.

This observation is essential for the subsequent discussion, and it can be expanded in three different ways.

In computing terms, at any time $t$ and in any layer $l$, the action of the system can be represented as an iterated application of the dynamics equations ([4](#Thmdefinition4)), with each of the particles realizing, for each equation in each layer (i.e., a total of $3L$ times), a form of micro-program, involving local computation and communication with other particles by broadcast. In a framework of local distributed computing (cf. e.g. ), it would be represented as a node performing the following form of local kernel as a part of a networked system:

- 1.
compute some message vector $m_{i}\in R^{d}$ locally (without communication with other particles), based only on current activation $x_{t,l,\,i}$, $y_{t,l,\,i}$ and previous state $Z_{i}(t-1)$,
- 2.
broadcast message $m_{i}\in R^{d}$ to other particles,
- 3.
receive the mean-field message $\bar{m}=\sum_{j=1}^{n}m_{j}\in R^{d}$, identical for all particles,
- 4.
update local activation variables for the next layer $l+1$, and update new state $\sigma_{i}(t)\subseteq Z_{i}(t)$, based on the received result $\bar{m}$ of the broadcast and local computation.

In Physical terms, we observe that the interaction field of the particles, which realizes the broadcast, is localized, and can at any time $t$ be expressed as a sum of pairwise particle interaction terms between particles $i,j\in 1,\ldots,t$. These pairwise interactions depend only on parameters $Z_{i}(t-1)$ and $Z_{j}(t-1)$, and the activation variables of these particles, representing properties of these particles at time $t$ and expressible through $O(Ld)$ scalars. This interaction field evolves with time $t$ together with $Z_{i}$ and $Z_{j}$.(^11^11 11 Note that $Z_{i}(t-1)$ depends only on ${\boldsymbol{\rho}}_{t-1,l}$, not ${\boldsymbol{\rho}}_{t,l}$. This is because of the stopping index of $\tau=t-1$ in the definition of attention $a^{*}$ in Def. [4](#Thmdefinition4), and is intentional.)

In Engineering terms, we observe that any transformation of a length-$n$ vector into another length-$n$ vector passes through an intermediary low-rank representation of dimension at most $d$. An example is the equation for $x_{t,l}$ in ([7](#S3.E7)), which reduces length-$n$ vector $y_{t,l}$ to a length $d$-vector through application of the encoder matrix $E$, before lifting the dimension back to $n$ by an application of the decoder ${D_{x}}$.

###### Observation 3 (local particle interaction ‘by mean-field’) .

### 3.4 Expressing BDH-GPU using BDH: preserving parameter and state size

BDH-GPU and BDH both represent $n$-particle systems. For a special parameter choice (of BDH), they have equivalent patterns of communication and of computation (up to placement of layer norms).

###### Observation 4 (BDH-Normfree is a special case of the BDH graph model) .

Models in the BDH-Normfree architecture (Eq. ([8](#S3.E8))) and models in the BDH architecture (Eq. ([6](#S3.E6))) are formally equivalent (i.e., the same model) subject to the following choice of model parameters of BDH:

$$ ${G_{x}}^{\mathfrak{e}}-{G_{x}}^{\mathfrak{i}}={D_{x}}E,\quad{G_{y}}^{\mathfrak{e}}-{G_{y}}^{\mathfrak{i}}={D_{y}}E,\quad{G_{s}}=\mathbf{1}^{n\times n},$ (9) $$

where $\mathbf{1}^{n\times n}$ is the all-ones matrix.∎

We discuss in more details below how BDH compares to BDH-Normfree in terms of size of state and number of parameters needed for one architecture to approximate the inference dynamics of the other. In general, BDH is not less expressive than its tensor-based counterpart.

For BDH-GPU parameters and state are naturally expressed using tensors of $O(nd)$ model parameters. In this section, we discuss how to express model parameters and state of BDH, in such a way as to maintain comparable size of parameter and model space.

###### Observation 4 (BDH-Normfree is a special case of the BDH graph model) .

#### 3.4.1 Expressing matrices D x , D y , E {D_{x}},{D_{y}},E as graphs G x , G y {G_{x}},{G_{y}}

We start by taking care of the first correspondence, that of parameter spaces of BDH-GPU and BDH. Asymptotically, BDH is strictly more expressive at the same number $O(nd)$ parameters. Recall from Eq. ([8](#S3.E8)) that the parameter space of BDH-GPU consists of three matrices ${D_{y}}{D_{x}}\in R^{n\times d}$, $E\in R^{d\times n}$, and (up to shifting of LayerNorms), their role is to encode the pairs of matrices ${D_{y}}E,{D_{x}}E,\in R^{n\times n}$, as used in Eq. ([7](#S3.E7)). In the Claim below, we capture the correct encoding of one such matrix pair in the form of a graph of $O(nd)$ parameters.

Consider a (directed, weighted) graph $G\in R+^{n\times n}$ on a set of vertices $V=\{1,\ldots,n\}$. We will consider a graph which need be directly a sparse graph, but can be represented as a square of a graph with few edges. Formally, we will say that $G\in\mathcal{G}^{2}(n,m)$, for some $m\in\mathbb{N}$, if there exists a graph $H\in R^{(n+s)\times(n+s)}$, with vertex set $V\cup S$, where $|S|=s$, such that $G=H^{2}[V]$, i.e., $G$ is the induced subgraph of $H^{2}$ restricted to vertex set $V$, and $H$ has at most $m$ (strictly positive) edges.

For a $G\in\mathcal{G}^{2}(n,m)$, we can consider an interpretation of a hidden layer $S$ between input layer $V$ and output layer $V$. All matrix weights coefficients are restricted to be non-negative, and the two linear layers are sparse with a total of at most $m$ non-negative connections.

Graphs in $\mathcal{G}^{2}(n,m)$ are naturally expressed through the edges of the previously defined graph $H$, using $O(n\log n+m)$ parameters. The class $\mathcal{G}^{2}(n,m)$ is strictly more expressive than the class of sparse ($m$-edge) graphs on vertex set $V$.(^12^12 12 The formal expression in the definition of the class of weighted graphs $\mathcal{G}^{2}(n,m)$ can be compared to that of the class of graph distance matrices admitting sparse hub labeling representation (or closely related landmark labeling). In our case, vertices in the hidden layer $S$ also have a natural interpretation of landmarks on directed paths connecting nodes of $V$.)
We will refer to the middle layer of vertices $S$ that makes such a representation possible as the *sparse synaptic layer*, to the graph $H$ on vertex set $V\cup S$ as the *sparse linear circuit*, and the graph $H^{2}[V]\in\mathcal{G}^{2}(n,m)$ as the *neuron-neuron interaction graph*.

The role of the constructed graphs is to serve for propagating linear dynamics of the form $v\mapsto Gv$, $v\in(R^{+})^{n\times 1}$, for graph-based local models. We have the following Observation.

###### Observation 5 .

Let $G\in\mathcal{G}^{2}(n,m)$ be a neuron-neuron interaction graph, with $G=H^{2}[V]$, where $H$ is the sparse linear circuit on vertex set $V\cup S$, which has $m$ edges. Then, the linear dynamics on graph $G$, $v\mapsto Gv$, can be efficiently expressed through two steps of linear dynamics on graph $H$, $v\mapsto H^{2}v$, for $v\in(R^{+})^{n\times 1}$. This representation requires $O(m)$ parameters.
∎

In the above, thee exact number of parameters needed to represent a graph of $m$ edges follows from conventions introduced in the Notation (Section [1.4](#S1.SS4)). In what follows, we will assume that BDH represents its parameter matrices through appropriate sparse linear circuit graphs $H$, which it uses to realize the linear neuron-neuron interaction dynamics $G$. We illustrate the correspondence between graphs $G$ and $H$ in Fig. [5](#S3.F5).

Figure: Figure 5: Neuron-neuron communication using graphs $G\in\mathcal{G}^{2}(n,m)$: correspondence between graph $H$ with $m$ edges (left), and neuron-neuron interaction graph $G=H^{2}$ (right). The approach allows to express linear signal propagation on a broad class of graphs $\mathcal{G}^{2}(n,m)$ using two steps of linear dynamics on a sparse circuit $H$, i.e., $Gz=H^{2}z$ for $z\in(R^{+})^{n}$.

We observe that BDH can express BDH-GPU parameter matrices with the same asymptotic number of parameters. The claim below applies to pairs of matrices $DE$, for $D={D_{y}}$ and $D={D_{x}}$.

###### Claim 3 .

For any matrices $D\in R^{n,d}$, $E\in R^{d,n}$, there exist neuron-neuron interaction graphs $G^{\mathfrak{e}},G^{\mathfrak{i}}\in\mathcal{G}^{2}(n,m)$, such that $G^{\mathfrak{e}}-G^{\mathfrak{i}}=DE$, with $m=O(nd)$.
In consequence, for the same asymptotic number of parameters $O(nd)$, graph-based feed-forward mechanisms of BDH are strictly more expressive than corresponding mechanisms in the tensor-based implementation, BDH-Normfree.

###### Proof.

The short proof of the Claim is deferred to Appendix [C.3](#A3.SS3).
∎

We note that the converse implication does not hold: an arbitrary graph $G^{\mathfrak{e}}\in\mathcal{G}^{2}(n,m)$ does not admit an exact low-rank decomposition $G^{\mathfrak{e}}=DE$. Indeed, in general any low-rank decomposition introduces a form of noise whose implications we discussed in Section [5.3](#S5.SS3): if $G^{\mathfrak{e}}$ has a modular (cluster) structure, the low-rank approximation $G^{\mathfrak{e}}\approx DE$ still allows a form of in-cluster propagation dynamics.

###### Observation 5 .

###### Claim 3 .

###### Proof.

#### 3.4.2 Expressing BDH-GPU attention on graphs: sparsification and trainability of G s {G_{s}}

We recall that by Observation [4](#Thmobservation4), the equivalence between the attention state ${\boldsymbol{\sigma}}_{t,l}$ in BDH and in tensor-based implementation holds for the case of the complete directed graph, ${G_{s}}=\mathbf{1}^{n\times n}$.

This means two things: first, in BDH, graph ${G_{s}}$ can be trainable, while in BDH-GPU it is not. This acts to the potential advantage of BDH for expressiveness.

Second, in BDH, the graph ${G_{s}}$ obtained through the direct correspondence is dense: with $n$ neurons, BDH would need $n^{2}$ synapses to precisely reflect BDH-Normfree. This aspect is more of a technical nuisance than an actual difference: the expressiveness of the attention mechanism of BDH, equipped with a sparse graph ${G_{s}}$, is sufficient to represent the attention operation as used in BDH-Normfree. Indeed, in the tensor-based BDH-GPU dynamics, the attention operation is immediately followed by a low-rank operation, ${\boldsymbol{\rho}}_{t,l}=E{\boldsymbol{\sigma}}_{t,l}$, where ${\boldsymbol{\rho}}_{t,l}$ has $nd$ parameters. Graph models can instead rely on a sparse graph ${G_{s}}$ to achieve the same form of state compression through sparsification.

###### Claim 4 .

The attention block of BDH-Normfree can be expressed using the attention block of BDH with a graph ${G_{s}}$ having $O(nd)$ edges, subject to a natural preparation of attention values entering the attention block of BDH (directly before this attention block).

###### Proof.

The formal statement of the Claim and its proof are deferred to Appendix [C.4](#A3.SS4).
∎

Going beyond the formal equivalence between BDH and BDH-GPU from Observation [4](#Thmobservation4), the above Claim, combined with Claim [3](#Thmclaim3), shows that BDH has at least the same expressiveness as BDH-GPU even for the same number of parameters $O(nd)$ and the same size of state $O(nd)$ per layer.

Independent of graph-based models, in the subsequent analysis of the feed-forward network and attention mechanisms of BDH-GPU, we will show that the matrices ${D_{x}}E,{D_{y}}E,\sigma\in R^{n\times n}$ of BDH-GPU admit a natural interpretation as $n$-node directed graphs (once appropriate threshold functions are applied). For example, the visualizations in Fig. [11](#S6.F11) and Fig. [10](#S5.F10) correspond to graph representations of matrices ${\boldsymbol{\sigma}}$ and ${G_{x}}:={D_{x}}E$ of BDH-GPU, respectively, after thresholding. This graph interpretation of matrices in BDH-GPU *also* defines the neuron-neuron communication graph of the underlying BDH model, given by the equivalence from Eq. ([9](#S3.E9)).

###### Claim 4 .

###### Proof.

## 4 Implementation and scaling laws

A code framework for BDH-GPU, representing the architecture from Definition [4](#Thmdefinition4), is made available in the Appendix [E](#A5). In this Section, we present some guidelines on choice of hyperparameters, and an empirical study of models implemented in the BDH-GPU architecture, as well as a comparison to the Transformer and other language model architectures.

### 4.1 Implementation characteristics of BDH-GPU

Figure: Figure 6: Diagram of one layer of the BDH-GPU architecture, following Eq. ([8](#S3.E8)). Layer inputs are $x_{l-1},y_{l-1}\in R^{n}$, layer outputs are $x_{l},y_{l}\in R^{n}$. Model parameters are contained in the $E\in R^{n\times d}$ and ${D_{x}},{D_{y}}\in R^{d\times n}$, and shared across all layers. Each layer has a state ${\boldsymbol{\rho}}_{l}\in R^{n\times d}$ which is used in the Linear Attention block and persisted over time. PyTorch code implementing the model is provided in Appendix [E](#A5)

##### Model scaling in neuron dimension n n .

The architecture $\textrm{BDH-GPU}(n,d)$ has two main dimensions, $n$ which is the dimension of its concept (*neuronal*) space, and $d\ll n$, which is its low-rank (*synaptic*) dimension. The model scales primarily with the number of neurons $n$. Almost all of the weights of the model are contained in three $n\times d$ parameter matrices called $E,{D_{x}},{D_{y}}$; thus, the number of parameters is precisely $(3+o(1))nd$.
The ratio between the dimensions $n$ and $d$ increases rapidly (“asymptotically”) with model size; already for a 25M-parameter model, a sound choice of dimensions is: $d=256$, $n=32768$, read as $32768$ neurons, each characterized by a total of $3d=3\cdot 256=768$ scalar parameters.

##### Layers and heads.

The architecture has $L$ layers (e.g., $L=10$). As in the Universal Transformer , all layers use the same set of weights for each of the parameter matrices.

The architecture may be equipped with several heads $h$, subdividing dimension $n$. The role of heads is limited to a single parameter-free LayerNorm, normalizing outcomes of linear attention separately for each head. The optimal number of heads is typically smaller than in the Transformer (e.g., $h=4$).

##### Linear attention with state aligned to neurons.

The state space of the model is fixed and large. It has the macro-interpretation of associative memory (like KV-cache, but organized differently), and is used to perform linear attention. For each layer, the state space is independent and has a fixed dimension $n\times d$, the same as the model weight matrices. Thus, a portion of $d$ parameters of a state is directly associated with each of the $n$ neurons. With each token processed, a fraction of the model’s state space is updated. Sharing of state between the $L$ layers is not performed in the vanilla architecture. As usual with SSM’s, there is no notion of a context window.

Similarly to BDH, BDH-GPU maintains a large recurrent state comparable in size with its total number of parameters (c.f. Fig. [4](#S3.F4)). This stems from the fact that both the recurrent state matrix, and parameter matrices are expressed as low rank $d$ factorizations of $n\times n$ graph transition matrices. We believe that this helps the model with generalization with respect to RNNs which have $O(N^{2})$ parameters which manipulate a state of size $O(N)$.

##### Sparse positive activation.

The architecture relies on a length-$n$ vector $x_{t,l}$ passed to the $l$-th layer for the $t$-th token processed, which can be assimilated to the vector giving rise to keys, values, and queries in the Transformer, but operating in higher dimension. As a crucial design assumption, this vector has all non-negative elements ($x_{t,l}\geq 0$).

An empirically observed fact is that the activation pattern of $x_{t}$ rapidly becomes sparse (in a typical training run, only $\rho\approx 5\%$ of the $n$ entries of vector $x_{t}$ are non-zero). This corresponds to the fraction of the state space read and updated for each token.

### 4.2 Comparison of BDH-GPU to GPT2-like Transformers

##### Architecture differences.

BDH-GPU in its vanilla form can be compared to the GPT2 architecture  with RoPE attention. In this comparison, BDH-GPU retains or strengthens the key advantages of the Transformer (parallel trainability, attention mechanism, scaling laws for loss versus parameter count, learning rate per token) on tests and benchmarks at the model scales we tested (1B parameters), across tasks such as language and translation.

The architecture of a single layer of BDH-GPU is presented in Fig. [6](#S4.F6). The most evident architecture differences between BDH-GPU and the Transformer include the following:

- $-$
BDH-GPU has fewer parameter matrices, allowing for more compact interpretation and analysis.
- $-$
BDH-GPU scales for parameters (and context length) almost exclusively in a single neuronal dimension, $n$.
- $-$
Key-value state and parameter matrices have matching dimensions and are highly localized together with state, with portions of these matrices attributable to individual neurons.
- $-$
There is no notion of context length in BDH-GPU, and consequently no hard bound on it.
- $-$
Attention of BDH-GPU is linear, but happens in the model’s large neuronal dimension.
- $-$
Activation vectors $x,y$ of BDH-GPU are positive (after passing through ReLU gates), and vectors $y$ are observed to be extremely sparse in practice.

##### Transformer-like scaling laws.

We have experimentally validated the scaling laws of BDH-GPU, expressing loss as a function of parameter count, for next-token-prediction tasks. At the same parameter scale, BDH-GPU generally compares favorably to the Transformer even on relatively short-context tasks requiring use of attention, such as translation, Fig. [7](#S4.F7). In general, on next-token prediction tasks, BDH-GPU appears to show improvement of loss reduction per token of data than the Transformer, i.e., *learns faster per data token*, both for the natural tasks we tested (see e.g. Fig. [7](#S4.F7)) and on synthetic puzzles.

Figure: Figure 7: Performance of BDH-GPU and GPTXL versus model size on a translation task. We have tested all models under the same training and evaluation regimes. All models show improved performance with scale. BDH-GPU uses exactly the formulation provided in Appendix [E](#A5), while BDH-GPU’ extends conditional gating of states and logits. All models are trained with truncated backpropagation through time on sequences 2048 characters long, and carry their state (${\boldsymbol{\rho}}$ matrix for BDH models and a buffer of last 4096 KV-Cache entries for GPTXL) between minibatches. BDH models are scaled only by varying the number of neurons $n$ and keep all other hyperparameters fixed, making them easy to scale. On the other hand, GPTXL were scaled in both the embedding dimension and the number of layers and required Dropout tuning for optimal performance. We observe that BDH-GPU’ matches the GPT Transformer at all model sizes we have evaluated. Details on model hyperparameters and training setup are provided in Appendix [B.2](#A2.SS2)
Refer to caption: https://arxiv.org/html/2509.26507/2509.26507v1/bdh_scaling.png

The BDH-GPU architecture appears to be a preferred choice for training setups where: (1) models need to learn from scarce data, or (2) training workloads need to be optimized for makespan. For the first setting, the training rate per token is the decisive factor. For the second setting, BDH-GPU can be used differently than the Transformer in distributed training and distributed inference setups because of the way it scales its dimensions.

##### FLOPS counts.

The theoretical count of arithmetic operations per token of BDH-GPU during inference is bounded by $O(ndL)$. Each parameter is accessed $O(L)$ times per token (with the typical sufficient number of layers being smaller than in the Transformer), and each element of state is accessed $O(1)$ times per token, with small hidden constants. These are rough bounds for a simple implementation, and do not take advantage of activation sparsity.

For short contexts BDH-GPU is amenable to parallel training with a causal self-attention kernel. The simple code template provided in the Appendix [E](#A5) is sufficient to reproduce the empirical results presented in this paper on a single GPU node. For longer contexts (typically above 4096 tokens for $d=256$), a state-space kernel for linear attention is faster and more space-efficient.

### 4.3 Comparison of BDH-GPU to other sequence processing architectures

##### Transformers with Linear Attention.

Linear attention works well when used in high dimension, subject to appropriate preparation of key vectors (as we discuss in Subsection [6.1](#S6.SS1)). An elegant way to eliminate non-linearity of attention, by applying preparation of key vectors through tensor product, was proposed in . We use a completely different approach to achieve attention in high dimension.

A much broader line of work on linear attention for the Transformer, initiated by  concerns applying linear attention in low dimension after appropriate preparation of keys and values. This is effectively a technique for SSM state compression, and it is not clear whether it relates favorably to other SSM state compression techniques. An empirical study of the amount of information recoverable from SSM’s with compressed state can be found in .

A general theoretical framework for analyzing the expressiveness of Linear Attention models with attention working with positive vectors can be found in the context of the FAVOR+ framework of the Performer . Finally, a general state-space formalism for Transformer models admitting Linear Attention was considered in .

##### Other types of Transformers.

Variants of the Transformer with identical parameters in all layers, like the Universal Transformer , have a number of desirable features, notably in terms of explainability and ease of defining metrics. The downside of sharing parameters between layers in the Universal Transformer is a slight time overhead for the feed-forward network operations, when measured in FLOPS per parameter. The situation is similar in BDH-GPU.

BDH-GPU has sufficient expressiveness to prepare keys and queries for the attention operation, so that the outcome of attention captures a similarity measure between keys and queries corresponding to the outcome of a class of Locality Sensitive Hashing (LSH) functions with a very large number of buckets (cf. Subsection [6.1](#S6.SS1)). The study of LSH-based KV-cache for the Transformer was initiated with the Reformer , and the LSH Transformer architecture introduced in the same work. Generally, the LSH Transformer is shown to rapidly approach Transformer baseline behavior in practice as the number of buckets increases. The class of LSH functions considered is not the same, but some intuitions gained in the study of LSH attention may carry over to BDH-GPU.

Finally, several lines of work have been devoted to making the Transformer work with longer context windows. Two distinct approaches, which work notably well, are the soft-rolling context window of the TransformerXL , and hierarchical attention . The BDH-GPU architecture is, generally, amenable to some of these extensions to the Transformer’s attention mechanism, while also providing new ways to extend context length in a more uniform manner.

##### Networks with sparse activation.

The use of the ReLU gate as a systematic way to achieve sparse activation was, to our knowledge, first exhibited in .

A recent variant of the Transformer called Spark Transformer  relies on a combination of top-k operations and soft thresholding to provide a reduction in both attention and feed forward network activations compared to the Transformer, achieving neuron sparse activation of 8%. Compared to our work, the method used therein to achieve activation sparsity effects is completely different (and rather involved). Beyond the question of sparsity, BDH-GPU is not more similar to the Spark Transformer than to the Transformer.

##### Oscillatory SSM’s.

BDH admits an interpretation at the micro-level as an oscillatory state-space network, as we outlined in Subsection [2.4](#S2.SS4). The concept of Oscillatory State Space Models has recently been applied to time series analysis , with the LinOSS model showing encouraging performance relative to other SSM’s (such as Mamba and LSTM’s) on tasks of long-horizon forecasting and time-series classification. Other than this, the use of SSM’s with the form of an oscillator network has been limited to smaller scale studies. We are not aware of any successful application of oscillatory SSM’s to the area of language models and reasoning in language, nor of oscillator network SSM’s at scale whatsoever, prior to BDH.

BDH unifies multiple lines of intuition found across existing models, offering a coherent framework in which the components naturally align. The result is a biologically plausible reasoning model with an interpretable structure and state-of-the-art performance that has been experimentally verified.

## 5 Analysis: emergence of modularity and scale-free structure

Large-scale reasoning systems appear to benefit from hierarchical structuring into sub-modules. In Machine Learning, the usual approach has been to design such a modular structure, by way of assigning roles and scales to different sub-modules explicitly. Many works have postulated modules capable of representing hierarchical relationships between features of objects, e.g., capsule networks . Some models have attempted to capture intelligence by recreating elements of structure recognized in brain study, going so far as to try to map functional sub-networks of the brain with empirically identified function into specific sub-modules in the design of a larger ML system, cf. .

In this work, we propose a learnable system which ends up with modularity. We show how scale-free modular structure emerges naturally when the model is implemented by a network with local graph dynamics.
In this Section, we discuss the emergence of the structure of inter-neuron connections of BDH during training, while in Section [6](#S6) we look at its temporal activation patterns during reasoning inference.

The rest of this section is organized as follows. In Subsection [5.1](#S5.SS1), we introduce basic concepts related to modularity and scale-free behavior of networks. We then look at the expressiveness of feedforward networks of BDH-GPU and their usefulness as a signal propagation dynamics in Subsections [5.2](#S5.SS2) and [5.3](#S5.SS3). In Subsection [5.4](#S5.SS4), we show theoretically how modular graph structure, with appropriate community voting mechanisms, emerges as a plausibly necessary element for the feed-forward networks ${D_{x}}E$ and ${D_{y}}E$ to function correctly. In Subsection [5.5](#S5.SS5), we look at the corresponding empirical properties of these matrices, and the scale-free and modularity properties of the corresponding graphs ${G_{x}}^{\mathfrak{e}}$ and ${G_{y}}^{\mathfrak{e}}$ of the underlying BDH graph dynamics.

### 5.1 Background: modularity and scale-free property of systems

##### Importance of modularity for information propagation.

Graph systems serving a function related to information propagation tend to achieve modular graph structure, and rely on it to obtain the most desirable tradeoff between efficiency and accuracy of the system dynamics. Such emergence of “hidden structure” may be observed e.g. through topic specialization of system regions, or through the coordinated voting behavior among nodes which organize themselves into communities, admitting higher local density. This type of graph community self-organization has two main advantages over a system with an explicit partition into subsystems. First, it allows nodes to belong to multiple communities, and to act as bridges between them. Second, it allows the scale and relationship between communities to evolve over time, as their relative importance changes or new connections emerge.

Historically, the crucial role of emergent modular structure for systems tasked with efficient knowledge retrieval at scale was first observed in the context of the World Wide Web before the year 2000, notably in the transition from catalogue-based systems (DMOZ Open Directory Project, craigslist, etc.) to naturally evolving systems based on webs of knowledge (Wikipedia, etc.), interlinked topic-based communities (reddit, etc.), and reliance on evolving network link structure for assigning expert weights to nodes in a voting process (Google PageRank, etc.). Formalization of modular properties followed soon after, with the mostly commonly used definition of modularity being proposed by Newman in . The main theoretical reference for studies of modularity is the Stochastic Block Model  and its later generalizations, e.g., to hierarchical settings. While the definition of Newman modularity is not (efficiently) constructive, it can in practice be closely approximated by greedy algorithms  or spectral approaches .

##### Scale-free property.

The scale-free property of natural systems dealing with information processing is generally accepted as a manifestation of their operation at criticality. This refers to operation within a regime where they are both sufficiently stable to enable efficient information retrieval in the short-term, and sufficiently adaptable to be able change their behavior abruptly as new knowledge inputs become available, invalidating previous paths of reasoning or knowledge retrieval. The generally accepted definition of scale-free behavior of such a dynamical system assumes that the likelihood of a new piece of information (or other localized innovation to the system) to affect $n^{\prime}$ nodes of the system, for any $n^{\prime}<n$, should by polynomially large in $1/n^{\prime}$. For most information propagation dynamics, under certain uniformity assumptions, e.g., that the new piece of information arrives at a uniformly random node of the system, a usual necessary (not sufficient) condition for scale-free property is for the distribution of node degrees to follow a power-law distribution.

In the practice of applied sciences studying real-world network phenomena, and in the absence of the possibility to perform more in-depth analysis, power-law degree distributions are sometimes equated with scale-free behavior. One notable research application involves modeling of extreme events: understanding scale-free behavior allows researchers to make predictions about rare, large events from data on smaller, more common ones.

For systems with known local graph dynamics, like those considered here, more refined analysis of scale-free properties are possible. We nonetheless also report heavy-tailed degree behavior as the most obvious lithmus test indicator of scale-free operation of the system.

### 5.2 BDH-GPU feed-forward network with the ‘ReLU-lowrank’ block

Low-rank matrices have been considered in multiple contexts of Machine Learning, from preference vectors to Internet latency estimation. In the setting of the Transformer, low-rank matrices form the basis of weight-matrix approximations such as LoRA .

The ReLU-lowrank block of BDH-GPU captures different properties than the above settings. Its most important effects for BDH-GPU are related to noise reduction, and faithful representation of a certain class of affinity functions on sparse positive vectors. This makes it suitable for use in combination with Linear Attention blocks. We discuss this point further in this Section.

##### Definition of ReLU-lowrank.

The parameters of BDH-GPU are concentrated in three matrices $E,{D_{x}},{D_{y}}$. The encoder matrix $E$ transforms length-$n$ vectors in the neuronal layer into length-$d$ vectors in the hidden layer. The two decoder matrices $D\in\{{D_{x}},{D_{y}}\}$ transform length-$d$ vectors in the hidden layer back to the neuronal layer.

We consider the *ReLU-lowrank* operation of passing through the encoder, one of the decoders, and a ReLU gate (cf. Eq. ([7](#S3.E7))), mapping vectors $z\in R^{n}$ into $f_{DE}(z)\in R^{n}$ as follows:

$$ $z\mapsto f_{DE}(z):=\left(DEz\right)^{+}.$ (10) $$

We note that the output $f_{DE}(z)\in(R^{+})^{n}$ always, and that in BDH-GPU we also always have $z\in(R^{+})^{n}$.

##### Expressiveness of ReLU-lowrank in BDH-GPU and MLP in the Transformer.

A single ReLU-lowrank block can be compared to a single MLP block of the Transformer. A different comparison provides closer matching of dimensions and structure of nonlinearities, by considering a single ReLU-lowrank with respect to a portion of the Transformer corresponding to the second MLP layer in an MLP block, i.e., starting with the hidden layer of neurons of the MLP in some layer $l$, skipping attention blocks, and followed by the ‘first’ linear layer of the MLP in layer $l+1$, finally followed by the non-linearity (typically GeLU) applied in the hidden layer of neurons in layer $l+1$. Either approach to expressiveness is valid to the extent where we analyze similarities between one architecture with $L$ layers and the other with “$O(L)$” layers.

In the spirit of universal approximation theorem frameworks, a (deep) layer-$L$ stacking of Transformer’s MLP block with ReLU activation, for Transformer latent dimension $D$ and MLP hidden layer dimension $cD$ (e.g., for $c=4$), is eventually (i.e., for $L\to+\infty$) a universal approximator for all vector functions up to dimension $D-O(1)$ . A similar universal approximation result eventually (i.e., for $L\to+\infty$) holds up to function dimension $n$ for residual ReLU-lowrank networks , however the convergence rate in $L$ is slower due to the smaller size of the hidden layer. These results translate directly to BDH-GPU architecture which also relies on ReLU with residual connections between layers. To summarize informally, for a Transformer with latent dimension $D$ and BDH-GPU with hidden dimension $d$, we expect their feed-forward networks to be comparably expressive (though usually without strict mathematical equivalence) as function approximators for functions up to some dimension $d^{\prime},d<d^{\prime}<D$, between $d^{\prime}$ and $D$ the Transformer can express a richer class of functions, and between $D$ and $n$, BDH-GPU can approximate some functions, whereas the Transformer does not use such high dimension in its vector representations.

We remark that in all cases, regardless of expressiveness of feed-forward mechanisms, BDH-GPU is set up so that it is only using inputs and producing outputs within the positive orthant, $(R^{+})^{n}\mapsto(R^{+})^{n}$.

The main point to consider is: *what classes of useful high-dimensional functions in the positive orthant does ReLU-lowrank naturally express?*

### 5.3 ReLU-lowrank as a signal propagation dynamics

##### Error of low-rank approximation (without ReLU).

Consider $R^{n}$ as a space spanned by a fixed set of $n$ orthogonal unit basis vectors $V=\{v_{1},\ldots,v_{n}\}$, called *nodes*.

The low-rank operation can be used to approximate affinities between pairs of nodes, in the following sense. For a given matrix $G^{\prime}\in R^{n\times n}$, consider low-rank matrices $D\in R^{n\times d},E\in R^{d\times n}$, such that $G:=DE$ approximates $G^{\prime}$ pointwise. (^13^13 13 Elements of $G$ can be computed pointwise by each pair of nodes: $V\times V\ni(v_{1},v_{2})\mapsto G:={v_{1}}^{T}DE{v_{2}}\in R$.)

Assume $\|G^{\prime}\|_{1,\infty}\leq 1$. An application of the Johnson-Lindenstrauss lemma shows that the following bound holds in the infinity norm: $\|G^{\prime}-G\|_{\max}=O(\sqrt{\log n\ /\ d})$ (cf. e.g. ). Then, for $z\in R^{n}=R^{|V|}$ with $\|z\|_{1}\leq 1$, we have:

$$ $\|G^{\prime}z-Gz\|_{+\infty}=O(\sqrt{\log n\ /\ d})$ (11) $$

However, no similar bound holds for $\|G^{\prime}z-Gz\|_{2}$. Even for ‘simple’ scenarios like the identity transformation $G^{\prime}=I_{n}$, the best low-rank approximation admits O(1) additive error in the L2-norm for almost all inputs, and even greater distortion (approaching $\sqrt{n}$) may appear in the L1-norm.

This makes the low-rank operation useful for determining affinity of pairs of coordinates in dimension $n$, but more problematic as a vector transformation function. However, the ReLU-lowrank mechanism (Eq. ([10](#S5.E10)) is able to suppress a part of the noise of the linear low-rank map, allowing to approximate a sufficiently broad class of non-linear operations.

##### Expressiveness of ReLU-lowrank for Markov chain propagation.

We will consider positive inputs $z\in R^{+n}$, focusing on sparse vectors.

One important case concerns approximating a Markov chain transformation $z\mapsto G^{\prime}z$, for some $G^{\prime}\in R+^{n\times n}$. For such a transformation in the positive orthant, adding the ReLU operation to the linear map does not change anything directly, $G^{\prime}z=\left(G^{\prime}z\right)^{+}$. However, when considering a low-rank matrix $G$, the non-linear transformation $\left(Gz\right)^{+}$ can provide a closer approximation of $G^{\prime}z$ for some classes of input vectors $z$, than the low-rank linear operation $Gz$.

We start with the following illustrative example.

###### Claim 5 (propagating a Markov chain) .

Let $G^{\prime}$ be the random walk matrix of a directed graph with out-degree $r$ (i.e., a stochastic matrix with $r$ non-zero entries of $1/r$ in each row), and let $v\in V$ be a node (basis vector), $\|v\|_{1}=\|v\|_{2}=1$. Then, for any $\varepsilon>0$, there exists $d=O(r^{3}\log n/\varepsilon)$ such that for some matrices $D\in R^{n\times d},E\in R^{d\times n}$, we have $\|G^{\prime}v-f_{DE}(v)\|_{1}=O(\varepsilon)$.

###### Proof (sketch).

Let $D^{*}\in R^{n\times(d-1)}$, $E^{*}\in R^{(d-1)\times n}$ denote matrices $D$, $E$ restricted to all but the last coordinate in dimension $d$. Pick $D,E$ so that $\|G^{\prime}v-D^{*}E^{*}v\|_{\infty}<\varepsilon^{*}$, where $\varepsilon^{*}=\varepsilon/r$, following Eq. ([11](#S5.E11)) (we have $\|G^{\prime}\|_{1,\infty}\leq 1$ by stochasticity of $G^{\prime}$). Further, set a fixed bias, placing $1$ on all entries of the last coordinate in dimension $d$ of $D$, and $-\varepsilon^{*}$ on all entries of the corresponding last coordinate in dimension $d$ of $E$. Taking into account this bias, we now have
$\|(G^{\prime}v-\varepsilon^{*}\mathbf{1})-DEv\|_{\infty}<\varepsilon^{*}$.

For all coordinates $v_{j}\in V$ such that ${v_{j}}^{T}G^{\prime}v=0$, we now have ${v_{j}}^{T}DEv<0$, hence also ${v_{j}}^{T}f_{DE}(v)=0$. For all other coordinates $v_{j}$, we have ${v_{j}}^{T}G^{\prime}v=1/r$, and $1/r-2\varepsilon^{*}<{v_{j}}^{T}f_{DE}(v)\leq 1/r$. Thus, $\|G^{\prime}v-f_{DE}(v)\|_{1}\leq 2\varepsilon^{*}r$, and the claim follows.(^14^14 14 As a point of elegance, we note that in this proof, ${v_{j}}^{T}f_{DE}(v)\leq 1/r$, so $f_{DE}(v)$ was not an *unbiased* estimator of $G^{\prime}v$. This is easily fixed in the first-order by introducing a global multiplicative bias of $(1+\varepsilon^{*})$ to the approximation, for example, substituting: $(1+\varepsilon^{*})D\mapsto D$.)
∎

The above observation shows how ReLU-lowrank deals with one specific class of graph affinity functions (random walks of adjacency of sparse graphs), for transformations of vectors which are nodes in our distinguished basis. We use this example as it is the simplest case which exhibits the benefit of threshold non-linearity: for basis vectors, the operation $f_{DE}$ captures a basic propagation effect which is well known (in general) to require a *full-rank* matrix $G^{\prime}\in R^{n\times n}$ if relying only on linear operations.

###### Claim 5 (propagating a Markov chain) .

###### Proof (sketch).

##### Propagation and reinforcement of signal.

The same thresholding approach, as discussed for Markov chains, turns out to be applicable to a wider class of signal propagation dynamics. It consists in first obtaining a positive-valued signal with heavy random noise, then applying a negative bias, and finally using the ReLU gate to act as a noise threshold.

Any linear function $G^{\prime}$ can be represented with a hidden layer of $s\leq n^{2}$ nodes, through two matrices $D^{\prime}\in R^{+n\times s}$ and $E^{\prime}\in R^{+n\times s}$, such that:

$$ $G^{\prime}=D^{\prime}E^{\prime}.$ $$

The above holds in general, and we will refer to such a representation of $G^{\prime}$ as having a sparse hidden (synaptic) layer. We will consider now the question of expressing non-negative functions, $G^{\prime}\in(R^{+})^{n\times n}$. An example of a valid representation of $G^{\prime}$ is given through a node-edge incidence representation, $D^{\prime}_{i,(i-1)n+j}=E^{\prime}_{(i-1)n+j,j}=\sqrt{G^{\prime}_{ij}}$, but usually this representation is not optimal in terms of the number of non-zero entries of $D^{\prime}$ and $E^{\prime}$.

In general, any low-rank approximation of $G$ can be equivalently expressed as $G\approx D^{\prime}P_{D}P_{E}^{T}E^{\prime}$, for some two matrices, $P_{D},P_{E}\in R^{s\times d}$. We will consider the most common class of low-rank approximations obtained by taking $P_{D}=P_{E}=P\sim\mathcal{N}(0,1)^{s\times d}/\sqrt{d}$. Consider a vector $z$ passing through the ReLU-lowrank operation, and the following vectors $u\in R^{s}$, $w\in R^{n}$:

$$ $\displaystyle u$ $\displaystyle:=E^{\prime}z$ $\displaystyle w$ $\displaystyle:=D^{\prime}PP^{T}u$ $$

If $v_{i}^{T}z$ has the interpretation of a signal being sent by node $v_{i}$, then $u$ is the encoded message being passed through the hidden layer of the network, and $v_{j}^{T}w$ is the message received by node $v_{j}$.

### 5.4 Modularity in BDH-GPU signal propagation

We are now ready to capture the essence of the signal propagation and reinforcement capability of the ReLU-lowrank system. To describe the conditions under which a neuron is able to decide whether it should, or should not activate. By a standard analysis of independent Gaussians, we have the following probabilistic statement, under random choice of $P$.

###### Claim 6 (selective neuron activation) .

Suppose that the signal of $u$ is uniformly concentrated on a set of nodes $A$ of the hidden layer, i.e., for some subset $A$ of indexes of the hidden layer, we have $u_{\bar{\alpha}}=0$ for $\bar{\alpha}\not\in A$, and $u_{\bar{\alpha}}\in[\frac{1-\kappa}{\sqrt{|A|}},\frac{1+\kappa}{\sqrt{|A|}}]$ for $\bar{\alpha}\in A$, so that $\|u\|_{2}\in[1-\kappa,1+\kappa]$ for some small constant $\kappa\geq 0$. Suppose each node $v_{j}\in V$ is connected in $D^{\prime}$ to some set of nodes $B_{j}$ in the hidden layer, $B_{j}=\{\bar{\beta}:D^{\prime}_{j,\bar{\beta}}\neq 0\}$, and let these connections weight be drawn uniformly $D^{\prime}_{j,\bar{\beta}}\in[\frac{1-\kappa}{\sqrt{|B_{j}|}},\frac{1+\kappa}{\sqrt{|B_{j}|}}]$ for $\bar{\beta}\in B_{j}$. Let $C_{j}=A\cap B_{j}$. Define the ratio:

$$ $\varrho:=\left.\sqrt{\frac{|C_{j}|}{|A|}\cdot\frac{|C_{j}|}{|B_{j}|}}.\right.$ $$

Then, there exists an absolute constant $c>0$, such that for any value of $w_{j}$ (where we recall that $w:=D^{\prime}PP^{T}u$), we have:

$$ $\Pr\left[w_{j}\geq(1-\kappa)^{2}\varrho-c\sqrt{\log n\ /\ d}\right]=1-O(1/n)\quad\textrm{and}\quad\Pr\left[w_{j}\leq(1+\kappa)^{2}\varrho+c\sqrt{\log n\ /\ d}\right]=1-O(1/n).$ (12) $$

Thus, the value of $w_{j}$ can be used by a neuron to obtain an estimation of $\varrho$, and apply a threshold to activate accordingly.
∎

###### Proof.

Observe that $w_{j}={(P^{T}D^{\prime}_{j,\cdot})}^{T}{(P^{T}u)}$. As $\|D^{\prime}_{j,\cdot}\|_{2}\in[1-\kappa,1+\kappa]$, a standard application of Johnson-Lindenstrauss to vector inner products gives
$\Pr\left[|w_{j}-{D^{\prime}_{j,\cdot}}^{T}{u}|\leq c\sqrt{\log n\ /\ d}\right]=1-O(1/n)$ for $c$ large enough. Since ${D^{\prime}_{j,\cdot}}^{T}{u}\in[(1-\kappa)^{2}\rho,(1+\kappa)^{2}\rho]$, the claim follows.
∎

Figure: Figure 8: The ReLU-lowrank feedforward network of BDH-GPU allows neurons to activate when triggered by activation signals in its own community. (a) Illustration of the selective neuron activation pattern in the proof of Claim [6](#Thmclaim6), showing the activation decision of node $v_{j}$ (left) based on active set $A$ in the sparse hidden layer. (b) Illustration of Eq. ([12](#S5.E12)) showing the relationship between sizes of sets in the sparse hidden layer: active set $A$, set $B_{j}$ connected to neuron $v_{j}$, and the intersection $C_{j}=A\cap B_{j}$: neuron $v_{j_{1}}$ becomes active, but neuron $v_{j_{2}}$ does not.
Refer to caption: https://arxiv.org/html/2509.26507/2509.26507v1/fig2.png

The ReLU-lowrank operation $f_{DE}$, after adding appropriate negative bias, can thus be used to propagate positive affinity functions $G^{\prime}$ on input vectors, performing the following form of thresholding: neurons $j$ in the output layer individually compute a form of local “F-score” $\varrho$ given by Eq. ([12](#S5.E12)) of the activation of the positive sparse hidden layer, and decide based on it whether they are good match for the output activation; if the threshold condition on $\varrho$ is not met, the neuron $j$ does not activate in the output vector (see Fig. [8](#S5.F8) for an illustration).

Equation ([12](#S5.E12)) naturally coincides with a pattern of communication within network graphs $G^{\prime}$ admitting positive Newman modularity , allowing nodes $v_{j}$ to correctly receive messages $u$ which in the hidden layer primarily reached a denser cluster of $G^{\prime}$ containing $v_{j}$. For a specific illustration, let $H$ be an undirected $k$-block stochastic block model (SBM) network  with $k\in\mathbb{N}$ blocks of $n/k$ nodes each, in-block edge density $p$ and out-of-block edge density $q<p$. We put $G^{\prime}:=D^{\prime}E^{\prime}=H^{2}$, i.e., the first connection layer of $G^{\prime}$ is $E^{\prime}=H$ and the second connection layer is also $D^{\prime}=H$. Suppose that $H$ is a random SBM graph with positive Newman modularity separated from $0$, i.e., let $\mu=\frac{k-1}{k}\frac{p-q}{p+(k-1)q}>0$. Following Claim ([6](#Thmclaim6)) with $\kappa=0$, we can find a ReLU-lowrank representation $G$ to achieve a communication scheme on $G^{\prime}$, such that a message sent from one node $z=v_{i}$ activates a node $v_{j}$ when $i$ and $j$ are in the same block with probability $1-O(1/n)$, and with probability $O(1/n)$ otherwise, when $\mu>\frac{1}{p}\sqrt{\log n\ /\ d}$.

We thus make the following intuitive observation.

###### Observation 6 (in-cluster signal reinforcement) .

The ReLU-lowrank representation of $\textrm{BDH-GPU}(n,d)$ is sufficient to represent in-cluster information spreading dynamics in models of graphs with constant in-cluster density and arbitrarily small positive modularity (such as the $k$-cluster Stochastic Block Model) when $d/\log n=\omega(1)$ is an arbitrarily slowly growing function.

While Claim [6](#Thmclaim6) and Observation [6](#Thmobservation6) are made with reference to an almost-uniform distribution of signal $u$ on the set of nodes of the middle layer, $u$ can have (and in practice does have) a distribution of density which is non-uniform, e.g., going across $a=O(\log n)$ different clustering scales, with a $(1/a)$-fraction of the signal represented at each scale. This allows neurons in the output layer to combine a smaller number of strong signals in its local cluster, with a larger number of weaker ones spread more globally. Such an approach coincides with the observed structure of the graph $D^{\prime}E^{\prime}$, discussed in Subsection [5.5](#S5.SS5).

###### Claim 6 (selective neuron activation) .

###### Proof.

###### Observation 6 (in-cluster signal reinforcement) .

##### Supermodularity on input perturbation.

We clarify how the properties of function $f_{DE}:(R^{+})^{n}\to(R^{+})^{n}$ relate to the previously discussed ability to make an input signal resonate “within a module” in a graph with hidden modular structure. First, note that $f_{DE}$ *is a subadditive function, but is not submodular in general* with respect to the set of $n$ coordinates of its input vector. In some of the regimes in which it appears to be operating, locally $f_{DE}$ exhibits a form of behavior opposite to submodularity, referred to as ‘supermodularity’, or ‘increasing returns’ of adding new coordinates to the input vector. This is already implicitly captured by Claim [6](#Thmclaim6), but we can consider a simpler example.

Take a variant of the setting from Observation [5](#Thmclaim5) with the same choice of $G^{\prime}$, and let $z\in(R^{+})^{n}$ and biases of $D$ be chosen so that all coordinates of $DEz$ are approximately equal to $-1.5/r\pm o(1)$ (this can be done by choosing e.g. $z_{j}=1/n$). Then, $f_{DE}(z)=0$, and for any $v_{i},v_{j}\in V$, $f_{DE}(z+v_{i})=0$ a.s., $f_{DE}(z+v_{j})=0$ a.s., but $f_{DE}(z+v_{i}+v_{j})$ has non-zero coordinates a.s. with values approximately $1/{2r}$, for all nodes $v_{k}$ which are common out-neighbor nodes of $v_{i}$ and $v_{j}$, i.e., for all $k$ such that $G^{\prime}(v_{i},v_{k})=G^{\prime}(v_{j},v_{k})=1/r$. This mechanism generalizes to finding common neighborhoods which have many connections to two given subsets of nodes, $V_{a}$ and $V_{b}$. In a setting where the considered affinity $G^{\prime}$ is bi-directional (e.g., a symmetric matrix), this corresponds to finding shortcut nodes, allowing to go from $V_{a}$ to $V_{b}$.

It follows that the neighborhood-reinforcing nature of the threshold dynamics of BDH-GPU, which plausibly follows from the logic of its role in inference and from the needs for an efficient computational process, is starkly different from the more often studied submodular behavior of threshold and cascade dynamics on real-world networks , and plausibly, much less smooth when considered as a dynamical process.

### 5.5 Empirical findings: parameter distribution in ReLU-lowrank matrix products

We consider the $D$ matrices (in the same way ${D_{y}}$ and ${D_{x}}$) and $E$ matrix obtained after training of BDH-GPU models, and used in the ReLU-lowrank operation Eq. ([10](#S5.E10)), $f_{DE}(z)=\left(DEz\right)^{+}$.

##### Choice of prior of matrix parameter distributions.

Following the discussion in Section [5.4](#S5.SS4), we expect matrix $G:=DE$ to reflect the clustering (modularity) structure of the neuron-neuron communication graph. Any plausible parameter distribution of matrix $G$ must therefore allow heavy-tailed distribution of entries. At the same time, a Gaussian noise term is inherent to low-rank matrix representation, and needs to be taken into account together with this heavy-tailed distribution.

We now provide a somewhat more fine-grained explanation, which leads to the prior on the structure of matrix $G$ as given by Eq. ([13](#S5.E13)). Consider a training set-up in which the ReLU-lowrank operation described by matrix $G$ is treated as an approximation of the same operation, governed by a high-rank matrix $G^{\prime}$, with $f^{\prime}(z):=\left(G^{\prime}z\right)^{+}$. Considering this block in isolation from the rest of the training system, the training of matrices $D$, $E$ goal corresponds to learning an approximation of $f^{\prime}$, with $D\in R^{n,d},E\in R^{n,d}$, such that $f(z)\approx f^{\prime}(z)$ holds for some class of vectors $z$.

For the rest of this analysis, we will consider the function $f^{\prime}$ as a ground truth reference for the intended operation of the ReLU-lowrank block. This type of analysis can be seen as plausible over short time spans in later phases of training of a BDH-GPU model, i.e., once individual neurons in $R^{n}$ have started to admit semantic or functional meaning, and so when function $D^{\prime}E^{\prime}$ describes a property of the problem being solved in a (frozen) concept space, and not a co-learning process between the representation of the concept space in $R^{n}$ and the functions applied to it.

We can represent $G^{\prime}:=D^{\prime}E^{\prime}$, where $D^{\prime}\in R^{n,s}$, $E^{\prime}\in R^{s,n}$, with $s=O(n^{2})$, are in general matrices of rank $n$; we have $f^{\prime}(z):=\left(D^{\prime}E^{\prime}z\right)^{+}$. Without loss of generality, we can choose from among the possible representations one with the following distribution of positive and negative elements: $D^{\prime}\in(R^{+})^{n,s}$, $E^{\prime}={E^{\prime}}^{\mathfrak{e}}-{E^{\prime}}^{\mathfrak{i}}$, with ${E^{\prime}}^{\mathfrak{e}},{E^{\prime}}^{\mathfrak{i}}\in(R^{+})^{s,n}$. We will write: $G^{\prime}={G^{\prime}}^{\mathfrak{e}}-{G^{\prime}}^{\mathfrak{i}}$, where ${G^{\prime}}^{\mathfrak{e}}=D^{\prime}{E^{\prime}}^{\mathfrak{e}}$, and ${G^{\prime}}^{\mathfrak{i}}=D^{\prime}{E^{\prime}}^{\mathfrak{i}}$. The main purpose of the chosen representation $G^{\prime}=D^{\prime}E^{\prime}=D^{\prime}({E^{\prime}}^{\mathfrak{e}}-{E^{\prime}}^{\mathfrak{i}})$ is to have matrices $D^{\prime}$, ${E^{\prime}}^{\mathfrak{i}}$, ${E^{\prime}}^{\mathfrak{e}}$ with much smaller outlying elements compared to matrix $G^{\prime}$, which leads to more justified conclusions about the uniform nature of the noise introduced by the low-rank decomposition.(^15^15 15 For a specific example, one very broad class of matrices $G^{\prime}$ is given by the product of sparse matrices $D^{\prime}$, ${E^{\prime}}$, in which each $s$-element row of $D^{\prime}$ (column of ${E^{\prime}}$) has at most $\Delta\ll n$ non-zero elements, each with value bounded by $O(1/\sqrt{\Delta})$, and all remaining $s-\Delta$ elements of these matrices are equal to $0$. The resulting elements, ${G^{\prime}}_{i}j=\sum_{\alpha}D^{\prime}_{i,\alpha}\{E^{\prime}\}_{\alpha,j}$, may be much less uniform, only satisfying ${G^{\prime}}_{i}j=O(1)$. This type of scenario captures the expressiveness of set intersection for “bag-of-words” models for language, or expressiveness of “hub label” representations for a measure of node proximity in a directed graph.)

Assume now that we learn to approximate function $f^{\prime}$ with $f_{DE}$ by trainable matrices $D,E$ through the following low-rank scheme:

$$ $G=DE:=(B_{D}+D^{\prime}P)(P^{T}E^{\prime}+B_{E}^{T}),$ $$

where $P\in R^{s,d}$ is *non-parametric* and the result of random sampling an almost-orthonormal random projection so that $PP^{T}\approx I_{s}$ (e.g. $P\sim\mathcal{N}(0,1/\sqrt{d})^{s,d}$), and $B_{D},B_{E}\in R^{n,d}$ represent *trainable* additional terms for compensating error or introducing bias, with the goal of minimizing some loss function $\mathcal{L}(f^{\prime},f_{DE})$. The terms $B_{D},B_{E}$ compensate the error introduced by the approximation $PP^{T}\approx I_{s}$, after the ReLU operation.

Let $Q:=PP^{T}=I_{s}+\delta_{I}+\delta_{Q}$, where $\delta_{I}\in R^{s\times s}$ is a diagonal error matrix, and $\delta_{Q}\in R^{s\times s}$ is a non-diagonal (hollow) matrix. We have:

$$ $G=DE=({G^{\prime}}^{\mathfrak{e}}-{G^{\prime}}^{\mathfrak{i}})+D^{\prime}\delta_{I}({E^{\prime}}^{\mathfrak{e}}-{E^{\prime}}^{\mathfrak{i}})+\underbrace{D^{\prime}\delta_{Q}E^{\prime}}_{\varepsilon_{Q}}+\underbrace{(B_{D}E+DB_{E}^{T})}_{\varepsilon_{B}}.$ $$

Since all elements of $D^{\prime},{E^{\prime}}^{\mathfrak{e}},{E^{\prime}}^{\mathfrak{i}}$ are non-negative and $I_{\delta}$ is diagonal, we can represent elements $G_{ij}$, for $i,j\in 1,\ldots,n$, as follows:

$$ $G_{ij}=(1+\varepsilon_{\delta\,ij}^{\mathfrak{e}}){G^{\prime}}^{\mathfrak{e}}_{ij}-(1+\varepsilon_{\delta\,ij}^{\mathfrak{i}}){G^{\prime}}^{\mathfrak{i}}_{ij}+\varepsilon_{Q\,ij}+\varepsilon_{B\,ij}$ (13) $$

where $|\varepsilon_{\delta\,ij}^{\mathfrak{e}}|=O(\sqrt{\log n\ /\ d})$ and $|\varepsilon_{\delta\,ij}^{\mathfrak{i}}|=O(\sqrt{\log n\ /\ d})$ have the interpretation of small multiplicative distortion.

Following ([13](#S5.E13)), we expect the elements of $G$ to be distributed as the sum of four different distributions.
The term ${G^{\prime}}^{\mathfrak{e}}_{ij}$ has the interpretation of positive ground truth elements of $G^{\prime}$. The term $-{G^{\prime}}^{\mathfrak{i}}_{ij}$ has the interpretation of negative ground truth elements of $G^{\prime}$; its use in combination with the ReLU mechanism can be interpreted as inhibitory action. Both of these terms are subject to slight multiplicative distortion.

The term $\varepsilon_{Q\,ij}$ has the interpretation of non-trainable noise (which depends only on $D^{\prime}$, $E^{\prime}$ and the random choice of $P$). Under reasonable assumptions on outlying elements of $D^{\prime},E^{\prime}$, it is a form of almost-Gaussian symmetric noise inherent to the considered class of low-rank projections, $\varepsilon_{Q\,ij}\rightarrow N(0,\sigma_{Q})$, for some $\sigma_{Q}\in R^{+}$, and the expected value of this noise is typically very close to $0$, even when considering the expectation of $\varepsilon_{Q\,ij}$ conditioned on known values of $\varepsilon_{Q\,i^{\prime}j^{\prime}}$ for a small number of indexes $(i^{\prime},j^{\prime})$ in the matrix.

Finally, $\varepsilon_{B\,ij}$ is a trainable term, whose norm tends to $0$ as $d$ increases. We expect it to have the interpretation of bias used to offset the low-rank Gaussian noise and perform denoising in the ReLU-gate, as previously discussed in Section [5.3](#S5.SS3). Because of the action of the ReLU gate, we plausibly expect the distribution of $\varepsilon_{B\,ij}$ to be skewed towards negative numbers, with $0>\mathbb{E}\varepsilon_{B\,ij}\gg\sigma_{Q}$.

From the above discussion of the four terms of the sum in Eq. ([13](#S5.E13)), we see that only one of these terms, ${G^{\prime}}^{\mathfrak{e}}_{ij}$, is expected to take values much larger than $\sigma_{Q}$ with non-negligible probability. We reach the conclusion that a part of the relevant signal of $G^{\prime}$ is concentrated in the right tail of large positive matrix entries of $G$.

###### Hypothesis 1 (right tail contains signal) .

Consider the interpretation that the ReLU-lowrank transformation $z\mapsto\left(Gz\right)^{+}$, with $G=DE$, has learned to act as an approximation of some other operation $z\mapsto\left(G^{\prime}z\right)^{+}$, where $G^{\prime}$ has no low-rank constraint imposed on it. Then the right tail of the distribution of matrix elements of $G$ corresponds to the right tail of the distribution of elements of $G^{\prime}$, starting from some positive threshold value $\sigma_{Q}$, associated with the noise of the low-rank decomposition. Formally, for almost all pairs of indices $i,j\in 1,\ldots,n$ such that $G_{ij}\gg\sigma_{Q}$, we also have ${G^{\prime}}^{\mathfrak{e}}_{ij}\gg\sigma_{Q}$.

The converse implication, that ${G^{\prime}}^{\mathfrak{e}}_{ij}\gg\sigma_{Q}$ implies ${G}_{ij}\gg\sigma_{Q}$, also plausibly holds under some stronger assumptions on the form of biases $\varepsilon_{B\,ij}$ which may follow from minimizing training error for the specific inference task considered.

This direct method of decoding $G$ from $G^{\prime}$ does not extend from the right tail towards the center of the distribution. For the choices of $n,d$ we make, we expect the term dominating most elements of matrix $G$ to be $\varepsilon_{Q\,ij}$. For example, when $G^{\prime}$ is a stochastic matrix, we expect to have $\sigma_{Q}=O(1/\sqrt{d})$ (cf. Eq. ([11](#S5.E11)) for the corresponding infinity-norm bound, $|\varepsilon_{Q\,ij}|=O(\sqrt{\log n\ /\ d})$). With $\sum_{i,j}|{G^{\prime}}^{\mathfrak{e}}_{i,j}|=n$ for a stochastic matrix,
we expect the right heavy tail of the element distribution of $G$ to have $\Omega(n\sqrt{d})$ elements (out of the $n^{2}$ matrix elements of $G$) which are clearly separated from the Gaussian noise.

We confirm empirically that the right tail of $G$, defined as above with respect to threshold $\sigma_{Q}$, turns out to contain a non-negligible portion of the parameter capacity of matrices $D$, $E$, even for very small models (10M to 100M parameters).

###### Hypothesis 1 (right tail contains signal) .

##### Experimental setup.

We prepared parameter matrices of a 24M-parameter BDH-GPU model configured with $h=4$ heads and $L=8$ layers, $n=h\cdot 2^{13}=2^{15}$ neurons, and hidden low-rank dimension $d=256$. We considered the weighted neuron-neuron interaction graph, having the encoder-decoder matrix pair $G={D_{x}}E$ as its node adjacency matrix on the set of neurons $V=1,\ldots,n$. For uniformity, we subsampled $G$ by picking node subsets $V^{(a)}$, $a\in\{1,2,3,4\}$, associated with each head, and considered the weighted subgraphs $G^{(ab)}=\{V,\{(u,v,G_{uv}):u\in V^{(a)},v\in V^{(b)}\}\}$, with $G^{(ab)}\in R^{n^{*}\times n^{*}}$ where $n^{*}=n/h=2^{13}$, each having $(n^{*})^{2}=2^{26}$ weighted edges.

We repeated the experiment $5$ times using models pretrained with different random seeds.

##### Findings.

For all of the $5$ models we pretrained for this purpose, exactly $3$ out of the $4$ encoder heads and all decoder heads adhered to the prior on parameter distribution given by Eq. ([13](#S5.E13)), showing a good correspondence for $12$ out of $16$ of their parameter sub-matrices $G^{(ab)}$.

We continue the discussion in this Section for one specific matrix $G^{(ab)}$ of one specific pretrained models, which was chosen as representative. The example we choose has $a=b$; and so the matrix $G^{(ab)}$ has an interpretation as $G[V_{a}]$, i.e., the subgraph of $G$ induced by vertex set $V_{a}$, which enables us to visualize the graph $G^{(aa)}$ more easily on its vertex set $V_{a}$.

We refer to the representative object of our study, i.e., to the matrix $G^{(aa)}$ of the selected model, as $G^{*}$. For any matrix $A$ and $\beta\geq 0$, we denote by $A_{\geq\beta}$ the matrix $A$ cut off at threshold $\beta$, i.e., ${A_{\geq\beta}}_{ij}=A_{ij}$ if $A_{ij}\geq\beta$, and ${A_{\geq\beta}}_{ij}=0$ otherwise.

The distribution of elements $G^{*}_{i,j}$ is presented in Fig. [9](#S5.F9) (a).

Figure: Figure 9: (a) Heavy-tailed element distribution and modularity analysis of the excitatory neuron-neuron connection graph contained the encoder-decoder matrix $G^{*}$. Distribution of elements of the encoder-decoder matrix $G^{*}\in R^{n^{*}\times n^{*}}$ of a BDH-GPU model with $n^{*}=8192$ neurons and $d=256$: histogram $\mathrm{freq_{G^{*}}}(x)$, its symmetric part $\mathrm{freq-symmetric_{G^{*}}}(x):=\min\{\mathrm{freq_{G^{*}}}(x),\mathrm{freq_{G^{*}}}(-x)\}$, and distribution skew $\mathrm{freq-skew_{G^{*}}}(x):=\mathrm{freq_{G^{*}}}(x)-\mathrm{freq-symmetric_{G^{*}}}(x)$. $\diamond$ (b) Estimate (lower bound) of Newman modularity of matrix $G^{*}_{\geq\beta}$ for different values of $\beta$, plotted as a function of the number of non-zero elements (edges) of $G^{*}_{\geq\beta}$. Modularity of random graph baselines are provided for reference, for the $G(n^{*},m)$ model with the same number of edges as $G^{*}_{\geq\beta}$, and for a matrix $(P_{1}P_{2}^{T})_{\geq\beta^{\prime}}$ with the same number of edges as $G^{*}_{\geq\beta}$, where $P_{1},P_{2}\sim\mathcal{N}(0,1)^{n^{*}\times d}$. The modularity estimates were obtained using the community structures returned by the Louvain algorithm, in the best of 5 clustering runs with different random seeds.
Refer to caption: https://arxiv.org/html/2509.26507/2509.26507v1/fig4.png

We find that the observed distribution corresponds well to the prior expected of it by Eq. ([13](#S5.E13)). We determine the threshold value $\beta\geq 0$ at which we expect to capture signal, ${G^{*}_{\geq\beta}}\approx{G^{\prime}_{\geq\beta}}$, following Hypothesis [1](#Thmhypothesis1). We find (from Fig. [9](#S5.F9)(a)) that the separation from noise happens for this specific matrix $G^{*}$ at $\beta_{1}\approx 1.2$, at which point the right heavy tail begins to dominate. However, already for much smaller values of $\beta$ we find that ${G^{*}_{\geq\beta}}$ has high modularity, and this actually increases as more non-zero values are added to ${G^{*}_{\geq\beta}}$ for smaller $\beta$, up to a maximum at $\beta_{2}\approx 1.0$ (Fig. [9](#S5.F9)(b)). Even for much smaller values of $\beta$, the modularity of ${G^{*}_{\geq\beta}}$ remains almost constant up to well above $2^{20}$ non-zero matrix entries on the $n^{*}=2^{13}$ nodes considered. The modularity of the baselines, of random graphs or random low-rank matrix products, quickly drops to 0 in this regime. This should be compared to the total number of parameters of the matrices ${D_{x}},E$ corresponding to $G^{*}$, i.e., $2\cdot 2^{13}\cdot 2^{8}=2^{22}$ parameters. A complementary analysis of the inhibitory signal, for a similarly defined matrix $|G^{*}_{\leq-\beta}|$, also finds that this structure has high modularity.

In auxiliary experiments, we looked at basic graph parameters of matrix $G^{*}_{\geq\beta}$, treated as a directed graph on its set of nodes. We set $\beta=1.2$, obtaining $m=46820$ non-zero entries (edges) in $G^{*}_{\geq\beta}$.
We found that $G^{*}$ has a heavy-tailed, power-law-like degree distribution, with generally more concentrated out-degree than in-degree (Fig. [10](#S5.F10)(a)).

Figure: Figure 10: (a) Unweighted in-degree and out-degree distribution for the $n^{*}=8192$ neuron nodes and $m=46820$ edges of matrix $G^{*}_{\geq\beta}$ with $\beta=1.2$. The distributions exhibit power law distributions, with different exponents, the out-degree distribution being more concentrated. (b) Visualization of graph $G^{*}_{\geq\beta}$, hinting at its core-periphery structure.
Refer to caption: https://arxiv.org/html/2509.26507/2509.26507v1/fig5.png

Generally, this finding is consistent with expectations as to the structure of a network with positive modularity. The difference of in- and out-degree distributions, while plausible and prevalent in real-world information dissemination networks, was not considered in Subsection [5.3](#S5.SS3).

Finally, a visualization of $G^{*}_{\geq\beta}$ (Fig. [10](#S5.F10)(b)) exhibits a core-periphery structure. This is again consistent with the expected modular structure.

###### Empirical Finding 3 .

We confirmed that during training, *a graph structure with positive modularity* appears in BDH-GPU model parameter matrices ${D_{x}}E$ and ${D_{y}}E$. This modular structure plausibly follows from the network’s inference function, and specifically from the cluster-aware information propagation dynamics supported by the ReLU-lowrank mechanism (Observation [6](#Thmobservation6)).

We also observed that for all of the studied models with $h=4$ heads, $1$ encoder sub-matrix out of $4$ has no heavy positive tail, and generally appears to capture a form of inhibitory structure ${G^{\prime}}^{\mathfrak{i}}$ from Eq. ([13](#S5.E13)). Since we have not provided convincing mechanisms for isolating negative signals in $G^{*}$ and these are easily confounded with the bias term $\varepsilon_{B}$, we omit this case from discussion. We remark that the apparent need for passing activations through such a separate “inhibitory circuit” is one of the most evident explanations for why introducing (a small number of) heads to BDH-GPU provides an improvement in model quality.

###### Empirical Finding 3 .

## 6 Analysis: linear attention, sparse positive activation, and monosemanticity

### 6.1 Macro-expressiveness of attention in BDH-GPU

The attention mechanism of BDH-GPU can be described at a coarse-grained level as a transformation mechanism for key-query-value vectors, similar to that in the Transformer. This description is complementary to the interpretation of the BDH-GPU attention mechanism at the micro-level of correlations between neuron pairs, which we defer to Section [6.2](#S6.SS2), which provides more insight into the way activation vectors used by BDH-GPU relate to the concept space of the model.

We compare the attention mechanism of BDH-GPU with the attention mechanism of the Transformer, describing both as reflections of a general attention mechanism. Specifically, we explain why, and up to what context length, the linear attention mechanism of BDH-GPU plausibly fits into macro-expressiveness frameworks of attention designed for the Transformer (based on RASP).

##### Basic properties of BDH-GPU attention.

The key-query space for BDH-GPU is $R^{n}$, the same as its neuron dimension, rather than the small dense dimension used by the Transformer. The keys and queries used by BDH-GPU are given by positive vectors, in $(R^{+})^{n}$, and are expressed by the same vector $x_{t,l}$ (noting that at time $t$, $x_{t,l}$ is used as a query, and only $x_{\tau,l}$, for $\tau\leq t-1$, are used as keys).

‘Value’ vectors of BDH-GPU remain in the small dimension, $R^{d}$, which at some model scales is comparable to the dimension used for attention ‘values’ in common configurations of the Transformer.

The relationship between softmax-based attention of the Transformer, regarded as a low-dimensional kernel for general linear attention, and linear attention for vectors in the positive orthant, was considered in a framework called FAVOR+ . Here, we provide a few complementary (simpler) observations, sufficient to grasp the main effects of the ability of Linear Attention to distinguish facts in context.

##### State capacity vs. distinction capacity.

The matrix ${\boldsymbol{\rho}}\in R^{n\times d}$, which is used to represent state for each layer of BDH-GPU, should theoretically have sufficient capacity to store $O(n)$ ‘value’ vectors in $R^{d}$ if considered as a lookup table for values. We now remark that its actual capability of *distinguishing facts* using the linear attention mechanism is also asymptotically close to $n$.

Attention is a mechanism of associative memory which, given a series of key-value pairs $((k_{1},v_{1})\ldots,(k_{t},v_{t}))\in(\Lambda_{k}\times R^{d})^{t}$, a query $q\in\Lambda_{q}$ and an affinity function $\phi(\cdot,\cdot):\Lambda_{q}\times\Lambda_{k}\to[0,1]$ between the space of queries and keys, returns the attention value:
$a_{t}=\sum_{\tau=1}^{t-1}\phi(q,k_{\tau})v_{\tau}$
(or a normalization thereof). With BDH-GPU, we consider ‘value’ vectors $v\in R^{d}$, where $d$ is small. The spaces of keys and queries may be assumed to coincide as $\Lambda=\Lambda_{k}=\Lambda_{q}$, and we consider in general a single key-query sequence, given by $(k_{t})_{t\in\mathbb{N}}$:(^16^16 16 This assumption is known to have moderate practical implications for trainability. In this specific discussion, it is ‘without loss of generality’, since one can consider $\Lambda=\Lambda_{1}\otimes\Lambda_{2}\otimes\ldots\otimes\Lambda_{t}$, and consider each $k_{i}$ as chosen from $\Lambda_{i}$, defining affinity $\phi(k_{t},k_{\tau}):\Lambda_{t}\times\Lambda_{\tau}\to[0,1]$ appropriately to handle successive keys and queries (effectively describing a general form of positional embedding).)

$$ $a_{t}=\sum_{\tau=1}^{t-1}\phi(k_{t},k_{\tau})v_{\tau}$ (14) $$

This key-query space $\Lambda$ may be considered as an abstract space, and represented in any way which is convenient, for as long as the affinity function $\phi(k_{t},k_{\tau})$ is preserved. For example, when the keys and queries are sampled from a finite (though possibly extremely large) set, there also exists some vector space dimension $\nu$ (possibly extremely large) and a function mapping $f:\Lambda\to S^{\nu}$, where $S^{\nu}=\{z\in R^{\nu}:\|z\|=1\}$ is the unit sphere, such that the scalar (dot, cosine) product in $S^{\nu}$ satisfies $f(k_{t})\cdot f(k_{\tau})=\phi(k_{t},k_{\tau})$. In other words, any affinity function $\phi$ becomes linear when represented in sufficiently high dimension, subject to suitable preparation of its arguments with function $f$. With $\nu$ extremely large, $S^{\nu}$ is a sort of Platonic ideal of a space in which the attention keys and queries live, with no relation to any specific model.

This type of argument, often used in considerations of Support Vector Machines, is linked to two challenges: (1) ensuring that the dimension actually considered by the network (in our case $n$) is high enough compared to the hypothetical dimension $(\nu)$, and (2) ensuring that a suitable preparation function $f$ exists and is easy to learn for the model.(^17^17 17 The Transformer can also be positioned in the same SVM framework: the Transformer’s attention represents a form of “kernel trick” for one specific affinity function $\phi$, with the kernel used to approximate it being the exponential function (in the case of softmax attention).) We now explain when the dimension $n$ can be considered sufficient, and what types of keys can be prepared by BDH-GPU.

##### Expressiveness of linear attention in dimension n n .

The Linear Attention mechanism aggregates key-value correlations over time. In general, the associated rate of accumulation of noise is manageable, up to the approximate scale of between $t=\tilde{\Omega}(\sqrt{n})$ and $t=\tilde{O}(n)$ key-value ‘facts’ stored in the attention of a given layer. We make the following statement about the Linear Attention mechanism in general.

###### Claim 7 (informal statement) .

The mechanism of Linear Attention, applied in dimension $R^{n}$, can approximately express an attention affinity function for up to $t=\tilde{O}(n)$ ‘key-value’ pairs in context, with ‘values’ having comparable L2-norm, under moderate assumptions on weak correlation of historical keys and uniformity of the expressed affinity function. Without such assumptions, Linear Attention can compute the correct affinity up to at least $t=\tilde{\Omega}(\sqrt{n})$ ‘key-value’ pairs in context, except for a negligible fraction of possible inputs. Keys and queries need to be suitably prepared beforehand.

*The formal statement and proof is provided in Appendix [C.2](#A3.SS2).*∎

The above claim captures the expressiveness of Linear Attention in dimension $R^{n}$, subject to some way of preparing keys and queries in $R^{n}$ by the model in blocks preceding the attention block. A model using Linear Attention has to learn its own way to prepare keys. In fact, different *natural* approaches to key preparation, for example using random projections or hashing on a set of $t$ vectors, lead to the same asymptotic statement of Claim [7](#Thmclaim7). (In the proof in the Appendix, we chose to use a particularly simple one.)

The specific way of preparing keys used (learned) by BDH-GPU for its Linear Attention is particularly interesting. Except for the effect of RoPE rotation, which introduces a negative positional effect in the affinity of keys and queries, BDH-GPU uses activation vectors (keys, queries) with only positive coordinates to represent its keys.

We discuss some aspects of how the positive activation vectors of BDH-GPU relate to Linear Attention.

###### Claim 7 (informal statement) .

##### Preparation of positive keys for Linear Attention.

Activation vectors of BDH-GPU belong to the positive orthant, and are often sparse. The interpretation of such vectors depends on whether we consider the positive orthant to be a “valid shape” for the latent concept space of the considered task (in this case, language and reasoning), or whether the task has to be embedded into such a space. For language, this would be a question of whether a word2vec-like internal representation of the concept space by the model has an inherent advantage over a
bag-of-words-like representation, especially when expressing concept affinities in attention.

We note that latent representation of key and query vectors in the positive orthant is natural for any problem which is *amenable to attention*. In the discussion of general attention given by Eq. ([14](#S6.E14)), we noted that the affinity function $\phi$ takes values in $[0,1]$, and we considered an embedding $f$ of a set of key vectors $k_{1},\ldots,k_{t}$ into $R^{\nu}$ such that $f(k_{t})\cdot f(k_{\tau})=\phi(k_{t},k_{\tau})\geq 0$. Given this condition on non-negativity of dot product on all pairs among the $t$ vectors considered, we could have, without loss of generality, used an appropriately rotated embedding $f$ so that $f(k_{\tau})\in(R^{+})^{\nu}$, thus *directly reducing the problem of general attention to a problem of linear attention in the non-negative orthant*. The question which remains is a subtle one: whether this type of embedding of the latent space of language and reasoning in $(R^{+})^{\nu}$ is ‘natural’, i.e., preserved over long periods of time of inference and training, notably longer than the short window $t$ of context used for Transformer-like attention.

In the rest of the paper, we are generally inclined to assume that representations in $(R^{+})^{\nu}$ of concepts, combinations of concepts, and density distributions over such combinations of concepts, are universal to language and reasoning.

We limit ourselves to a very brief discussion of a way to represent attention keys with positive vectors for problems for which such a concept representation is not natural.

###### Using LSH to move key vectors into the positive orthant.

Locality Sensitive Hashing (LSH) is one technique for converting arbitrary vectors in a lower-dimensional space $R^{a}$, for some fixed $a\in\mathbb{N}$, into vectors in $(R^{+})^{n}$, in a way which can be used to describe certain ‘sharp-boundary’ affinity functions $\phi$ in $R^{a}$. Consider an $n\times a$ matrix represented as $n$ fixed random vectors $\lambda_{1},\ldots,\lambda_{n}\in R^{a}$, and a corresponding sequence of $n$ appropriately chosen gating functions $\gamma_{1},\ldots,\gamma_{n}:R\to R^{+}$. For a vector $v\in R^{a}$, we define:

$$ $b(v):=\gamma([\lambda_{1}\ldots\lambda_{n}]v)=(\gamma_{i}({v}^{T}{\lambda_{i}}))_{1\leq i\leq n}.$ (15) $$

Each $i$-th element of vector $b$ thus corresponds to the outcome of the $i$-th bucket of LSH.

The bucketing function $b$ may now be used to prepare queries and keys as attention inputs. If $\gamma_{i}$ is a $\{0,1\}$-valued threshold function, then, for $q,k_{i}\in R^{a}$, ${b(q)}^{T}{b(k_{i})}$ is an attention affinity function between $q$ and $k_{i}$, equal to the number of LSH buckets shared between $q$ and $k_{i}$.

###### Observation 7 .

The LSH vector affinity function $b$, given by Equation ([15](#S6.E15)), using $n$ buckets on vectors in $R^{a}$ for some $a\in\mathbb{N}$, can be expressed through Linear Attention with attention keys in the positive orthant $(R^{+})^{n}$.∎

In the ReLU-based setup considered in BDH-GPU, an appropriate function $b$ is plausibly easy to learn. LSH is a ‘sharp-boundary’ technique, well-suited for finding $k$-nearest-neighbors of a queried vector in a set of keys. Hence, the class of attention affinity functions, naturally expressible using BDH-GPU, also includes such ‘sharp’ functions.

###### Observation 7 .

###### Attention in the positive concept space of language and reasoning.

BDH-GPU uses the positive orthant $(R^{+})^{n}$ as its latent space for representing combinations of concepts in its activation vectors. Attention keys and queries are prepared entirely in this positive orthant.

When representing a task of reasoning or language inference in a high-dimensional space, positive activation vectors in $(R^{+})^{n}$ have a natural interpretation of convex combinations of concepts. Such convex combinations of concepts may represent both semantically connected concepts (“bags-of-concepts”), and mixed states of uncertainty between unconnected concepts. In this interpretation, a positive vector is considered as a state of certain knowledge when its L1-norm and L2-norm align closely. Note that for a (normalized) probability vector, the only vectors for which L1-norm and L2-norm coincide precisely are distributions concentrated on a single coordinate.

Linear Attention of BDH-GPU is capable of amplifying very small differences between keys in the L1-norm when matching queries to keys. Consider, for instance, two probability distribution vectors $x_{1},x_{2}\in(R^{+})^{n}$, where $x_{1}=(\alpha,\frac{1-\alpha}{n-1},\frac{1-\alpha}{n-1},\ldots,\frac{1-\alpha}{n-1})$ and $x_{2}=(\frac{1-\alpha}{n-1},\alpha,\frac{1-\alpha}{n-1},\ldots,\frac{1-\alpha}{n-1})$, for some $0<\alpha<1$. Now, vectors $x_{1}$ and $x_{2}$ almost coincide when treated as probability distributions, $\|x_{1}-x_{2}\|_{1}=O(\alpha)=\|x_{1}-x_{2}\|_{\mathrm{TVD}}$. However, they are extremely different when considered as keys for the Linear Attention mechanism, with $x_{1}$ showing very weak affinity to $x_{2}$: ${x_{1}}^{T}{x_{2}}=O(\alpha^{-2}n^{-1}){x_{1}}^{T}{x_{1}}$.

###### Observation 8 .

In key-query matching, the Linear Attention mechanism of BDH-GPU is able to separate positive keys which are close in the L1-norm, strongly amplifying L1-norm differences of activation vectors.∎

This mechanism can be treated as complementary to the propagation dynamics of positive activations in the feed-forward network, discussed in Section [5.3](#S5.SS3).

###### Observation 8 .

##### Natural support for long context.

There is no bound on context length in BDH-GPU, so the actual $t=\tilde{\Omega}(\sqrt{n})$ to $t=\tilde{O}(n)$ “equally important facts” that a BDH-GPU model can distinguish in each layer in view of Claim [7](#Thmclaim7) do not have to correspond to the latest $t$ “facts” seen in context. For example, if, for some layer $l$, mechanisms from lower layers deem a given entry to be irrelevant for layer $l$, and provide an extremely weak attention ‘value’ for this layer, and this key-value entry is effectively seen as omitted. This mechanism corresponds to weaker signals $y$ in a layer which needs to take no action on a given input, e.g., does not have to remember it (cf. Fig. [14](#S6.F14)). Indeed, empirically we observe progressive de-noising of state in the higher layers, with only small fractions of input tokens requiring significant key-value state update in the middle layers across the entire spectrum of state ${\boldsymbol{\rho}}$ of neurons.

As a result, the middle and higher layers of BDH-GPU may, in principle, have unbounded look-back on context. Nonetheless, as context length $t$ increases, we find that damping of historical signals over long sequences is necessary in BDH-GPU to avoid overwhelming the model with noise from stale context. For the vanilla version of the architecture, we found that RoPE combined with ALiBi provide a sufficient remedy, and model performance improves as context length increases. More advanced techniques for BDH-GPU, related to selective forgetting, state compression, or other forms of state optimization, can also be added to the architecture.

### 6.2 Micro-interpretation of attention in BDH-GPU

BDH maintains its state in the $n\times n$ matrix ${\boldsymbol{\sigma}}$ that has a clear interpretation as synapse weights that connect neurons (cf. Section [2](#S2)). On the other hand, BDH-GPU’s state ${\boldsymbol{\rho}}$ is a $n\times d$ matrix. To perform the analysis for BDH-GPU, in this section we recover ${\boldsymbol{\sigma}}$ from the relation:

$$ ${\boldsymbol{\sigma}}_{t-1,l}=\sum_{\tau<t}{y_{\tau,l-1}}{x_{\tau,l}}^{T}U^{t-\tau}$ (16) $$

Figure: Figure 11: BDH’s state ${\boldsymbol{\sigma}}$ encodes neuron connections as a scale-free graph showing clear heavy-tailed (power-law-like) degree distribution.
Refer to caption: https://arxiv.org/html/2509.26507/2509.26507v1/sigma_degrees_powerlaw.png

We first analyze the neuron relationship graph encoded by matrix ${\boldsymbol{\sigma}}$. As explained in Section [2.2](#S2.SS2), ${\boldsymbol{\sigma}}$ can be interpreted as a graph of context dependent implications between $x$ and $y$. We compute the ${\boldsymbol{\sigma}}$ matrix for 0-th head at layer 5 of an 8-th layer network trained on Europarl translation corpus  (we provide more details in Appendix [B.3](#A2.SS3)). We filter out negative entries which are introduced by the RoPE positional embeddings  and enforce a small positive threshold on remaining values to further sparsify the network structure. We plot the histograms of neuron in- and out-degrees, unraveling a scale-free network structure.

Encouraged by the emergent network structure, we have identified a few synapses that are activated at recognizable concepts, we show examples in the next section.

### 6.3 Empirical findings: monosemantic synapses

We have identified in the ${\boldsymbol{\sigma}}$ matrix entries (synapses) that show activity whenever a currency name or country name, both frequently occurring in the Euro-parliament transcripts, is present in the processed sentence. We have identified the synapses by searching for entries in ${\boldsymbol{\sigma}}$ that have predictive power at separating sentences containing a concept from contrast sentences. We present a few examples in Figure [12](#S6.F12). We note that the synapses strength changes abruptly after words that are related to each concept. The same synapse is activated for concepts in both French and English sentences, even when the words used are different (e.g. “livre sterling” vs “British Pound”). Synapse selectivity to a semantic context stems directly from sparsity of neuron activations as shown in Fig. [13](#S6.F13).

Figure: Figure 12: Evolution of values set by BDH-GPU on 2 specific synapses which we have named (following their interpretation) as “currency synapse” and “country synapse”, relating to concepts naturally present in European Parliament transcripts on which the model was trained. We can notice that mentions of country or currency names result in an increase of the respective synapse value, indicating a stronger presence of the concept in the context. Moreover, the synapses consistently became activated in both French and English, confirming the (notice how it reacts both to “British Pound” and “livre sterling”). For visual clarity, we indicate changes that clear a small threshold with the $*$ character (the changes in activity when the system is processing the translation of a source sentence tend to be small).
Refer to caption: https://arxiv.org/html/2509.26507/2509.26507v1/money_country_synapses_enfrespt.png

Figure: Figure 13: Sparse updates to synapses related to meaningful concepts stem from sparse neuronal activations. BDH-GPU maintains in its recurrent state a “currency synapse” (a concept naturally present in the Europarl corpus, see also Fig. [12](#S6.F12)). The synapse is updated using a Hebbian learning rule when activity in $y$ activations at a preceding layer (4 in the example) leads to firing of neuron $x$ in the next layer (5).
Refer to caption: https://arxiv.org/html/2509.26507/2509.26507v1/synapse_bis.png

To confirm the selectivity of the synapses, we have generated, using ChatGPT, 50 sentences relating to European currencies, and another set of 50 sentences speaking about European politics, but not mentioning currencies. A one-sided Mann–Whitney U test revealed that sentences relating to currencies received significantly higher “Currency synapse” values than those without the currency concept ($U=2368$ with $U_{\textrm{opt}}=2500$, $p<10^{-14}$). The rank-biserial correlation was $0.86$, further confirming association between Currency concept presence and synapse value.

### 6.4 Empirical findings: sparse neuron activations

Sparsity of signals is often a prerequisite to their interpretability. In section [6.3](#S6.SS3) we have shown that BDH has monosemantic synapses, selectively activated by occurrences of specific concepts. In this section, we experimentally show that neuron activity correlates with signal predictability: fewer neurons are active, or equivalently, layer activations become sparser, for more predictable input signals.

We have trained a BDH-GPU model with $n=65536$ neurons, $d=256$, $L=4$ layers, and tokenization on letters of the Latin alphabet, to perform a single synthetic next-token prediction task. The input sequence started with a fixed $13$-letter warm-up sequence, followed by $8$ repetitions of an $8$-letter random word (“fact”), with the same pattern repeating every $13+8\cdot 8=77$ letters. In Fig. [14](#S6.F14), we show neuron activity patterns. We can notice that neurons in higher layers are active during warm-up and fact introduction, then become quiet. We then group neurons by their RoPE frequencies and find that largest difference of activity during memorization and repetition is shown by the slow-acting neuron population.

Figure: Figure 14: Neurons in BDH-GPU are less active (signal is sparser) when the input is predictable. The input sequence started with a fixed $13$-letter warm-up sequence, followed by $8$ repetitions of an $8$-letter random word (“fact”), with the same pattern repeating every $13+8\cdot 8=77$ letters. (a) Fraction of neurons with non-zero entry $y_{t,l}$ in different layers $l$, with fact memorization effect noted through increased activation level in layer $2$. The activation in layer $2$ has $4.0\%-7.5\%$ non-zero entries during memorization and approximately $2.5\%$ non-zero entries during repetition. (b) Detailed breakup of activation sparsity in layer $2$, with neurons bucketed into equal fractions by their RoPE phase: $\textrm{freq}0\in[1,4]$, $\textrm{freq}1\in[4,16]$, $\textrm{freq}2\in[16,64]$, $\ldots$, $\textrm{freq}7\in[16384,65536]$. The slow-acting half of the neuron population ($\textrm{freq}4-\textrm{freq}7$) exhibits the largest amplitude ratio between peak activation during memorization and repetition phases.
Refer to caption: https://arxiv.org/html/2509.26507/2509.26507v1/fig12av2.png

From a biological standpoint, sparse and surprisal-driven neuron activation lowers energy consumption — despite fluctuations in low level percepts (in the experiment tokens are changing at every timestep), neurons in higher layers are inactive and do not expand energy. From a Deep Learning perspective, it has been recently shown that input complexity is related to predictability of internal representations of Transformers . BDH makes this very explicit and does not require a separate prediction network: the predictable steady-state consists of zero activations, and input complexity entails neuronal activity. This suggests that BDH, natively, at a neuron level, implements mechanisms reminiscent of adaptive computation time  and conditional computation , used in modern Transformers to lower computational effort during inference.

Finally, sparse activation vectors in BDH imply that potentiation of specific synapses occurs rarely during inference. This is useful from the point of view of interpretability, noise reduction in Linear Attention state, and opens the door to simplified and compressed representations, notably for state and for gradient backpropagation DAG’s.

## 7 Playing with the Hatchling

### 7.1 Model merging: concatenating two models

Updating models with up-to-date knowledge and expanding models knowledge-base will become crucial in practical applications of AI. On possible solution is model composability, potentially allowing building of larger models by assembling a number of smaller, specialized models into a larger, more powerful one. A natural hope for such a system would be the achievement of ”more is different than a sum of its part” effect. In the following experiment we are showing that doing so is relatively straight forward with BDH-GPU. This is because BDH-GPU can be scaled by varying only the number of neurons $n$. In this section we explore whether we can create larger models directly concatenating smaller models trained on disjoint subsets of data. Details in Appendix [B.4](#A2.SS4). We have experimented with the following simple model merging procedure:

- 1.
Train a base model on a chosen language pair. In the experiment we have used English-Spanish (En-Es) translation data, and have trained a model with $n=24576$ neurons (19M parameters).
- 2.
Clone the base model and continue training on two datasets: English-French (En-Fr) and English-Portuguese (En-Pt).
- 3.
We then merge the weights of the En-Fr and En-Pt models to create a new En-FrPt model with $n=24576\cdot 2=49152$ neurons (38M parameters).
To create the merged model we:
(a)
concatenate all parameter tensors that have an ‘$n$’ dimension (e.g. ${D_{y}}$, ${D_{x}}$, $E$, RoPE frequency buffers) along their $n$ dimension,
(b)
average all other parameters (e.g. token embeddings and token prediction weights).
To validate the hypothesis that direct model merging is feasible, we report all results on the merged model without any subsequent training or finetuning. However, we have verified that the merged model quickly improves when trained on all language pairs.
- 4.
After each stage we evaluate the models on all involved language pairs: En-Es, En-Fr, En-Pt, regardless of the data seen by the model up to this stage.

**Table 2: Validation next token prediction losses (lower is better) of translation models trained on different language pairs and then merged. We evaluate each model on En-Es, En-Fr, En-Pt language pairs separately. We can see that the base model can translate between English and Spanish, while on En-Fr and En-Pt tasks it falls back on perplexities of an unconditional English language model (loss about $0.65$) and can’t generate proper French or Portuguese. After tuning on French or Portuguese the model learns to translate between respectively English and French or English and Portuguese, while somewhat retaining the capacity to translate Spanish to English and losing the capability to translate English to Spanish. The merged model can translate Spanish, French, and Portuguese to English, however it mixes these three languages when asked to translate from English. This is consistent with qualitative results shown in Figure [15](#S7.F15).**
|  | *Translation into English* | *Translation from English* |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| *Model:* | Es$\rightarrow$En | Fr$\rightarrow$En | Pt$\rightarrow$En | En$\rightarrow$Es | En$\rightarrow$Fr | En$\rightarrow$Pt |
| 1: Base En-Es | 0.36 | 0.77 | 0.64 | 0.35 | 2.21 | 2.27 |
| 2: Base (1) tuned on En-Fr | 0.58 | 0.36 | 0.68 | 2.57 | 0.31 | 2.54 |
| 3: Base (1) tuned on En-Pt | 0.44 | 0.76 | 0.34 | 1.79 | 2.20 | 0.33 |
| 4: Merged (2$\|$3) | 0.43 | 0.40 | 0.39 | 1.45 | 0.77 | 0.86 |

We report quantitative results in Table [2](#S7.T2) and show qualitative results of merged model operation in Figure [15](#S7.F15). The merged model shows human-like degradation of operation: while it retained the capability to generate and translate into English, it has lost the ability to generate proper text in Spanish, French, or Portuguese, mixing words and grammatical constructs. We have verified that a small amount of training on all language pairs restore the model’s proficiency in Spanish, French and Portuguese. However, we decided to report on the behavior of the merged model without any subsequent tuning to highlight the possibilities of model engineering offered by the large and sparse working dimension $n$ of BDH-GPU.

Figure: Figure 15: Conditional and unconditional samples generated from a English-Spanish-French-Portuguese translation model created by direct concatenation of parameters of models trained on distinct language pairs.

The BDH-GPU model merging experiment has shown that when the model latent space promotes concept disentangling (c.f. Section [6.2](#S6.SS2) on monosemanticity) then it is feasible to directly compose concepts in this space, e.g. by concatenation of weights from different models. This feature of the BDH architecture allows us to see the models as composable computer programs with emergent properties.

### 7.2 Training without backpropagation through time

Sparsity of synapse activations in BDH opens the door to efficient approximations to backpropagation through time. The main intuition is that we only need to remember when a synapse has changed, and the $i,j$ coordinates of the synapse implicitly encode which neurons were active and should take part in error signal backpropagation.

In this section we report results of preliminary experiments on the impact of removal of backpropagation through time on model performance. For the PyTorch implementation in Appendix [E](#A5), this corresponds to ‘detach’-ing variables K and V in the implementation of the LinearAttention class.

In particular, we found that such a model, trained without any backpropagation through time, retained some ability to model language, but lost the ability to match concepts between different languages during translation. For translation tasks like those presented in Table [2](#S7.T2), loss values for English increased from a loss level of approximately $0.65$ for an unconditional English language model (trained with backpropagation over time), to loss of approximately $0.75-1.05$ for a model trained without backpropagation over time, depending on model variant, regardless of whether English was the source or target language in translation. No significant difficulties were encountered during training when crossing the barrier of the letter-bigram language model, at loss value $2.4$.

Beyond side-effects of the general design, we did not optimize the BDH-GPU model for suitability of training without backpropagation. We consider this architecture to be a good starting point for bootstrapping further investigations in this direction.

## 8 Conclusions

### 8.1 Takeaways for model engineering

This paper leads up to a new class of language and reasoning models which eliminate architecture nonuniformities, notably in terms of scaling for model size, and handling of time scales of inference.

The BDH-GPU architecture introduced in this paper opens the following opportunities:

- 1.
*New ways of scaling models for time and size.* BDH-GPU is a state-space model which scales for size in one large dimension $n$ (neurons in this dimension are indexed by RoPE oscillator frequency). Subject to appropriate sharding, this also leads to a desirable form of locality: important data is located just next to the sites at which it is being processed. This minimizes communication, and eliminates the most painful of all bottlenecks for reasoning models during inference: memory-to-core bandwidth.
- 2.
*Faster model iteration.* During training and inference alike, BDH-GPU provides insight into parameter and state spaces of the model which allows for easy and direct evaluation of model health and performance, notably, through sparsity-related measures and through aggregates and statistics on the large pool of homogeneous neurons, even for relatively small models. Attention and parametric layers alike operate on the same neuron dimension (‘concept dimension’).
- 3.
*Direct explainability of model state.* Elements of state of BDH-GPU are directly localized at neuron pairs, allowing for a micro-interpretation of the hidden state of the model.
- 4.
*New opportunities for ‘model surgery’.* The BDH-GPU architecture is, in principle, amenable to direct composability of model weights in a way resemblant of composability of programs. This concerns the potential both the direct composition of separately trained model parts, as well as ‘surgery’ of parameter spaces of models, by inserting fragments of manually programmed protocols into machine-learned code.

### 8.2 Implications for brain science

We have obtained a micro-foundational description of attention for artificial language and reasoning models, expressed in a framework of local graph dynamics. This has been found to be consistent with the effects observed for the same function of *attention for language and reasoning* in the brain. By introducing a translation layer based on similarity of function between the artificial and biological planes, for blocks of feed-forward neural networks and attention mechanisms, our work points to the following hypothesis: *complex systems effects which are observed in the brain, around modular scale-free network structure, synaptic plasticity, and Hebbian learning arose from its core purpose — doing reasoning — and not from any specific longer-term training dynamics which the brain applies*.

We have exhibited how a general attention mechanism can be efficiently implemented as an artificial neuronal system with spiking neurons and synapse plasticity. More formally, we first describe the class of local interaction dynamics which any system *plausibly needs* to implement attention mechanisms. We then confirmed that the edge-reweighting rule is *sufficient* to allow a certain artificial Language Model (BDH-GPU) to operate at least at the level of the Transformer. For an artificial network, the edge-reweighting rule intuitively describes the interaction between two artificial neurons exhibiting rapid state-change behavior, and one synaptic neuron interconnection element exhibiting plasticity as shown in Fig. [13](#S6.F13).

More broadly, this work may potentially serve to support efforts aiming to isolate, from among the many extremely complex electrochemical patterns and signal dynamics occurring in the brain, those that are crucial for *solving tasks in-context (based on attention)*, from those that potentially serve other purposes, such as transfer of information from short-term memory to long-term memory, or long-term improvement of brain function (learning).

##### How this work helps with axiomatization of learning theory in the brain.

Attempts to understand the brain, starting from the perspective of longer time scales of training, have proved extremely challenging, defying progress. This paper pin-points attention-based reasoning at shorter time scales as ‘the other end of the string’, and hints how, from here, untangling the entire story will plausibly be easier.

For natural systems undergoing continuous learning, the time scales to look at are: language function and reasoning (chain-of-thought inference), then short-to-long memory transfer from state to network weights, adaptation of structure: changes to interconnections, and finally, changes to neuron nodes.

For long time scales, this reduces the question of finding supervised training dynamics form the most general case, to a specific class of local dynamics: an interaction kernel performing ‘edge-reweighting’ rules. As these rules appear fundamental to logical inference and biochemical processes alike, its universality in processes that the brain is responsible for is plausible also beyond the realm of language-based reasoning.

*From a systems perspective, we arrive at the following possible explanation.* The brain generally tries to be lazy in terms of energy expense, and does things as late as it can. Only reasoning needs to happen close to a critical regime, because it involves executing a real-time program which needs to be responsive, since the life and success of the biological organism depends on it. Then, for a certain time, which may be minutes for humans, the brain has enough synapses in it to represent (almost) all useful information it needs for reasoning, decision-making, etc. — all stored in short-term state, at synapses (and/or neurons). Some of the neuron activations which the brain performs at this time scale represent ‘gradients of state’ — the gradients of in-context learning, passed on to modify synapse strength, in a weight-update process. As time goes by, the system runs out of state space. Then, memory processes work to iron things out, preserving in more permanent neuron connection weights and graph structure the elements of state that have been reinforced by feedback signals. Overall, there are fewer and fewer things that need to be remembered across progressively longer time scales. However, this entire memory process is, plausibly, subsidiary to the definition of the dynamics of reasoning and the synaptic dynamics of state that we discuss in this paper. In other words, the best form of description of the relaxation from state into longer-term memory follows from the specific kernel of the reasoning dynamics, such as the edge-reweighting kernel.

As for the ratio of time scales (measured in tokens for language), we can estimate that the time lapse after which harmonizing state with a memory process becomes important is of about the same order of magnitude as the average time between ‘writes’ (significant transmission increases) for individual synaptic elements (see e.g. Fig. [14](#S6.F14)). In our models, this time is lower-bounded by the inverse of sparsity of the vector $y$, i.e., $1/\rho\approx 1/5\%=20$ tokens, but it could be much larger for larger systems; we also do not force it in any way to be sparser during training. During training with backpropagation, if the backpropagation window $T$ is short enough, $T<1/\rho$ tokens, we can plausibly assume that a synapse changes state only once in that window (and is used multiple times), hence the DAG of gradient backwards propagation is much more direct to embed within the system graph. Backpropagation is then a question of ‘routing’ gradients in the neuron communication graph, and not one of disentangling them. All natural training approaches, whether based on backpropagation, or any more direct form of relaxation ‘from state into weights’, appear to bottleneck on the amount of available state space on synapses, becoming necessary at about $T\sim 1/\rho$ by a simple information-theoretic argument on state storage capacity.

Regardless of how much of this is an accurate description, and how much an intuition, at the very least, it appears we may now have a way forward. Some part of the “global mystery” of learning in the brain can be reduced to a more “localized problem” of state-to-operator transfer for some relatively compact form of state-space dynamics (i.e., one specific local graph kernel). This change of perspective brings in both a completely new ‘problem landscape’ in which to navigate towards a complete solution, as well as a set of new methods to use for the different types of graph structure changes involved in learning, including approaches from distributed computing, evolving network theory, and graph rewiring systems.

At this point, it seems one natural next step would be to ground the current discussion more deeply in findings of brain science, to refine or simplify the *actual kernels* used by brain reasoning (which was not the objective of this paper), and potentially seek validation through experiment.

### 8.3 Societal impact

This paper is a voice in favor of bringing principled understanding to reasoning in Machine Learning. Axiomatic AI provides an opportunity to reduce risks related to unpredictable behavior of AI models, and, to open or accelerate new development directions. The subject matter which we consider here serves as a direct introduction to the most crucial problem that lies ahead: controlling the behavior of autonomous AI reasoning models and AI systems as they progress across time scales, from seconds to years.

Figure: Table 3: Comparison of properties of language and reasoning model architectures: the GPT2 Transformer, BDH-GPU, BDH, and brain models.

### Acknowledgments

The authors thank David Sussillo, Navdeep Jaitly, and Emanuele Natale for insightful discussions on reasoning and the brain, and for early feedback on this write-up.
We also thank Samy Bengio for comments on the presentation.
We kindly acknowledge the support of all of the Pathway team, notably, Paweł Podhajski for his amazing help with cluster setup, Victor Szczerba and Z Schwab for all discussions over coffee, and Kamil Piechowiak and Chris Ociepa for constructive comments on the presentation. AK thanks Christos Papadimitriou for being the direct inspiration for us to embark on this journey.

### Author contributions

AK conceived the BDH and BDH-GPU architectures, conceived most of the theory, developed most of the model source code, conceived and performed experiments on synapses, and wrote most of the paper.

PU contributed crucial elements of BDH-GPU architecture, contributed model and framework source code, contributed to theoretical analysis, and performed experiments.

JCh led, designed, and oversaw methodology of experiments, led framework development, contributed major improvements to BDH-GPU architecture, contributed to the theory, implemented baselines, performed experiments, and substantially redacted the paper.

ZS conceived the project, guided research directions, introduced particle-interaction interpretation, acted as final judge in research decisions, and substantially redacted the paper.

MB optimized model source code, contributed framework source code, and performed experiments.

## Appendix A Connection between generalization of reasoning and computational expressiveness

State-of-the-art reasoning models have the interpretation of (Turing-complete) programs, executed over a certain period of time. This shifts the emphasis of generalization, from discovering the structure of mathematical functions which maps inputs to outputs, to discovering a class of runnable programs, which take as input a given class of input prompts, and process these prompts “in the right direction”.

Consider a given reasoning task, whose scope is defined as a set $\mathcal{P}$ of valid input prompts, given as bounded-length token sequences over some alphabet $\Omega$. Given a prompt from $\mathcal{P}$, a model solving the considered task is eventually (i.e, after some number of steps of reasoning) expected to generate an output, in the form of a bounded-length token sequence over the same alphabet $\Omega$, which is subjected to evaluation. Consider language models sampled from some probability distribution $\mathcal{M}_{1}$ over parameter sets in some architecture $\mathcal{A}_{1}$.

Now, suppose that for some other model architecture $\mathcal{A}_{2}$ there exists a distribution $\mathcal{M}_{2}$ over language models in $\mathcal{A}_{2}$ such that, for a valid input prompt chosen uniformly at random from $\mathcal{P}$, the outputs sampled from a model $M_{1}\sim\mathcal{M}_{1}$ and the outputs sampled from a model $M_{2}\sim\mathcal{M}_{2}$, have (almost) the same distribution in the space of bounded-length sequences over $\Omega$, and are both obtained within some asymptotic bound on the number of steps of reasoning, in expectation. The described setting is equivalent to saying that *models $\mathcal{M}_{2}$ have generalized the considered task $\mathcal{P}$ in (almost) the same way as models $\mathcal{M}_{1}$*. Indeed, conversely, if the described condition did not hold, we could, in a finite number of trials, distinguish solutions to problem $\mathcal{P}$ obtained by model families $\mathcal{M}_{1}$ and $\mathcal{M}_{2}$.

Now, consider model architectures $\mathcal{A}_{1},\mathcal{A}_{2}$ which apply Chain-of-Thought reasoning . A model in such an architecture has the interpretation of a trainable probabilistic program, taking inputs from $\mathcal{P}$, and the architectures themselves represent computational machine architectures. Moving to a discussion of computational expressiveness, we obtain the following statement.

###### Observation 9 .

Given a probability distribution of models $\mathcal{M}_{1}$ in architecture $\mathcal{A}_{1}$, suppose there exists a distribution over models in architecture $\mathcal{A}_{2}$ which generalizes on task $\mathcal{P}$ in the same way as models from $\mathcal{M}_{1}$. Then, the machine architecture $\mathcal{A}_{2}$ has sufficient computational expressiveness to simulate programs from $\mathcal{M}_{1}$ efficiently on the set of inputs $\mathcal{P}$, i.e., $\mathcal{A}_{2}$ contains programs which obtain an (almost) identical distribution of outputs within the given bounds on running time.
∎

In particular, we note that if we were to consider the special case of $\mathcal{A}_{1}$ being reasonable human agents, we could say that architecture $\mathcal{A}_{2}$ generalizes reasoning, in the same way as humans, if we can train models $\mathcal{M}_{2}$ in $\mathcal{A}_{2}$ which accurately reproduce the outcomes of reasoning for some sample $\mathcal{M}_{1}$ of humans in $\mathcal{A}_{1}$.

This leads us naturally to describe Language Model generalization through a universal reference to the principles of operation of the human brain, treated as a distributed computing architecture, and not through a characterization of language and reasoning prompts $\mathcal{P}$ that the model should be able to deal with in some specific way.

###### Observation 9 .

## Appendix B Further description of experiments

### B.1 Language translation task

We have evaluated our models on a mixed language modeling and translation task derived from the Europarl corpus . The corpus consists of sentence-level aligned translations of transcripts of European Parliament proceedings. For each language pair, we treat the data as a long stream of interleaved source and target sentences (sampling for each sentence which language is the source, and which is the target) on which we train decoder only models. Thus, models are jointly trained as language models and translators. We train all models using Truncated Backpropagation Through Time . Subsequent minibatches served by the data loader are related: each is a continuation of the previous. Each model maintains a recurrent state, carried across minibatches: ${\boldsymbol{\rho}}$ matrix for BDH-GPU and a FIFO buffer of recent KV-cache entries for the TransformerXL  baseline. We train all models on raw UTF8 data. We are mainly interested in model comparison and prefer to keep the experimental setup as simple as possible. A few minibatches are shown in Fig. [16](#A2.F16).

The joint language modeling and translation formulation has several benefits:

- 1.
Next token prediction is representative for LLM training. Simple architectures, such as decoder-only models are sufficient.
- 2.
The task promotes models with long context capabilities — subsequent sentences are related and the model can meaningfully
utilize long context to model the source language sentences.
- 3.
The task promotes models which carry state across minibatches, as training data is temporally coherent and the final model state at the end of one minibatch is a natural initialization of hidden state on the next minibatch.
- 4.
Translation can be seen as language modeling coupled with fuzzy copying. Successful models will need to develop in-context learning capabilities such as inductive heads .

Figure: Figure 16: Exemplary sequence of 10 successive minibatches from the translation task. The model is trained on raw UTF8 bytes (for visualization we pad multi-byte UTF8 characters with “•” symbol). Special token strings <F:lang_code> and <T:lang_code> delimit source and target sentences. Minibatches are temporally coherent: source sentences are followed by their translations, and subsequent source sentences are part of the same larger document.

### B.2 BDH Scaling Experimental Details

We provide details on models used in scaling experiments described in Section [4.2](#S4.SS2). All models were implemented in PyTorch  and trained on the Europarl  task described in Section [B.1](#A2.SS1). We have kept the same training regime for all models at all sizes: En-PL and En-Cs language pairs (380MB total). All models trained on raw UTF8 bytes seeing a total of 1.2B tokens (about 3 epochs). All minibatches were 2048 tokens long, but we have varied the number of examples in the minibatch (varying number of tokens in each minibatch) to accommodate different memory requirements of different models. We have used multi-GPU training using the Distributed Data Parallel approach using AdamW  with learning rate $10^{-3}$, and 1000 warm-up step followed by linear learning rate decay over the course of training to $10^{-4}$, adaptive gradient clipping , and weight decay $0.1$. Models were trained to operate on a context longer than minibatch length using Truncated Backpropagation Through time .

The Baseline model, dubbed GPTXL, was a GPT2-like transformer  based off the NanoGPT  implementation with KV-cache carried across minibatches as in TransformerXL . We have used ALiBi positional biases . We list its hyperparameters for various model sizes in Table [4](#A2.T4). Optimal Dropout was selected using a small sweep at each model size.

**Table 4: Hyperparameters for GPTXL baselines in scaling experiments. The model architecture follows GPT2 , with a FIFO buffer of past KV-cache entries .**
| model | num | embd | num | MLP | dropout | Carried KV-cache |
| --- | --- | --- | --- | --- | --- | --- |
| size | layer | dim | head | dim |  | size |
| 25M | 9 | 480 | 5 | 1920 | 0.01 | 4096 |
| 50M | 12 | 576 | 6 | 2304 | 0.02 | 4096 |
| 100M | 15 | 768 | 8 | 3072 | 0.02 | 4096 |
| 200M | 18 | 960 | 10 | 3840 | 0.002 | 4096 |
| 400M | 25 | 1152 | 12 | 4608 | 0.005 | 4096 |
| 800M | 28 | 1536 | 16 | 6144 | 0.15 | 4096 |

BDH-GPU directly uses model code provided in Appendix [E](#A5). BDH-GPU’ adds xLSTM-like gating mechanism , and merges next token predictions from all layers. Both BDH-GPU and BDH-GPU’ use same architectural hyperparameters, gathered in Table [5](#A2.T5).

**Table 5: Hyperparameters for BDH-GPU models in scaling experiments.**
| model | num | $d$ | $n$ | num | dropout |
| --- | --- | --- | --- | --- | --- |
| size | layer |  |  | head |  |
| 25M | 8 | 256 | 32768 | 4 | 0.1 |
| 50M | 8 | 256 | 65536 | 4 | 0.1 |
| 100M | 8 | 256 | 131072 | 4 | 0.1 |
| 200M | 8 | 256 | 262144 | 4 | 0.1 |
| 400M | 8 | 256 | 524288 | 4 | 0.1 |
| 800M | 8 | 256 | 1048576 | 4 | 0.1 |

### B.3 BDH Monosemantic Synapse Experiment Details

We provide details for models used in exploration of monosemantic synapses in Section [6.2](#S6.SS2). The model was trained on Europarl  described in Section [B.1](#A2.SS1). It had $d=256,n=49152$, $4$ attention heads, and $8$ layers. The model was trained on about one epoch of En-Es, En-Pt, and En-Fr data (total 1.9B tokens) in a Distributed Data Parallel setup using AdamW  with learning rate $10^{-3}$, 1000 warm-up step followed by linear learning rate decay over the course of training to $10^{-4}$, adaptive gradient clipping , and weight decay $0.1$. We have used Truncated Backpropagation Through time, carrying over the recurrent state of attention and training on sequences of length $2048$ characters at a time. We have used minimal Dropout  of $0.01$.

### B.4 BDH Merging Experiment Details

We provide details for models described in Section [7.1](#S7.SS1)
All models were trained on Europarl  described in Section [B.1](#A2.SS1). We provide model architecture hyperparametrs in Table [6](#A2.T6). Models were trained on about two passes over the training set in a Distributed Data Parallel setup using AdamW  with learning rate $10^{-3}$, 1000 warmup step followed by linear learning rate decay over the course of training to $10^{-4}$, adaptive gradient clipping , and weight decay $0.1$. We have used Truncated Backpropagation Through time, carrying over the recurrent state of attention and training on sequences of length $2048$ characters at a time. We have used minimal Dropout  of $0.01$.

**Table 6: Architecture and training details for model merging experiments.**
| Model | Init. | Training | Data size | Training | n | d | num. | num. | param. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | from | data | (bytes) | tokens |  |  | heads | layers | count |
| BaseEnEs | — | En-Es | 612M | 1.2B | 24576 | 256 | 4 | 8 | 19M |
| TunedEnFr | BaseEnEs | En-Fr | 640M | 1.2B | 24576 | 256 | 4 | 8 | 19M |
| TunedEnPt | BaseEnEs | En-Pt | 616M | 1.2B | 24576 | 256 | 4 | 8 | 19M |
| MergedEnEsFrPt | TunedEnFr+TunedEnPt | — | — | — | 49152 | 256 | 4 | 8 | 38M |

## Appendix C Omitted formal claims and proofs

### C.1 Proof of Observation 1

###### Proof.

The equivalence is straightforward to verify, rewriting the linear-algebraic multiplication expressions of Eq. ([6](#S3.E6)) in Einstein summation notation and comparing respective index pairs. At any time, during the execution of rules for layer $l$, variables $X(i)$, $Y(i)$ and $\sigma_{l}(i,j)$ in the protocol description, for $i,j\in\{1,\ldots,n\}$ correspond to the $i$-th coordinate of vectors $x_{t,l}$ (based on $x_{t,l-1}$ from the previous round), $y_{t,l}$ (based on $y_{t,l-1}$ from the previous round), and matrix entry ${\boldsymbol{\sigma}}_{t,l}$ (based on ${\boldsymbol{\sigma}}_{t-1,l}$ from the previous token). The auxiliary variable $A(i)$ corresponds to a similar auxiliary vector $a_{t,l}:={\boldsymbol{\sigma}}_{t-1,l}x_{t,l}$ in an intermediate step of computation of $y_{t,l}$ from $x_{t,l}$. The parameter $u(i,j)\in R^{+}$ associated with an element of state follows from the definition of matrix $U$; we assume for simplicity that $U$ is diagonal (which corresponds to the case of ALiBi). Finally, in Table [1](#S2.T1), the auxiliary node variables $X^{\mathfrak{e}}(i),X^{\mathfrak{i}}(i),Y^{\mathfrak{e}}(i),Y^{\mathfrak{i}}(i)$ are used to handle the thresholding of the inhibitory circuit.
∎

###### Proof.

### C.2 Formal statement of Claim 7 (linear attention)

We provide the following Claim, expressing the operation of attention under *$C$-non-adversarial* key vectors $(k_{\tau})$, $t=1\ldots t$, understood in the sense that there exists $C\in N$, $0\leq C<t-1$ such that, if considering $(k_{\tau})$ as a sequence of random variables, each $f(k_{\tau})$, $\tau=1\ldots t$, can be considered sampled independently at random in $S^{\nu}$ with respect to all keys sampled previously, except for at most $C$ such keys. We put $C=t-1$ for adversarial inputs, or if this condition cannot be satisfied at all due to the nature of function $f$.

###### Claim 8 .

Let $\Lambda$ be a space of keys and queries, let $\phi:\Lambda\times\Lambda\to[-1,1]$ be an attention affinity function, and let $f:\Lambda\to S^{\nu}$, for some $\nu=O(\mathrm{poly}(n))$, be such that for any $q,k\in R$, we have $f(q)\cdot f(k)=\phi(q,k)\pm O(n^{-100})$.
Fix $\delta>0$ and $C\in\mathbb{N}$. Let $A_{\phi,t}$ be a block which computes attention $a_{t}$ given by Eq. ([14](#S6.E14)), for a given sequence of key-query inputs $(k_{1},\ldots,k_{t})$ and values $(v_{1},\ldots,v_{t})$, where $t<\delta n/((C+1)\log n)$ is fixed, $k_{\tau}\in\Lambda$, and $v_{\tau}\in R^{d}$ are of similar strength in the L2-norm, with $c_{1}\leq\|v_{\tau}\|\leq c_{2}$, for all $\tau=1\ldots t$, for some constants $0<c_{1}\leq c_{2}$. Then the (simplified) linear attention equation of BDH-GPU:

$$ ${a^{*}_{t}}:=\sum_{\tau=1}^{t-1}{v_{\tau}}{x_{\tau}}^{T}{x_{t}}$ (17) $$

expresses $A_{\phi,t}$ with $O(\sqrt{\delta})$-error in the L2-norm (i.e., $\|a^{*}_{\tau}-a_{\tau}\|=O(\sqrt{\delta})$, provided that the input vector $(k_{\tau})$ is $C$-non-adversarial, under a suitable randomly chosen key preparation function $f^{\prime}:\Lambda\to R^{n}$ , $x_{\tau}:=f^{\prime}(k_{\tau})$, where $f^{\prime}$ depends on $f$, w.h.p. in $n$ with respect to choice of $f^{\prime}$.

###### Proof (sketch).

To simplify notation, assume w.l.o.g. that $\Lambda=S^{\nu}$ and $f=idem$; to undo this assumption, at the end of the proof we apply $f\circ f^{\prime}$ for preparation in place of $f^{\prime}$.

All vectors $v$ and the result $a_{t}$ we are looking to calculate are in $R^{d}$. With this notation, the attention task we are approximating is:

$$ $a_{t}=q\sum_{\tau=1}^{t}k_{\tau}^{T}v_{\tau}.$ (18) $$

(this is still the general form of attention almost precisely equivalent to ([14](#S6.E14)), not a special case).

The goal is to show how, subject to $t<\delta n/\log^{2}n$, linear attention in dimension $n$ given by ([17](#A3.E17)) is a sufficiently precise estimation of ([18](#A3.E18)).

Consider now, with $\Lambda=S^{\nu}$, $f^{\prime}:S^{\nu}\to R^{n}$, where we recall that $x_{\tau}:=f^{\prime}(k_{\tau})$, to be a suitable dimensionality reduction preserving approximation of scalar product between $R^{\nu}$ and $R^{n}$. For simplicity of argument, we let $f^{\prime}:R^{\nu}\to R^{n}$ be a standard Johnson-Lindenstrauss transform, with the additional property that $f^{\prime}(-z)=-f^{\prime}(z)$ for all $z\in R^{\nu}$ (easy to obtain from any other Johnson-Lindenstrauss transform $f^{\prime\prime}$ by taking $f^{\prime}(z):=(f^{\prime\prime}(z)-f^{\prime\prime}(-z))/2$). The distortion of scalar product in $R^{n}$ is then known to be bounded as follows: $|{k_{\tau}}^{T}{k_{t}}-{x_{\tau}}^{T}{x_{t}}|=O(\varepsilon)(\|k_{\tau}\|+\|k_{t}\|)=O(\varepsilon)$, w.h.p. with respect to choice of $f^{\prime}$. Here, $\varepsilon=\sqrt{\log n/n}=O(\sqrt{\delta})/\sqrt{(C+1)t\log t}$, where the last inequality follows from the assumption on $t$ made in the Claim.

We now consider the sequence $r_{\tau}:={k_{\tau}}^{T}{k_{t}}-{x_{\tau}}^{T}{x_{t}}$, for $\tau<t$. Set aside the (at most $C$) elements $r_{\tau}$ for which $k_{\tau}$ and $k_{t}$ are not independent. For all other elements, consider that $|r_{\tau}|=O(\varepsilon)$ as established previously, and the sign $r_{\tau}/|r_{\tau}|$ is chosen independently at random with respect to all but at least $C$ elements by the conditions imposed on $f^{\prime}$ and $k_{\tau}$. It follows that $\sum_{\tau=1}^{t}r_{\tau}$ can be represented as a sum of $O(C)$ martingales, each of which has length $O(t/(C+1))$ and all elements bounded by $O(\varepsilon)$ with $\varepsilon=O(\sqrt{\delta})/\sqrt{(C+1)t\log t}$. The Claim follows directly, by applying Azuma’s inequality to each of these martingales independently.
∎

Considering the extreme cases of $C=0$ and $C=t-1$, the above Claim leads directly to Claim [7](#Thmclaim7), clarifying over what time, linear attention can be used to express general attention.

###### Claim 8 .

###### Proof (sketch).

### C.3 Proof of Claim 3

###### Proof.

The proof is almost immediate, through the construction of an appropriate neuron-synapse interaction graphs $H^{\mathfrak{e}}$, $H^{\mathfrak{i}}$ such that $G^{\mathfrak{e}}={H^{\mathfrak{e}}}^{2}[V]$ and $G^{\mathfrak{i}}={H^{\mathfrak{i}}}^{2}[V]$. Consider $E^{\prime}\in(R^{+})^{2d\times n}$ such that $E^{\prime}_{\alpha,j}=\left(E_{\alpha,j}\right)^{+}$ and $E^{\prime}_{\alpha+d,j}=\left(-E_{\alpha,j}\right)^{+}$, for $j\in\{1,\ldots,n\}$ and $\alpha\in\{1,\ldots,d\}$. Define $D^{\mathfrak{e}},D^{\mathfrak{i}}\in(R^{+})^{n\times 2d}$ so that:

$$ $(D^{\mathfrak{e}}-D^{\mathfrak{i}})E^{\prime}=DE.$ $$

Indeed, notice that this is always possible by redistributing elements of $D$ into $D^{\mathfrak{e}}$ and $D^{\mathfrak{i}}$ (putting $D^{\mathfrak{e}}_{i,\alpha}=D^{\mathfrak{i}}_{i+d,\alpha}=\left(D_{i,\alpha}\right)^{+}$) and $D^{\mathfrak{i}}_{i,\alpha}=D^{\mathfrak{e}}_{i+d,\alpha}=\left(-D_{i,\alpha}\right)^{+}$), so that, for all $i,j\in\{1,\ldots,n\}$ and $\alpha\in\{1,\ldots,d\}$, we have:

$$ $(D^{\mathfrak{e}}_{i,\alpha}-D^{\mathfrak{i}}_{i,\alpha})E^{\prime}_{\alpha,j}+(D^{\mathfrak{e}}_{i,\alpha+d}-D^{\mathfrak{i}}_{i,\alpha+d})E^{\prime}_{\alpha+d,j}=D_{i,\alpha}E_{\alpha,j}.$ $$

Considering $S=\{1,\ldots,2d\}$, the definition of $H^{\mathfrak{e}}$ as the union of edges of $D^{\mathfrak{e}}$ and $E^{\prime}$ on input neuron layer $V$, hidden layer $S$, and output neuron layer $V$ follows. Likewise, we define $H^{\mathfrak{i}}$ as the union of edges of $D^{\mathfrak{i}}$ and $E^{\prime}$.

We verify that for $G^{\mathfrak{e}}={H^{\mathfrak{e}}}^{2}[V]$ and $G^{\mathfrak{i}}={H^{\mathfrak{i}}}^{2}[V]$, we have $G^{\mathfrak{e}}-G^{\mathfrak{i}}=DE$, and the Claim holds.
∎

###### Proof.

##### Considerations of building linear circuits.

The above proof makes the neuron-synapse interaction graphs $H^{\mathfrak{e}}$, $H^{\mathfrak{i}}$ sparse in terms of the number of edges, as required to show that the number of parameters are preserved by correspondence. However, it is a purely technical construction, and nodes in the synaptic layer have high degree, $n$. While preserving strict equivalence of linear dynamics, the degrees of nodes of the considered graphs in the synaptic layer can be reduced in this construction, at the cost of increasing the number of edges of graphs $H^{\mathfrak{e}}$, $H^{\mathfrak{i}}$. (For example, subdividing each node of the synaptic layer into $a^{2}$ nodes can be used to reduce their degree $\Theta(a)$-times, while increasing the number of edges $\Theta(a)$-times; putting $a=\sqrt{n/d}$ we reach graphs $H^{\mathfrak{e}}$, $H^{\mathfrak{i}}$ with degree $O(\sqrt{nd})$ in both the neuron and synaptic layers, and consequently $O(n\sqrt{nd})$ edges.)

Reduction of internal degrees in this circuit is also possible by introducing more than 1 hidden layer, creating a form of branching circuit. The implementation for this in a distributed way remains very simple, as the considered dynamics of the form $z\to Gz$ are linear (token-propagation dynamics). The bound on the number of edges needed to represent such a circuit remains $O(nd)$, even when the circuit has constant degree.

The technical construction of the linear circuits $H^{\mathfrak{e}}$, $H^{\mathfrak{i}}$ provided in this Appendix do not affect results concerning the analysis of the structure of neuron-neuron interaction graphs $G^{\mathfrak{e}}$, $G^{\mathfrak{i}}$. These neuron-neuron interaction graphs plausibly maintain a heavy-tailed, power-law-like degree distribution, as is the case for the models considered empirically in Section [5.5](#S5.SS5).

### C.4 Formal statement of Claim 4

###### Claim 9 .

Let ${D_{y}},E$ be parameter matrices of BDH-Normfree. Then, there exists a graph ${G_{y}}\in\mathcal{G}(n,O(nd))$, expressible through a sparse linear circuit, a graph ${G_{s}}$ having $O(nd)$ edges, and a sparse linear value preparation function $A:{R^{+}}^{n}\to{R^{+}}^{n}$, such that, for any sequence of keys $(x_{\tau,l})_{0\leq\tau\leq t}$ and values $(y_{\tau,l-1})_{0\leq\tau\leq t}$, with $x_{\tau,l},y_{\tau,l-1}\in{R^{+}}^{n}$, we have:

$$ $({G_{y}}^{\mathfrak{e}}-{G_{y}}^{\mathfrak{i}}){\boldsymbol{\sigma}}^{*}_{t-1,l}x_{t,l}={D_{y}}E{\boldsymbol{\sigma}}_{t-1,l}x_{t,l},$ $$

where ${\boldsymbol{\sigma}}_{t-1,l}=\sum_{\tau<t}{y_{\tau,l-1}}{x_{\tau,l}}^{T}U^{t-\tau}$ represents the attention state of BDH-Normfree following Eq. ([16](#S6.E16)), and ${\boldsymbol{\sigma}}^{*}_{t-1,l}=\left(\sum_{\tau<t}{A(y_{\tau,l-1})}{x_{\tau,l}}^{T}U^{t-\tau}\right)\odot{G_{s}}$ represents the corresponding attention state of the BDH system with sparse attention on graph ${G_{s}}$, subject to appropriate preparation of attention values using function $f_{y}$.

Before we start the proof, we make a general point about the formulation of the claim. We are considering the problem of expressing (or more generally, approximating) the matrix operator ${\boldsymbol{\sigma}}_{t-1,l}$ by another, sparser one. The setting of our problem can be distilled into obtaining an equality or approximation of the form $E{\boldsymbol{\sigma}}_{t-1,l}\approx E^{*}{\boldsymbol{\sigma}}^{*}_{t-1,l}$, where $E\in R^{d\times n}$ is a given low-rank matrix, $E^{*}\in R^{d\times n}$ can be defined arbitrarily, and ${\boldsymbol{\sigma}}^{*}$ is defined as in the statement of the Claim. If we content ourselves with an approximation, then it is possible to have ${\boldsymbol{\sigma}}^{*}={\boldsymbol{\sigma}}$ (i.e., put $f_{y}=idem$), using for example the stochastic sparsification framework of , or a value-dependent variant (cf. e.g. ). The samples chosen by such a framework in a value-dependent variant would lead to a graph $G_{s}$ which plausibly reflects the power-law element distributions that we empirically observe in ${\boldsymbol{\sigma}}$.

Figure: Figure 17: Non-uniform graph attention: interpretation of $E({\boldsymbol{\sigma}}_{l,t}\odot G_{s})$ after sparsification of graph $G_{s}$.

While the spirit of such an approximation is generally valid, we opt in the proof for a simpler, purely technical argument applicable to our specific setting, which gives a strict equality in the statement of Claim [9](#Thmclaim9) subject to linear preparation of attention values with a function $A$. In practice, this would mean that two successive layers of BDH with sparse state are sufficient to express a layer of BDH-Normfree under this reduction.

To prove the claim, it is enough to embed the connection structure of the encoder matrix, treating it as a graph, into ${G_{s}}$.

###### Proof.

(of Claim [9](#Thmclaim9))
Fix arbitrarily subset $D\subseteq V$ of neurons, with $|D|=2d$. For the given matrix $E\in R^{d\times n}$ from BDH-GPU, let $E^{\prime}\in(R^{+})^{2d\times n}$ be defined as in the proof of Claim [3](#Thmclaim3) in Appendix [C.3](#A3.SS3), and let ${D_{y}}^{\mathfrak{e}}$, ${D_{y}}^{\mathfrak{i}}$ also be applied as in that proof for considerations of decoder ${D_{y}}$. Define the value preparation function $A$ as the immersion of vectors over $V$ into $D$ using $E^{\prime}$. Define ${G_{s}}$ to be the all-ones matrix on the $2d$ columns corresponding to $D$, and zeros elsewhere. Then, define $E^{*}\in R^{2d\times n}$ to be a diagonal matrix acting on its first $2d$ elements (corresponding to $D$), and zeros elsewhere. Setting ${G_{y}}^{\mathfrak{e}}={D_{y}}^{\mathfrak{e}}E^{*}$ and ${G_{y}}^{\mathfrak{i}}={D_{y}}^{\mathfrak{i}}E^{*}$, we obtain the claim.
∎

###### Claim 9 .

###### Proof.

## Appendix D Desirable properties of a local graph dynamics for language models

We outline several general criteria of computational expressiveness and computational efficiency which a distributed computing system has to meet to effectively deal with language and reasoning. For this, we take a first-principles approach, relying only on very fundamental properties which an attention-based language model appears to need to capture, and which are applicable far beyond the specific case of BDH --- plausibly, being equally applicable to human and human-like reasoning.(^18^18 18 In particular, the reader will have no doubt observed that graph settings applicable to language inference and reasoning systems, which involve task inputs spread out over time and the emergence of graph structure, are very different from graph-based frameworks which directly associate the task to solve with the communication graph (the latter case includes most considerations of: Graph Neural Networks, Graph Transformers, the LOCAL/CONGEST model of distributed computing, Approximate Message Passing systems, etc.))

###### Hypothesis 2 .

We expect any efficient graph-based distributed system dealing with language and reasoning using an attention-based approach to have the following characteristics:

- •
[No Easy Simulation] The system achieves computationally irreducible dynamics, i.e., it provides no systematic opportunity to predict the outcomes of its inference or approximate its dynamics in a numerically easier way than by running the system itself.
- •
[Particles Talk] The state-space dynamics of the distributed system is a non-linear interacting particle dynamics, i.e., the system does not admit an efficient representation as a non-interacting particle system, but relies on a form of non-linear evolution expressed through (at least) two-particle interactions. (Such interactions are necessary, in particular, to enable multi-point correlation analysis on language inputs, when assuming only a small number of inference steps of the system per output token.)
- •
[Attention Deforms Pairwise Connections] The system is capable of computing correlations between pairs of scalar variables localized at different nodes of the distributed system, and storing the state of such correlations so that the result is accessible from these two nodes. (This is plausibly needed to express attention in a state-space system.)
- •
[Time Dictates Structure] The communication graph of the distributed system does not, in itself, represent any specific task input to solve, but reflects a trained model (a program), whereas tasks are represented as inputs to this program, presented over time. The communication graphs used to solve language and reasoning problems are expected to display modular, scale-free structure.

*A detailed discussion of the four items of the Hypothesis is provided below.*

###### Hypothesis 2 .

#### [No Easy Simulation] ⋄ \diamond Computational models have irreducible dynamics.

We start by recalling a general observation which is applicable to most learning systems $L$ (machine learning models, biological systems) that have learned how to do computations: they are likely to have chosen state-space dynamics that will allow them to resolve their computational problem with the least effort during inference. In other words, *if there is a physical system $P$ that solves a given computational problem, and if there exists a simulation $S(P)$ of this physical system that would approximate system $P$ with less effort, the learning system $L$ will be following the dynamics of $S(P)$, not those of $P$.*

We provide a few hypothetical examples for intuition, anchored in different areas of particle dynamics.

If $P$ were the particle dynamics of electrons in a resistor network, the simulation $S(P)$ could be a calculation based on Ohm’s law with a Laplacian solver — and we would consequently expect the dynamics of our computational system $L$ to follow the Laplacian solver code, and not to simulate electron dynamics.

If $P$ were the ensemble of billions of Internet users performing short walks clicking through links of the world wide web, the simulation $S(P)$ would be a calculation of aggregate behavior, reminiscent of PageRank, and we would expect $L$ to encode the parallel dynamics of Map-Reduce matrix operations of PageRank, not the simulation of individual agents.

If $P$ were a quantum system amenable to approximation by perturbation theory, we would expect $L$ to simulate the (classical) calculus of this perturbation theory, and not the quantum system $P$ directly.

Most mechanical systems admit some form of more efficient simulation, which means the the dynamics of such systems are rarely a suitable choice for neuronal models. Anecdotally, in nature, only very simple systems like the Physarum slime mold  rely on direct action (with hydrostatic pressure gradients) to perform their optimization process; and contemporary neuroscience research suggests that even the simplest neuronal brains do not perform their work in a similar “fluid-mechanical” manner.

The irreducibility of $L$ means that this system is stretched to the limits of stability, just as a highly optimized numerical algorithm would be have been simplified and optimized to the limit of numerical stability. This relates to the limits of dimensionality reduction techniques that we have explored through a largely equivalent information-lens perspective of loss of precision and loss of information which it inflicts upon the model.

#### [Particles Talk] ⋄ \diamond Latent concept spaces arise from outcomes of particle-particle interactions.

Dynamics of systems with multiple particles moving around in a (deformable) environment fall into two broad categories, depending on the strength of interaction between different parts of the dynamics. In the simpler setting, particles can be assumed *at short time scales* to be moving in an environment unchanged by other particles — the concurrent action of other particles, which would change the environment, does not need to be taken into account when representing individual particle motion, nor is it necessary to consider particle-particle interactions. By contrast, in the more general setting, the dynamics of multiple particles are tightly coupled, and their dynamics need to be modeled (simulated) together.

An example of a dynamics with no coupling would be a dynamics of multiple independent random walkers, such as the previously mentioned dynamics of electricity in wires, or the dynamics of PageRank. Examples of dynamics including interactions between particles, which may either happen directly or be moderated through the environment, include cellular automata, particle method simulations and molecular simulations, or swarms of communicating agents.

The natural representation of state-space models as moving particles comes from the following interpretation. A distributed system with depth-$L$ computations (not least BDH or the BDH-GPU model given by the state equations ([4](#S3.E4))) is amenable to interpretation as a system of walker particles performing an $L$-step walk over layers, starting at some token $t_{0}$ in the input layer $0$ and, in each time step $t\geq t_{0}$, either pausing (skipping a time step) or moving on to the next layer, until they reach the last layer $L$ in some time step $t_{f}$, at which point they leave the system, contributing to the distribution of the $t_{f}$-th output token. When attempting this approach with *independent* walkers, the distribution of tokens output by such a system could be described by correlation functions following or resembling the Dyson series, $\sum_{\tau_{L}=0}^{t}\sum_{\tau_{L-1}=0}^{\tau_{L}-1}\ldots\sum_{\tau_{1}=0}^{\tau_{2}-1}F(\textrm{input}(\tau_{1}),\ldots,\textrm{input}(\tau_{L}))$. However, the output of attention (e.g., the linear attention output $a^{*}$ given by equation ([4](#S3.E4)) for BDH-GPU, or defined similarly in other state space models based on linear attention), cannot be represented as a Dyson formula when unrolling the dynamics backwards through layers (even if it looks deceptively similar at first glance). Each entry retrieved from attention is an interplay between two moments of time: the moment at which the key-value pair was entered, and the moment at which the corresponding query arrived. In consequence, the considered dynamics can be represented, in each layer, as a linear sum of two-point correlations between current time $t$ and some point $\tau$ in the past. Thus, in the $l$-th layer, this recursion can (with some approximation) be unrolled into a linear combination of functions of sets of $2^{l}$ input tokens (provided in the $0$-th layer), but cannot be represented through correlation functions $F$ on smaller sets of tokens (e.g., of size linear in $l$). Otherwise put, a system like BDH can be described using particles performing $l$-step walks when *relying on intermediate elements of KV-state ${\boldsymbol{\sigma}}$*, which are produced during interactions with other walker particles in intermediate layers, but needs to be viewed through at least $2^{l}$-point correlation functions defined directly on input tokens in the input layer.

The considered point is relevant because it *precludes many forms of modeling of attention-based language dynamics, in particular those using non-interacting particle theories*. The precluded approaches include:

- •
$L$-grams, word2vec-like $L$-skip-grams , as well as any other $L$-point correlations of past input tokens.
- •
$L$-step non-interacting random walk models (walks inside the network structure, which move from input layers towards output layers across time).
- •
systems known to be equivalent to the above, such as approximations of classical spin-chain systems by means of Feynman integral path lengths bounded by $L$ , and many forms of graph/GNN kernels based on $L$-th powers of the graph Laplacian.
- •
by extension, $L$-layer state-space systems which perform excessive compression (size reduction) of their state, in a way which eliminates most long-term correlations.

We can ask if this requirement for communication between particles is an artifact of the construction of BDH (and similarly, of the Transformer), or if it comes from a genuine need related to language and reasoning tasks. For language problems per se, the need for multi-point token correlation in $L$-layer language modeling plausibly follows from the expectation that the model should have the ability to create a syntax tree of a sentence by means of a single quick parallel scan over words in this sentence. With this assumption, the depth $L$ of computation used to build a language syntax tree should be sufficient to represent the number of *levels* of the syntax tree that the model is able to process naturally, but can be (and in general, should plausibly be) much smaller than the number of *leaves* (words) of this syntax tree. This is consistent with the RASP-L-based understanding of the Transformer’s capabilities, which allows for expressing depth-L trees in a depth-L Transformer.(^19^19 19 This does not mean the problem is easy; synthetic problems inspired by this type of tree problem were (for us) among the hardest to train into a Transformer with no Chain-of-Thought — as compared to RASP-L problems described in and others we tested.)

Such a way of mapping the tree structure of problems into the model’s layers, from bottom to top, also essentially captures the “generative” nature of the considered models, which rely on concept spaces created and stored in state in intermediate layers, to guide both language comprehension and reasoning on language. Thus, the ability to handle language syntax trees efficiently, in itself, precludes the previously-mentioned types of modeling approaches.

#### [Attention Deforms Pairwise Connections] ⋄ \diamond The interaction process X ⁡ ( i ) , Y ⁡ ( j ) → σ ⁡ ( i , j ) X(i),Y(j)\to\sigma(i,j) describes attention.

The preceding discussion in paragraph [Particles Talk] grounds state-of-the-art state-space language models in the world of interacting particle systems.

Whenever the global vector-based description of a state-space model calls for a three-point operation, such as the trilinear operation of key-value-query attention, this translates into the nature of pairwise (for polynomial interaction terms, degree-two) non-linear particle interactions in the transition equations of the same model when described at the level of particles. Notably, at scale, *the state-space transition equations of an attention-based model plausibly involve altering or deforming correlation strength between pairs of particles, with such pairs being represented as interaction variables in the state of the system*. This requirement on structure, repeated across layers, can be seen as sufficient: interactions of particle pairs are about the only requirement on non-linear rulesets that the system needs to be support, as demonstrated by the simple local transition rules of BDH.

Overall, the statement “attention is all you need”, which describes a system-level global property,
translates into “${X(i),Y(j)\to\sigma(i,j)}$ is all you need” at the level of particle dynamics of a state-space language model.

#### [Time Dictates Structure] ⋄ \diamond Inputs to reasoning problems are sequential, not graph-based.

Many real-world graphs are anchored in a spatial embedding of their nodes which is given by external constraints. For example, the structure of many social and transportation networks is impacted by the geographical placement of people and infrastructure on the globe.

In designing the dynamics for BDH, we are free from such spatial constraints. The graph topology corresponding to the model is free to take the shape needed to best resolve the problem. The problem itself is encoded as a sequence of tokens which arrive over time to the model (we take here a state-space view of the system).
We can naturally presume that the structure of the model graph of BDH is shaped in a way which follows from two aspects: this temporal encoding of information, and from the abstract (Platonic) latent space of concepts needed to deal with language and reasoning.

When looking for the right particle dynamics for language models, it seems reasonable to discard all *unnecessary* aspects of spatial constraints.

One example of a particle interaction system which includes externally imposed constraints on the structure of the state space is that of cellular automata operating on a two-dimensional grid. While 2D cellular automata have appealed to public imagination, appearing in attempts to observe the emergence of intelligence at least since the 1970’s, they are, in fact, an extremely cumbersome choice for representing in-context reasoning or language for any attention-based model. State-of-the-art language models seem to have no structural need for a low-dimensional grid in their dynamics. Arguably, the connection structure which needs to emerge in a graph system, allowing it to work efficiently in a setting of efficient information search
is precisely the opposite: it is a multi-scale, expander-like system of shortcuts, cf. e.g. . This scale-free graph structure is expected to correspond to the scale-free temporal behavior observed in natural systems .

In the rest of this paragraph we briefly review other areas of computer science, and how they relate to the particle dynamics we are looking for in terms of their relationship to handling temporal inputs and the constraints they impose on the structure of the state-space.

The freedom of choice of graph topology in solving problems around language and in-context reasoning, which we are dealing with here, can be contrasted with settings in which the graph is, at the same time, part of the system dynamics (encoding interactions in the system) and a part of the statement of the problem input. This is particularly true for models of distributed computing inspired by computer networking (LOCAL, CONGEST, etc.) and other forms of interaction networks (Approximate Message Passing, quantum LOCC, etc.), where the same graph $G$ represents the communication network for the dynamics, and encodes the problem input — with the required output being some function of $G$ (e.g., a clustering, coloring, spanning tree, etc.). Some distributed problems on graphs can be formulated so that the input and required output are independent of the graph structure, the notable ones being: majority consensus, leader election, information broadcasting, and computing aggregates. For such problems, the graph represents only a communication system, whose topology is more an obstacle to overcome, than an actual help in solving the problem. This applies also to architectures in Machine Learning which adhere to a known graph structure, such as Graph Neural Networks or Graph Transformers, when solving problems whose inputs are not naturally embedded in such a structure.

A handful of approaches in distributed computing are intended to describe systems which compute a function of an input signal which, like language, is spread out sequentially over time, and where computations happen while this signal is still arriving. In particular, some forms of particle dynamics can be distilled from the theory of self-stabilizing systems , giving rise to settings where the system is expected to adapt its state in response to a time-changing input (see e.g. ).
Among distributed streaming frameworks, one approach which, owing to its design, admits an elegant particle-based interpretation for time-changing inputs, is the incremental computing framework . This framework emphasizes temporal commutativity, and is well suited to expressing dynamics of non-interacting particles, such as PageRank-like computation performed incrementally with Map-Reduce on time-changing graphs, or building nearest-neighbor indexes on sets of changing vectors. It does not naturally extend to the non-linear particle-particle interaction dynamics that appear in the context of attention (see paragraph [Particles Talk]).

## Appendix E BDH-GPU PyTorch code listing

The code listing below implements BDH-GPU (Definition [4](#Thmdefinition4)) for PyTorch version 2.7. It is self-contained, except for the implementation of RoPE which needs to be filled by the user. With respect to the state dynamics of Eq. ([8](#S3.E8)), it provides an extension supporting heads. The placement of layer norms and residual connections is modified with respect to Eq. ([8](#S3.E8)); in general, this aspect offers some flexibility.

This implementation assumes the simplest case of a fixed context window of length $T$. An unbounded context window is technically supported using a state-space kernel for Linear Attention, and works best following appropriate adaptation of the model for truncated backpropagation through time (see Appendix [B.2](#A2.SS2)).