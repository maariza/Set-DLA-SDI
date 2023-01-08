# Set-DLA-SDI

The original Diffusion Limited Aggregation (DLA) algorithm is a discrete version of dendritic growth and is best known for analyzing the formation of branching structures. Nowadays, a computational tool that allows slightly more accurate measurements of complex systems, such as neurons, is a compound set of algorithms: the modified DLA algorithm (mDLA) and SDI index.
The mDLA  is a modified version of the original DLA that allows to replicate binary objects through random walks. The mDLA has as initial state an immovable particle, called seed, located in a network. Following this, the mDLA algorithm adds a second random particle, far from the origin, capable of moving along the network until it finds an adjacent location to the seed. If found, the second particle adheres and becomes part of the cluster. The particles are added, one by one, until the end of the simulation. 

The shape diffusivity index (SDI) is used to quantify spatial relationships and measure the ''diffusivity'' of the analyzed images. For comparing the distributions, the Kullback-Leibler divergence, known as KL-divergence was used. This quantifies how much one probability distribution differs from anothe.
A scale between 0 and 1 was used. Complex forms close to normal DLA took values closer to 1 and decreased to 0 as the complexity of the figure decreased. The index was verified with two images, one with low (Radial Glial cell) and one with high (mature Purkinje cell) complexity. The code was run seven times per neuron to verify if the SDI was consistent in each simulation. 


