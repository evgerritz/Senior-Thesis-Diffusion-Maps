# Senior-Thesis-Diffusion-Maps
## Abstract

Many humans have the remarkable ability to
correctly guess the artist of a painting they have
never seen before, so long as they have seen
enough other paintings by that artist. This fact
implies there may exist an invariant between
multiple works by one author, which we will
refer to as “style.” A few questions then naturally arise: 1. Can we teach a computer to
learn such an invariant? 2. If so, by analyzing
what the computer does, can we arrive at an understanding of how humans perform this task?
3. How can we be sure that the computer is
truly learning the invariant and not memorizing
image statistics?  

We looked at Chinese calligraphy as an example of a particularly challenging instance of this
problem. We hypothesized that the output of
an intermediate layer of a sufficiently advanced
deep neural network can be used as a kernel to
learn a manifold of an invariant of calligraphic
style. We used the diffusion maps nonlinear
dimensionality reduction algorithm to assess a
given kernel’s ability to cluster images of characters never seen by the neural network, and
were ultimately able to learn a kernel that produced compelling clusters of calligraphers on
the learned manifold of calligraphic style.


## Report and Poster
- ![Report](CPSC490Report.pdf)
- ![Poster](CPSC490Poster.pdf)
