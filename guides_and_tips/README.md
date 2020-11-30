# Computational Mathematics 
**Project 17**

(M) is a so-called extreme learning, i.e., a neural network with one hidden layer, y=W2σ(W1x)
, where the weight matrix for the hidden layer W1 is a fixed random matrix, σ(⋅) is an elementwise activation function 
of your choice, and the output weight matrix W2 is chosen by solving a linear least-squares problem 
(with L_2 regularization).

* (A1) is an algorithm of the class of accelerated gradient methods 
[references: https://www.cs.cmu.edu/~ggordon/10725-F12/slides/09-acceleration.pdf, 
http://www.cs.toronto.edu/~adeandrade/assets/aconntmftc.pdf,
https://arxiv.org/pdf/1412.6980.pdf] applied to (M).

* (A2) is a closed-form solution with the normal equations and your own implementation of Cholesky (or LDL) factorization.

No off-the-shelf solvers allowed.
Some tips from project guideline:

### 2.2  Setting the stage 
The first section of your report should contain a description of the problem and the methods that you plan to use.
This is just a brief recall, to introduce notation and specify which variants of the methods you plan to use. 
Your target audience is someone who is already familiar with the content of the course. There is no need to repeat a 
large part of the theory: we are sure that you know how to do that, given enough time, books, slides, and internet bandwidth.
In case adapting the algorithm to your problem requires some further mathematical derivation 
(example: developing an exact line search for your problem, when possible, or adapting an algorithm to deal more 
efficiently with the special structure of your problem), you are supposed to discuss it here with all the necessary 
mathematical details. Discuss the reasons behind the choices you make (the ones you can make, that is, since several of 
them will be dictated by the statement of the project and cannot be questioned). We suggest to send us a version of this
section by e-mail as soon as it is done, so that we can catch misunderstandings as soon as possible and minimize the 
amount of work wasted. Note that we do not want to see code at this point: that would be premature to produce (for you) 
and unnecessarily complicated to read (for us).

### 2.3  What to expect from the algorithm(s)
Next, we expect a brief recall of the algorithmic properties that you expect to see in the experiments. 
Is the algorithm(if it is iterative) guaranteed to converge? Is it going to be stable and return a good approximation of
the solution(if it is direct)? What is its complexity? Are there any relevant convergence results? Are the hypotheses of
these convergence results (convexity, compactness, differentiability, etc.) satisfied by your problem? If not, what are
he“closest” possible results you have available, and why exactly are they not applicable?  Do you expect this to 
be relevant in practice? Each time you use some specific result (say, a convergence theorem), please be sure to report 
in detail what the assumptions of the result are, what consequences exactly you can derive from them, and the source where
you have taken it (down to the number of theorem/page). Also, please be sure to discuss in detail why the assumptions 
are satisfied in your case, or which assumptions are not satisfied or you cannot prove they are. Again, we suggest to
send us a version of this section by e-mail as soon as it is done. No code yet.

### 2.4  Write your code
Coding the algorithms is a major part of the project. Languages that are often convenient for numerical computation are
(depending on the task) Matlab, Python, and C/C++, but you can use any reasonable programming language. You are expected
to implement the algorithm yourself; it should not be a single line of library call. However,you can use the numerical 
libraries of your language of choice for some of the individual steps: for instance, you can use numpy.linalg.norm to 
evaluate the error, or Matlab’s A \ b to solve a linear system that appears as a sub-step(unless, of course, writing a
linear solver to compute that solution is a main task in your project). You can (and should) also use numerical libraries
to compare their results to yours: for instance, you can check if your algorithm is faster or slower than Matlab’squadprog,
if it produces (up to a tolerance) the same objective value, or how the residual of your solution compares with that produced
by A \ b. When in doubt if you should use a library, feel free to ask us.Your goal for this project is implementing and
testing numerical algorithms: software engineering practices such asa full test suite, or pages of production-quality 
documentation, are not required. That said, we appreciate well-written and well-documented code (who doesn’t?). You are
free to use tools such as git to ease your work, if you are familiar with them (but giving us a pointer to the git 
repository is not the expected way to communicate with us).

### 2.5  Choose and describe the experimental set-up 
Next, we expect a brief description of the data you will test your algorithms on. For “ML projects” this will typically 
be provided by the ML course, but still a modicum of description is required. For “no-ML projects”, it will typically 
have to be either generated randomly, or picked up from the Internet, or a combination of both. This is not always 
trivial: the random generation process can be tweaked to obtain “interesting” properties of the data (what kind
of solution can be expected, how well or ill-conditioned the problem is, . . . ). These aspects should be described in
there port. You are supposed to test the algorithm on a realistic range of examples, in terms of size, structure, and 
sparsity: it is typically not OK if your largest example is 10×10. Get a sense of how algorithms scale, and what is the
maximum size of problems that you can solve reasonably quickly on an ordinary machine.Numerical experiments have two
purposes:
* Confirm that the methods work as expected: how close do they get to the true solution of the problem? How can you
check it? Is there a “residual”, or “gap” value that you can check? Do they converge with the rate(linearly, sublinearly,
 superlinarly, . . . ) that the theory predicts? Does the error decrease monotonically, if it is expected to do so?
* Evaluate trade-offs and compare various algorithms: which algorithm is faster?  If algorithm A takes fewer iterations
than algorithm B, but its iterations are more expensive, which one is the winner?  How does this depend on the
characteristics of the problem to be solved (size, density, . . . )?

Comparison with off-the-shelf algorithms is also welcome (and often useful to check correctness) to assess whether your
approach could ever be competitive under the right conditions (and what these are). Setting thresholds and algorithmic
parameters is a key and nontrivial aspect. This is one of the “dark secrets” of numerical algorithms: basically any
algorithm you can write has parameters, and it will misbehave if you don’t set them properly.
They can have a huge impact on performance. Thus, a minimal testing activity about the effect of these parameters is
almost surely needed. A full-scale test à-la hypermetameters optimization in ML is also possible,and welcome, but this
is anyway different from the one in ML, even for “ML projects”. The properties of interest are fundamentally different: 
in ML one looks for learning (accuracy, recall, . . . ), while in this course one is looking at“how close the solution
I got is to what I would have liked to get, and how costly was it to get there”. Hence, reporting here the same
hypermetameters optimization tables done for ML does not cut it. Not that these cannot be reported,a s they still give
some potentially interesting information, but it is not the information we are crucially interested in.When designing
your experiments, and later your graphs and/or tables, you should have these goals in mind. To quote a famous
mathematician, “the purpose of computing is insight, not numbers.”

### 2.6  Report results of your experiments
A few plots and/or tables are required to display your results. Have a purpose in mind: as for the choice of experiments,
the plots should display the important features of the algorithm: convergence speed, residual, computational time, . . .
Plots should be readable. If 90% of your plot are barely-visible horizontal or vertical lines, look for a better way 
to display information. Logarithmic scales are usually a good idea to display quantities (such as residuals or errors) 
that vary by orders of magnitude. In particular, a sequence with exact linear convergence (errork+1=C·errork) becomes a 
straight line in logarithmic plots; hence they display well the convergence speed of algorithms.For “ML projects”, 
the quantities you want to plot may be quite different from these you would in a ML course.While the “learning rate” (here “convergence rate”)
 is the same, the out-of-sample performance is much less relevant for us.