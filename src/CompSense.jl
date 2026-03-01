# Copyright (c) 2026
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

module CompSense

using LinearAlgebra
using Convex
using SCS
using Random
using Combinatorics

# Sparse recovery algorithms
export IRWLS, L0EM, SL0, OMP, FISTA, IHT, CoSaMP, AKRON, KRON, ReweightedL1,
       BasisPursuit, BPDN, LASSO, SP, NIHT, ADMM, AMP

# Sensing matrix generators
export gaussian_sensing,
       bernoulli_sensing,
       fourier_sensing,
       dct_sensing,
       hadamard_sensing,
       sparse_sensing,
       uniform_sensing,
       toeplitz_sensing

# Utilities
export generate_sparse_signal, add_noise

# Metrics
export recovery_error, support_recovery, snr, nmse, phase_transition

# Sensing matrix analysis
export mutual_coherence, babel_function, spark, column_coherence_matrix

# Deprecated (kept for backward compatibility)
export cs_model

include("IRWLS.jl")
include("L0EM.jl")
include("SL0.jl")
include("OMP.jl")
include("FISTA.jl")
include("IHT.jl")
include("CoSaMP.jl")
include("AKRON.jl")
include("KRON.jl")
include("ReweightedL1.jl")
include("BasisPursuit.jl")
include("BPDN.jl")
include("LASSO.jl")
include("SP.jl")
include("NIHT.jl")
include("ADMM.jl")
include("AMP.jl")
include("utils.jl")
include("metrics.jl")
include("analysis.jl")

end
