# Low Rank Singular Value Decomposition
using LinearAlgebra

# Construct LowRankSVD DataStructure
"""
    LowRankSVD <: Factorization
Matrix factorization type of the singular value decomposition (SVD) approximation of a low rank matrix `A`.
LowRankSVD is return type of [`svd_lowrank(_)`](@ref), the corresponding matrix factorization function.
If `F::SVD` is the factorization object, `U`, `S`, `V` and `Vt` can be obtained
via `F.U`, `F.S`, `F.V` and `F.Vt`, such that `A ≈ U * Diagonal(S) * Vt`.
The singular values in `S` are sorted in descending order.
"""
struct LowRankSVD{T,Tr,M<:AbstractArray{T}} <: Factorization{T}
    U::M
    S::Vector{Tr}
    Vt::M

# Inner Constructor Methods
    function LowRankSVD{T,Tr,M}(U, S, Vt) where {T,Tr,M<:AbstractArray{T}}
      size(U,2) == size(Vt,1) == length(S) || throw(DimensionMismatch("$(size(U)), $(length(S)), $(size(Vt)) not compatible"))
      new{T,Tr,M}(U, S, Vt)
    end
end

# Outer Constructor Methods
LowRankSVD(U::AbstractArray{T}, S::Vector{Tr}, Vt::AbstractArray{T}) where {T,Tr} = LowRankSVD{T,Tr,typeof(U)}(U, S, Vt)
function LowRankSVD{T}(U::AbstractArray, S::AbstractVector{Tr}, Vt::AbstractArray) where {T,Tr}
    LowRankSVD(convert(AbstractArray{T}, U),
                convert(Vector{Tr}, S),
                convert(AbstractArray{T}, Vt))
end

# 
function get_approximate_basis(
    A::AbstractArray{T}, l::Int64, niter::Int64 = 2, M::Union{AbstractArray{T}, Nothing} = nothing) where T
    
    """Return tensor :math:`Q` with :math:`l` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`.
    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.
    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, l, can be
              choosen according to the following criteria: in general,
              :math:`k <= l <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`l = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`l = k + 0..2` may be sufficient.
    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator
    Args::
        A (AbstractArray{T}): the input tensor of size :math:`(m, n)`
        l (Int64): the dimension of subspace spanned by :math:`Q`
                 columns.
        niter (Int64, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.
        M (AbstractArray{T}, optional): the input tensor of size
                              :math:`(m, n)`.
    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    m, n = size(A)
    Ω = rand(T, (n, l))

    Y = zeros(T, (m, l))
    Y_H = zeros(T, (n, l))

    if M === nothing 
        F_j = qr!(mul!(Y, A, Ω))
        for j = 1:niter
            F_H_j = qr!(mul!(Y_H, A', Matrix(F_j.Q)))
            F_j = qr!(mul!(Y, A, Matrix(F_H_j.Q)))
        end
    else
        Z = zeros(T, (m, l))
        Z_H = zeros(T, (n, l))
        F_j = qr!(mul!(Y, A, Ω) - mul!(Z, M, Ω))
        for j = 1:niter
            F_H_j = qr!(mul!(Y_H, A', Matrix(F_j.Q)) - mul!(Z_H, M', Matrix(F_j.Q)))
            F_j = qr!(mul!(Y, A, Matrix(F_H_j.Q)) - mul!(Z, M, Matrix(F_H_j.Q)))
        end
    end
    Matrix(F_j.Q)
end

function _svd_lowrank(A::AbstractArray{T}, l::Int64, niter::Int64 = 2, M::Union{AbstractArray{T}, Nothing} = nothing) where T
    
        m, n = size(A)
    if M === nothing
        Mt = nothing
    else
        Mt = transpose(M)
    end
    At = transpose(A)

    if m < n || n > l
        """
        computing the SVD approximation of a transpose in
        order to keep B shape minimal (the m < n case) or the V
        shape small (the n > l case)
        """
        Q = get_approximate_basis(At, l, niter, Mt)
        Qc = conj(Q)
        if M === nothing
            Bt = A * Qc
        else
            Bt = A * Qc - M * Qc
        end
        U, S, Vt = svd!(Bt)
        Vt = Vt * Q'
    else
        Q = get_approximate_basis(A, l, niter, M)
        if M === nothing
            Bt = Q' * A
        else
            Bt = Q' * (A - M)
        end

        U, S, Vt = svd!(Bt)
        U = Q * U
    end
    return LowRankSVD(U, S, Vt)
end

function svd_lowrank(A::AbstractArray{T}, l::Int64, niter::Int64 = 2, M::Union{AbstractArray{T}, Nothing} = nothing) where T
    """Return the singular value decomposition LowRankSVD(U, S, Vt) of a matrix
    or a sparse matrix :math:`A` such that
    :math:`A ≈ U diag(S) Vt`. In case :math:`M` is given, then
    SVD is computed for the matrix :math:`A - M`.
    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al, 2009.
    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator
    .. note:: The input is assumed to be a low-rank matrix.
    .. note:: In general, use the full-rank SVD implementation
              :func:`torch.linalg.svd` for dense matrices due to its 10-fold
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              :func:`torch.linalg.svd` cannot handle.
    Args::
        A (AbstractArray{T}): the input matrix of size :math:`(m, n)`
        l (Int64): a slightly overestimated rank of A.
        niter (Int64, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2
        M (AbstractArray{T}, optional): the input tensor of size
        :math:`(m, n)`.                

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <https://arxiv.org/abs/0909.4061>`_).
    """
    return _svd_lowrank(A, l, niter, M)
end
