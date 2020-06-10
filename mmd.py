import torch


def ed_kernel_poly(X, Y, gamma=2):
    eps = 1E-5
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())

    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)

    sq_diff_norms = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() + eps
    sq_diff_norms = torch.clamp(sq_diff_norms, min=eps) # remove erroneous negative values

    vect_norms = Z.norm(dim=1)
    A = sq_diff_norms.sqrt()
    A = A**(1/gamma)

    K = vect_norms.reshape(-1, 1) + vect_norms.reshape(1, -1) - A
    K_XX, K_XY, K_YY = K[:m, :m], K[:m, m:], K[m:, m:]

    mmd2_D = _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False)
    return mmd2_D


def ed_kernel(X, Y):
    eps = 1E-5
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())

    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)

    sq_diff_norms = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() + eps
    sq_diff_norms = torch.clamp(sq_diff_norms, min=eps) # remove erroneous negative values

    vect_norms = Z.norm(dim=1)
    A = sq_diff_norms.sqrt()
    K = vect_norms.reshape(-1, 1) + vect_norms.reshape(1, -1) - A
    K_XX, K_XY, K_YY = K[:m, :m], K[:m, m:], K[m:, m:]

    mmd2_D = _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False)
    return mmd2_D


################################################################################
# Helper functions to compute variances based on kernel matrices
# Source: https://github.com/OctoberChang/MMD-GAN
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X).reshape(1)
        sum_diag_Y = torch.sum(diag_Y).reshape(1)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2