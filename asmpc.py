import numpy as np
import pandas as pd
import scipy as sc
import statsmodels.api as sm

from secretsharing import (
    secret_int_to_points,
    points_to_secret_int,
    get_large_enough_prime,
)

np.random.seed(0)


def dot(x):
    return sum(x * x)


# Since the secretsharing library does not support floats, we use fixed point encoding to transform them into integers. np.int64 is not supported, we use regular python int.
fpFactor = int(np.exp2(23))
# For of use in this prototype we chosse a fixed factor, a power of 2 to introduce less numerical error (theoretically).

# TODO: Convert this to numpy, using object arrays or tensors.

fpFactSq = fpFactor * fpFactor
fpFactDiv = fpFactor / fpFactor


def toInt(x):
    return int(x * fpFactor)


def toIntVec(v):
    return [toInt(x) for x in v]


def toIntMat(M):
    return [toIntVec(v) for v in M]


def toInt3D(Ms):
    return [toIntMat(M) for M in Ms]


# TODO: Since / is an ufunc we should be able to get toFloat to broadcast to vectors and matrices.
def toFloat(x):
    return np.float64(x / fpFactor)


def toFloatVec(v):
    return np.array([toFloat(x) for x in v])


def toFloatMat(A):
    return np.array([toFloatVec(v) for v in A])


def compute_D(Ns, K):
    D = sum(Ns) - K - 1
    return D


# Generate an N vector, with the first elements being "Ns" and the rest being "default" up to the number of players.
def gen_N_vec(Ns, default, players):
    Ns.extend((players - len(Ns)) * [default])
    return Ns


def generate_player(N, M, K):
    y = np.random.standard_normal(N)
    X = np.random.standard_normal(N * M).reshape(N, M)
    C = np.random.standard_normal(N * K).reshape(N, K)
    R = np.linalg.qr(C)[1]
    # The QR decomposition could also be distributed/parallelized.
    return (y, X, C, R)


# This takes three equally long vectors of Ns, Ms, Ks.
def generate_player_vec(Ns, M, K):
    players = [generate_player(N, M, K) for N in Ns]
    return players


def player_compute(player, invR):
    y, X, C, R = player
    Q = np.matmul(C, invR)
    Qty = np.matmul(Q.conj().T, y)
    QtX = np.matmul(Q.conj().T, X)
    yy = dot(y)
    Xy = np.matmul(X.conj().T, y)
    XX = np.array([dot(col) for col in X.T])
    return (Q, Qty, QtX, yy, Xy, XX)


def all_players_compute(players, invR):
    # The * operator unpacks the container to positional arguments.
    Qs, Qtys, QtXs, yys, Xys, XXs = zip(
        *[player_compute(player, invR) for player in players]
    )
    return (Qs, Qtys, QtXs, yys, Xys, XXs)


# Do share creation in one go to choose smallest possible prime.
# For less communication in MPC we should choose an appropriate prime, which could have a hardcoded default value.
def int_list_to_shares(ints):
    prime = get_large_enough_prime([sum(ints)])
    plyrs = len(ints)
    # We initialize the secret sharing with threshold = players - 1
    shares = [secret_int_to_points(x, plyrs - 1, plyrs, prime) for x in ints]
    return (shares, prime)


def list_vector_to_shares_vector(v):
    sv = [int_list_to_shares(ints) for ints in zip(*v)]
    return sv


def list_matrix_to_shares_matrix(M):
    sM = [list_vector_to_shares_vector(v) for v in zip(*M)]
    return sM


# Aggregate the i-th point of each polynomial in the i-th vector. The vectors represent the secret shared data each player holds.
def aggregate_share_points(shares, prime):
    aggregated_point_vectors = zip(*shares)
    return (aggregated_point_vectors, prime)


# The argument of this function is a list of points sharing the same first coordinate.
def sum_point(point_list):
    sum_y = sum([point[1] for point in point_list])
    x = point_list[0][0]
    return (x, sum_y)


# pnum is the number of the player holding the container, as well as the x coordinate of the summed point.
def sum_point_vectors(apvs, prime):
    summed_point_vectors = [sum_point(apv) for pnum, apv in enumerate(apvs)]
    return (summed_point_vectors, prime)


def summed_points_to_plain_sum(spvs, prime):
    npl = len(spvs)  # The number of players
    # To reconstruct, we take the first players-1 points, but leaving out any one point is possible.
    plain_sum = points_to_secret_int(spvs[0 : (npl - 1)], prime)
    return plain_sum


def shares_to_plain_sum(shares, prime):
    plain_sum = summed_points_to_plain_sum(
        *sum_point_vectors(*aggregate_share_points(shares, prime))
    )
    return plain_sum


def shares_vector_to_plain_sum_vector(sv):
    psv = [shares_to_plain_sum(*shares) for shares in sv]
    return psv


def shares_matrix_to_plain_sum_matrix(sM):
    psm = [shares_vector_to_plain_sum_vector(sv) for sv in sM]
    return psm


# SMC
# TODO: Add network communication.
def private_compute(players_computed):
    Qs, Qtys, QtXs, yys, Xys, XXs = players_computed

    yy = toFloat(shares_to_plain_sum(*int_list_to_shares(toIntVec(yys))))

    Xy = toFloatVec(
        shares_vector_to_plain_sum_vector(list_vector_to_shares_vector(toIntMat(Xys)))
    )

    XX = toFloatVec(
        shares_vector_to_plain_sum_vector(list_vector_to_shares_vector(toIntMat(XXs)))
    )

    Qty = toFloatVec(
        shares_vector_to_plain_sum_vector(list_vector_to_shares_vector(toIntMat(Qtys)))
    )

    QtX = toFloatMat(
        shares_matrix_to_plain_sum_matrix(list_matrix_to_shares_matrix(toInt3D(QtXs)))
    )
    return (yy, Xy, XX, Qty, QtX)


def public_compute(private_computed):
    yy, Xy, XX, Qty, QtX = private_computed

    QtyQty = dot(Qty)
    QtXQty = np.matmul(QtX.conj().T, Qty)
    QtXQtX = np.array([dot(col) for col in QtX.T])

    yyq = yy - QtyQty
    Xyq = Xy - QtXQty
    XXq = XX - QtXQtX
    return (QtyQty, QtXQty, QtXQtX, yyq, Xyq, XXq)


# Input:   [int], int, int, int, int
def compute(inNs, num_of_pl, num_of_elem, M, K):
    Ns = gen_N_vec(inNs, num_of_elem, num_of_pl)

    players = generate_player_vec(Ns, M, K)

    # Public or tree or SMC
    # TODO: implement invR as SMC
    invR = np.linalg.inv(
        np.linalg.qr(np.concatenate([player[3] for player in players], axis=0))[1]
    )

    players_computed = all_players_compute(players, invR)

    private_computed = private_compute(players_computed)

    public_computed = public_compute(private_computed)

    QtyQty, QtXQty, QtXQtX, yyq, Xyq, XXq = public_computed

    return (QtyQty, QtXQty, QtXQtX, yyq, Xyq, XXq, players, Ns, K)


def compute_coeffs(computed):
    QtyQty, QtXQty, QtXQtX, yyq, Xyq, XXq, players, Ns, K = computed

    D = compute_D(Ns, K)
    # Public

    beta = np.divide(Xyq, XXq)

    pow2 = np.vectorize(lambda x: np.power(x, 2))
    sigma = np.sqrt(np.divide(np.divide(yyq, (XXq - pow2(beta))), D))

    tstat = np.divide(beta, sigma)
    pval = 2 * sc.stats.t(df=D).cdf(-np.abs(tstat))
    df = pd.DataFrame({"beta": beta, "sigma": sigma, "tstat": tstat, "pval": pval})

    return df


# Compare to primary analysis for first M0 columns of $X$
def test_local(players):
    M0 = 5

    ys, Xs, Cs, Rs = zip(*players)

    y = np.concatenate(ys)
    X = np.concatenate(Xs, axis=0)
    C = np.concatenate(Cs, axis=0)

    # This is the 0th iteration. We reshape so broadcasting with C works.
    fit0 = sm.OLS(y, np.column_stack((X[:, 0].reshape(-1, 1), C))).fit()
    res = np.array(
        (fit0.params[0], fit0.bse[0], fit0.tvalues[0], fit0.pvalues[0])
    ).reshape(1, -1)

    for m in np.arange(1, M0):
        fit = sm.OLS(y, np.column_stack((X[:, m].reshape(-1, 1), C))).fit()
        # from R: fit = lm(y ~ X[,m] + C - 1)
        # X[:,m] = choose mth column
        nres = np.array(
            (fit.params[0], fit.bse[0], fit.tvalues[0], fit.pvalues[0])
        ).reshape(1, -1)
        res = np.concatenate((res, nres))

        df = pd.DataFrame(
            {
                "beta": res[:, 0],
                "sigma": res[:, 1],
                "tstat": res[:, 2],
                "pval": res[:, 3],
            }
        )

    return df

# Test with example data from dash.r
def reftest():
    computed = compute([1000, 2000, 1500], 3, 1000, 10000, 3)
    df1 = compute_coeffs(computed)
    df2 = test_local(computed[6])
    print("Secret shared computation:")
    print(df1.head(5))
    print("Plaintext computation:")
    print(df2)

# More generic test
def test(number_of_players):
    computed = compute([], number_of_players, 1000, 10000, 3)
    df1 = compute_coeffs(computed)
    df2 = test_local(computed[6])
    print("Secret shared computation:")
    print(df1.head(5))
    print("Plaintext computation:")
    print(df2)

# Uncomment the next line to be able to run the reference test with 'python asmpc.py'
# reftest()
