import numpy as np
import dask
import dask.array as da
from dask import delayed, compute
import dask.array as da
from numpy.linalg import qr, svd


## CHOLESKY
def cholesky_tsqr(X_da : dask.array.Array):
    def gramMatMul(x): #Declaring it this way will make the name appear in the Dask dashboard
        return x.T @ x
    def MatMul(x, R_inv): # Again, as above
        return x @ R_inv
    def Inverse(x):
        return np.linalg.inv(x)
        
    # A list of delayed tasks for each partition of the dataset. Each partition computes the local Gram matrix (as a delayed task)
    chunks_delayed = [dask.delayed(gramMatMul)(chunk) for chunk in X_da.to_delayed().ravel()]
    # Now sum all the local Gram matrices to get the global Gram matrix
    Gram_global_delayed = dask.delayed(sum)(chunks_delayed)   ## !! This is not parallel, but seems to be the best choice
    # Compute R as the Cholesky decomposition on the global Gram matrix (as a delayed even if a serial operation just call .compute at the end)
    R = dask.delayed(np.linalg.cholesky)(Gram_global_delayed)
    
    #R = R.persist() # This time, persist R (compute it but don't send it to the client
    R_inv = dask.delayed(Inverse)(R)

    X_da = X_da.persist()    # Persist again X_da, since X_da.to_delayed seems to cause troubles. If already in memory, this does nothing
    Q = X_da.map_blocks(MatMul, R_inv, dtype=X_da.dtype)
    #Q = Q.persist() # Persist Q, so that it won't be sent directly to the client. To make things homogenous, Q must be persisted outside of function
    return Q, R



# INDIRECT TSQR
def direct_tsqr(A : da.Array, compute_svd : bool = False):
    '''
    Tall-and-skinny QR (or SVD) decomposition via Direct TSQR.

    Parameters
    ----------
    A : dask.array.Array
        Input matrix (m x n) with m >> n, stored as a Dask array.
    compute_svd : bool, optional, default False
        If True, computes the thin SVD decomposition instead of QR.

    Returns
    -------
    If compute_svd is False:
        Q : dask.array.Array, shape (m, n)
        R : dask.array.Array, shape (n, n)
    If compute_svd is True:
        U : dask.array.Array, shape (m, n)
        S : dask.array.Array, shape (n,)
        Vt : dask.array.Array, shape (n, n)
    '''
    n, row_chunks = A.shape[1], A.chunks[0]
    p = len(row_chunks)
    A_blocks = A.to_delayed().ravel().tolist()

    # Step 1: (map) perform QR decomposition in parallel on each block
    QR1 = [delayed(qr)(block) for block in A_blocks]
    Q1s = [qr[0] for qr in QR1]
    R1s = [qr[1] for qr in QR1]

    # Stack R1s vertically
    R1 = delayed(np.vstack)(R1s)

    # Step 2: (reduce) perform global QR decomposition
    QR2 = delayed(qr)(R1)
    Q2, R2 = QR2[0], QR2[1]
    Q2s = [Q2[i*n:(i+1)*n, :] for i in range(p)]

    if compute_svd:
        SVD = delayed(svd)(R2)
        U_R, S, Vt = SVD[0], SVD[1], SVD[2]
        # Step 3: (map) building the final U by multiplying Qs blocks and U_R
        U_blocks = [da.from_delayed(delayed(lambda Q1, Q2, U_R: Q1 @ Q2 @ U_R)(Q1, Q2, U_R),
                    shape=(row_chunks[i], n), dtype=A.dtype)
                    for i, (Q1, Q2) in enumerate(zip(Q1s, Q2s))]
        U = da.concatenate(U_blocks)
        return U, da.from_delayed(S, (n,), dtype=A.dtype), da.from_delayed(Vt, (n, n), dtype=A.dtype)
    else:
        # Step 3: (map) building the final Q by multiplying Qs blocks
        Q_blocks = [da.from_delayed(delayed(lambda Q1, Q2: Q1 @ Q2)(Q1, Q2),
                    shape=(row_chunks[i], n), dtype=A.dtype)
                    for i, (Q1, Q2) in enumerate(zip(Q1s, Q2s))]
        Q = da.concatenate(Q_blocks)
        return Q, da.from_delayed(R2, (n, n), dtype=A.dtype)