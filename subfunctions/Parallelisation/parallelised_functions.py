# split x0, y0 into 'Ncores' batches for parallel computing
def split3D(a, n):
    k, m = divmod(a.shape[2], n)
    return (a[:,:,i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


# split x0, y0 into 'Ncores' batches for parallel computing
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))



