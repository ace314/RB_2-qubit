import multiprocessing as mp

def job(x):
    return x*x

def multicore():
    a = [3]*10
    pool = mp.Pool()
    res = pool.map(job, a)
    print(res)


if __name__ == '__main__':
    multicore()