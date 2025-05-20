import numpy as np
import numpy.random as rd
import argparse

def generate_system(n:int):
    '''Generate system of linear equations Ax=b with n variables'''
    # A = np.zeros((n,n), dtype=np.int8) # use params from -100 to 100 for simplicity
    # b = np.zeros((n,), dtype=np.int8)
    
    rng = rd.default_rng()
    A = rng.uniform(-10, 10, size=(n,n))
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) - np.abs(A[i,i]) + rng.uniform(1, 5)
        neg = rng.binomial(1, 0.5)
        if neg:
            A[i,i] = -A[i,i]
    b = rng.uniform(-50, 50, size=(n,))
    
    try:
        x = np.linalg.solve(A,b)
    except np.linalg.LinAlgError as e:
        print("Random linear system failed")
        raise e
    return A, b, x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_size", default=5)
    parser.add_argument("--num_systems", default=1)
    opt = parser.parse_args()
    
    As = []
    bs = []
    xs = []
    num_systems = int(opt.num_systems)
    system_size = int(opt.system_size)
    count = 0
    while count < num_systems:
        try:
            A, b, x = generate_system(system_size)
            As.append(A)
            bs.append(b)
            xs.append(x)
            count += 1
        except np.linalg.LinAlgError:
            continue
        except Exception:
            continue
    
    # save systems to file
    for i in range(num_systems):
        # print(f"System no. {i+1}:")
        # print("A: ", As[i])
        # print("b: ", bs[i])
        # print("x: ", xs[i])
        with open(f"data/system_{i+1}_size{system_size}.txt", "w") as f:
            f.write(f"{system_size}")
            f.write("\n")
            for row in As[i]:
                f.write(" ".join([str(j) for j in row]))
                f.write("\n")
            f.write(" ".join([str(j) for j in bs[i]]))
            f.write("\n")
            f.write(" ".join([str(j) for j in xs[i]]))
        # print()
    

if __name__ == '__main__':
    main()