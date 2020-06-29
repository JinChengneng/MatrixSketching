import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


def gaussian_proj(s_int, a_mat=None, a_mat_list=None):
    if a_mat is not None and a_mat_list is None:
        m_int, n_int = a_mat.shape
        s_mat = np.random.randn(n_int, s_int) / np.sqrt(s_int)
        sketch_a_mat = np.dot(a_mat, s_mat)
        return sketch_a_mat
    elif a_mat_list is not None and a_mat is None:
        m_int, n_int = a_mat_list[0].shape
        s_mat = np.random.randn(n_int, s_int) / np.sqrt(s_int)
        sketch_a_mat_list = []
        for a_mat in a_mat_list:
            sketch_a_mat = np.dot(a_mat, s_mat)
            sketch_a_mat_list.append(sketch_a_mat)
        return sketch_a_mat_list


def realfft_row(a_mat):
    n_int = a_mat.shape[1]
    fft_mat = np.fft.fft(a_mat, n=None, axis=1) / np.sqrt(n_int)
    if n_int % 2 == 1:
        cutoff_int = int((n_int+1) / 2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int, n_int))
    else:
        cutoff_int = int(n_int/2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int+1, n_int))
    c_mat = fft_mat.real
    c_mat[:, idx_real_vec] *= np.sqrt(2)
    c_mat[:, idx_imag_vec] = fft_mat[:, idx_imag_vec].imag * np.sqrt(2)
    return c_mat


def srft(s_int, a_mat=None, a_mat_list=None):
    if a_mat is not None and a_mat_list is None:
        n_int = a_mat.shape[1]
        sign_vec = np.random.choice(2, n_int) * 2 - 1
        idx_vec = np.random.choice(n_int, s_int, replace=False)
        a_mat = a_mat * sign_vec.reshape(1, n_int)
        a_mat = realfft_row(a_mat)
        c_mat = a_mat[:, idx_vec] * np.sqrt(n_int / s_int)
        return c_mat
    elif a_mat_list is not None and a_mat is None:
        n_int = a_mat_list[0].shape[1]
        sign_vec = np.random.choice(2, n_int) * 2 - 1
        idx_vec = np.random.choice(n_int, s_int, replace=False)
        sketch_a_mat_list = []
        for a_mat in a_mat_list:
            a_mat = a_mat * sign_vec.reshape(1, n_int)
            a_mat = realfft_row(a_mat)
            c_mat = a_mat[:, idx_vec] * np.sqrt(n_int / s_int)
            sketch_a_mat_list.append(c_mat)
        return sketch_a_mat_list


def countsketch(s_int, a_mat=None, a_mat_list=None):
    if a_mat is not None and a_mat_list is None:
        m_int, n_int = a_mat.shape
        hash_vec = np.random.choice(s_int, n_int, replace=True)
        sign_vec = np.random.choice(2, n_int, replace=True) * 2 - 1
        sketch_a_mat = np.zeros((m_int, s_int))
        for j in range(n_int):
            h = hash_vec[j]
            g = sign_vec[j]
            sketch_a_mat[:, h] += g * a_mat[:, j]
        return sketch_a_mat
    elif a_mat_list is not None and a_mat is None:
        m_int, n_int = a_mat_list[0].shape
        hash_vec = np.random.choice(s_int, n_int, replace=True)
        sign_vec = np.random.choice(2, n_int, replace=True) * 2 - 1
        sketch_a_mat_list = []
        for a_mat in a_mat_list:
            sketch_a_mat = np.zeros((a_mat.shape[0], s_int))
            for j in range(n_int):
                h = hash_vec[j]
                g = sign_vec[j]
                sketch_a_mat[:, h] += g * a_mat[:, j]
            sketch_a_mat_list.append(sketch_a_mat)
        return sketch_a_mat_list


def leverage(s_int, a_mat=None, a_mat_list=None):
    if a_mat is not None and a_mat_list is None:
        # calculate leverage vector
        n_int = a_mat.shape[1]
        _, _, v_mat = np.linalg.svd(a_mat, full_matrices=False)
        lev_vec = np.sum(v_mat ** 2, axis=0)
        # generate sketch matrix by leverage
        a_mat = a_mat.T
        prob_vec = lev_vec / sum(lev_vec)
        idx_vec = np.random.choice(n_int, s_int, replace=False, p=prob_vec)
        scaling_vec = np.sqrt(s_int * prob_vec[idx_vec]) + 1e-10
        sx_mat = a_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        return sx_mat.T
    if a_mat_list is not None and a_mat is None:
        # calculate leverage vector
        a_mat = a_mat_list[0]
        n_int = a_mat.shape[1]
        _, _, v_mat = np.linalg.svd(a_mat, full_matrices=False)
        lev_vec = np.sum(v_mat ** 2, axis=0)
        # generate sketch matrix by leverage
        a_mat = a_mat_list[0].T
        b_mat = a_mat_list[1].T
        prob_vec = lev_vec / sum(lev_vec)
        idx_vec = np.random.choice(n_int, s_int, replace=False, p=prob_vec)
        scaling_vec = np.sqrt(s_int * prob_vec[idx_vec]) + 1e-10
        sx_mat = a_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        sy_mat = b_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        return [sx_mat.T, sy_mat.T]


def leverage_approx(s_int, a_mat=None, a_mat_list=None):
    if a_mat is not None and a_mat_list is None:
        # calculate leverage vector
        n_int = a_mat.shape[1]
        b_mat = gaussian_proj(s_int, a_mat=a_mat)
        u_mat, sig_vec, _ = np.linalg.svd(b_mat, full_matrices=False)
        t_mat = u_mat.T / sig_vec.reshape(len(sig_vec), 1)
        b_mat = np.dot(t_mat, a_mat)
        lev_vec = np.sum(b_mat ** 2, axis=0)
        # generate sketch matrix by leverage
        a_mat = a_mat.T
        prob_vec = lev_vec / sum(lev_vec)
        idx_vec = np.random.choice(n_int, s_int, replace=False, p=prob_vec)
        scaling_vec = np.sqrt(s_int * prob_vec[idx_vec]) + 1e-10
        sx_mat = a_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        return sx_mat.T
    if a_mat_list is not None and a_mat is None:
        # calculate leverage vector
        a_mat = a_mat_list[0]
        m_int, n_int = a_mat.shape
        b_mat = gaussian_proj(s_int, a_mat=a_mat)
        u_mat, sig_vec, _ = np.linalg.svd(b_mat, full_matrices=False)
        t_mat = u_mat.T / sig_vec.reshape(len(sig_vec), 1)
        b_mat = np.dot(t_mat, a_mat)
        lev_vec = np.sum(b_mat ** 2, axis=0)
        # generate sketch matrix by leverage
        a_mat = a_mat_list[0].T
        b_mat = a_mat_list[1].T
        prob_vec = lev_vec / sum(lev_vec)
        idx_vec = np.random.choice(n_int, s_int, replace=False, p=prob_vec)
        scaling_vec = np.sqrt(s_int * prob_vec[idx_vec]) + 1e-10
        sx_mat = a_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        sy_mat = b_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        return [sx_mat.T, sy_mat.T]


def evaluate_multiply(s_int, fun_proj):
    # evaluate the error and computation time of x*x.T
    repeat = 10
    err_xx = 0
    time_xx = 0
    for i in range(repeat):
        time_start = time.time()
        c_mat = fun_proj(s_int, a_mat=x_mat)
        time_end = time.time()
        err_xx += np.linalg.norm(xx_mat - np.dot(c_mat,
                                                 c_mat.T), ord='fro') / xx_norm
        time_xx += time_end - time_start
    err_xx /= repeat
    time_xx /= repeat
    print('Approximation error xx for s=' + str(s_int) + ':    ' + str(err_xx))
    print('Computation time xx for s=' + str(s_int) + ':    ' + str(time_xx))
    # evaluate the error and computation time of x*y.T
    err_xy = 0
    time_xy = 0
    for i in range(repeat):
        time_start = time.time()
        c_mat_list = fun_proj(s_int, a_mat_list=[x_mat, y_mat])
        time_end = time.time()
        c_mat = c_mat_list[0]
        d_mat = c_mat_list[1]
        err_xy += np.linalg.norm(xy_mat - np.dot(c_mat,
                                                 d_mat.T), ord='fro') / xy_norm
        time_xy += time_end - time_start
    err_xy /= repeat
    time_xy /= repeat
    print('Approximation error xy for s=' + str(s_int) + ':    ' + str(err_xy))
    print('Computation time xy for s=' + str(s_int) + ':    ' + str(time_xy))
    return err_xx, err_xy, time_xx, time_xy


def evaluate_multiply_with_different_sketch_size(fun_proj, fun_label):
    # evaluate the error of multiply with different sketch size
    err_multiply = [[] for _ in range(2)]
    time_multiply = [[] for _ in range(2)]
    sketch_size_list = [8, 16, 32, 64, 128, 256, 516, 1024]
    for i in sketch_size_list:
        err_xx, err_xy, time_xx, time_xy = evaluate_multiply(i, fun_proj)
        err_multiply[0].append(err_xx)
        err_multiply[1].append(err_xy)
        time_multiply[0].append(time_xx)
        time_multiply[1].append(time_xy)
    plt.plot(sketch_size_list, np.log(err_multiply[0]), label = 'x*x.T')
    plt.plot(sketch_size_list, np.log(err_multiply[1]), label = 'x*y.T')
    plt.xlabel('sketch size of '+fun_label)
    plt.ylabel('log(error)')
    plt.legend()
    plt.show()
    return err_multiply, time_multiply


def evaluate_lsr(s_int, fun_proj):
    epsoids_num = 2
    min_loss = np.Inf
    err_arr = []
    for i in range(epsoids_num):
        [x_sketch, y_sketch] = fun_proj(s_int, a_mat_list=[x_mat, y_mat])
        weight_sketch = np.linalg.lstsq(x_sketch.T, y_sketch.T, rcond=None)[0]
        loss_sketch = np.linalg.norm(np.dot(x_mat.T, weight_sketch) - y_mat.T)
        if loss_sketch < min_loss:
            min_loss = loss_sketch
        err_arr.append(loss_sketch)
        print(s_int, i, loss_sketch, min_loss)
    return min_loss, np.mean(err_arr)


def evaluate_lsr_with_different_sketch_size(fun_proj, fun_label):
    # evaluate the error of least squares regression with different sketch size
    print("evaluate the error of least square with ", fun_label)
    weight_lsr = np.linalg.lstsq(x_mat.T, y_mat.T, rcond=None)[0]
    loss_lsr = np.linalg.norm(np.dot(x_mat.T, weight_lsr) - y_mat.T)
    print("loss of origin problem is ", loss_lsr)
    sketch_size_list = [8, 16, 32, 64, 128, 256, 516, 1024, 2048]
    min_err_lsr = []
    avg_err_lsr = []
    time_lsr = []
    for i in sketch_size_list:
        time_start = time.time()
        results = evaluate_lsr(i, fun_proj)
        time_end = time.time()
        min_err_lsr.append(results[0])
        avg_err_lsr.append(results[1])
        time_lsr.append(time_end - time_start)
    print('The min errors are : ', min_err_lsr)
    print('The average errors are : ', avg_err_lsr)
    print('The computation times are : ', time_lsr)
    plt.plot(sketch_size_list, err_lsr, label = fun_label)
    plt.plot(sketch_size_list, [loss_lsr]*8 , label = 'Optimal')
    plt.xlabel('sketch size')
    plt.ylabel('Error of least square regression')
    plt.legend()
    plt.show()


# import data
print("importing data")
rawdata_mat = np.load('data/YearPredictionMSD.npy', mmap_mode='r')
# rawdata_mat = rawdata_mat[0:100000, :]
print("The shape of raw data matrix is ", rawdata_mat.shape)
# separate x and y from the raw data
x_mat = rawdata_mat[:, 1:].T
m_int, n_int = x_mat.shape  # n_int >> m_int
y_mat = rawdata_mat[:, 0].reshape((1, n_int))
# calculate the true value of x*x.T and x*y.T
xx_mat = np.dot(x_mat, x_mat.T)
xx_norm = np.linalg.norm(xx_mat, ord='fro')
xy_mat = np.dot(x_mat, y_mat.T)
xy_norm = np.linalg.norm(xy_mat, ord='fro')

# use gaussian projection to get a sketch of x
sketch_gaussian = gaussian_proj(100, a_mat=x_mat)
print(sketch_gaussian.shape)
sketch_gaussian_list = gaussian_proj(100, a_mat_list=[x_mat, y_mat])
print(sketch_gaussian_list[0].shape, sketch_gaussian_list[1].shape)

# use SRFT to get a sketch of x
sketch_srft = srft(100, x_mat)
print(sketch_srft.shape)
sketch_srft_list = srft(100, a_mat_list=[x_mat, y_mat])
print(sketch_srft_list[0].shape, sketch_srft_list[1].shape)

# use count sketch to get a sketch of x
sketch_countsketch = countsketch(100, x_mat)
print(sketch_countsketch.shape)
sketch_countsketch_list = countsketch(100, a_mat_list=[x_mat, y_mat])
print(sketch_countsketch_list[0].shape, sketch_countsketch_list[1].shape)

# use leverage score sampling to get a sketch of x
sketch_leverage = leverage(100, x_mat)
print(sketch_countsketch.shape)
sketch_leverage_list = leverage( 100, a_mat_list=[x_mat, y_mat])
print(sketch_leverage_list[0].shape, sketch_leverage_list[1].shape)

# use approximated leverage score sampling to get a sketch of x
sketch_leverage_approx = leverage_approx(100, x_mat)
print(sketch_countsketch.shape)
sketch_leverage_approx_list = leverage_approx( 100, a_mat_list=[x_mat, y_mat])
print(sketch_leverage_list[0].shape, sketch_leverage_list[1].shape)

# evaluate multiply with different sketch size
err_multiply, time_multiply = evaluate_multiply_with_different_sketch_size(leverage_approx, 'leverage_approx')
name = ['err_xx', 'err_xy', 'time_xx', 'time_xy']
results = pd.DataFrame(columns=name, data=np.array(err_multiply+time_multiply).T)
results.to_csv('results_leverage_approx.csv')

# evaluate least squares regression with different sketch size
evaluate_lsr_with_different_sketch_size(gaussian_proj, 'gaussian')

