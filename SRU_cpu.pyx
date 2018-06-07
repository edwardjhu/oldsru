# author: Felix Wu
# cython: infer_types=True
import numpy as np
cimport cython
from libcpp cimport bool

ctypedef fused my_type:
    int
    double
    float
    long

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sru_compute_cpu_cy(my_type [:, :, :] x_prime,
                         my_type [:, :, :] x_tilde,
                         my_type [:, :, :] forget,
                         my_type [:, :, :] reset,
                         my_type [:, :] c_init,
                         int di,
                         int activation_type,
                         int mask_h):
    # inputs: bidir, length, c_init, mask_h, x_tilde, forget, x_prime, reset, h
    # outputs: c_final, h
    length = x_prime.shape[0]
    batch_size = x_prime.shape[1]
    d = x_prime.shape[2]

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is float:
        dtype = np.float32
    else:
        dtype = np.long
    h =  np.zeros([length, batch_size, d], dtype=dtype)
    c_t =  np.zeros([batch_size, d], dtype=dtype)

    c_t[:, :] = c_init[:, :]

    if di == 0:
        for t in range(length):
            c_t[:, :] = (c_t[:, :] - x_tilde[t, :, :]) * forget[t, :, :] + x_tilde[t, :, :]

            if activation_type == 0:
                g_c_t = c_t
            elif activation_type == 1:
                g_c_t = np.tanh(c_t)
            # elif activation_type == 2:
                # g_c_t = nn.functional.relu(c_t)
            else:
                assert False, 'Activation type must be 0, 1, or 2, not {}'.format(activation_type)

            h[t, :, :] = (g_c_t * mask_h - x_prime[t, :, :]) * reset[t, :, :] + x_prime[t, :, :]
    else:
        for t in range(length - 1, -1, -1):
            c_t[:, :] = (c_t[:, :] - x_tilde[t, :, :]) * forget[t, :, :] + x_tilde[t, :, :]

            if activation_type == 0:
                g_c_t = c_t
            elif activation_type == 1:
                g_c_t = np.tanh(c_t)
            # elif activation_type == 2:
                # g_c_t = nn.functional.relu(c_t)
            else:
                assert False, 'Activation type must be 0, 1, or 2, not {}'.format(activation_type)

            h[t, :, :] = (g_c_t * mask_h - x_prime[t, :, :]) * reset[t, :, :] + x_prime[t, :, :]

    return h, c_t 


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sru_compute_bi_cpu_cy(my_type [:, :, :, :] x_prime,
                            my_type [:, :, :, :] x_tilde,
                            my_type [:, :, :, :] forget,
                            my_type [:, :, :, :] reset,
                            my_type [:, :] c_init,
                            int bidir,
                            int activation_type,
                            int mask_h):
    # inputs: bidir, length, c_init, mask_h, x_tilde, forget, x_prime, reset, h
    # outputs: c_final, h
    length = x_prime.shape[0]
    batch_size = x_prime.shape[1]
    d = x_prime.shape[3]

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is float:
        dtype = np.float32
    else:
        dtype = np.long
    h =  np.zeros([length, batch_size, bidir*d], dtype=dtype)
    c_t =  np.zeros([batch_size, bidir*d], dtype=dtype)

    c_t[:, :] = c_init[:, :]
    for t in range(length):
        c_t[:, :d] = (c_t[:, :d] - x_tilde[t, :, 0, :]) * forget[t, :, 0, :] + x_tilde[t, :, 0, :]

        if activation_type == 0:
            g_c_t = c_t[:, :d]
        elif activation_type == 1:
            g_c_t = np.tanh(c_t[:, :d])
        # elif activation_type == 2:
            # g_c_t = nn.functional.relu(c_t)
        else:
            assert False, 'Activation type must be 0, 1, or 2, not {}'.format(activation_type)

        h[t, :, :d] = (g_c_t * mask_h - x_prime[t, :, 0, :]) * reset[t, :, 0, :] + x_prime[t, :, 0, :]

    for t in range(length - 1, -1, -1):
        c_t[:, d:] = (c_t[:, d:] - x_tilde[t, :, 1, :]) * forget[t, :, 1, :] + x_tilde[t, :, 1, :]

        if activation_type == 0:
            g_c_t = c_t[:, d:]
        elif activation_type == 1:
            g_c_t = np.tanh(c_t[:, d:])
        # elif activation_type == 2:
            # g_c_t = nn.functional.relu(c_t)
        else:
            assert False, 'Activation type must be 0, 1, or 2, not {}'.format(activation_type)

        h[t, :, d:] = (g_c_t * mask_h - x_prime[t, :, 1, :]) * reset[t, :, 1, :] + x_prime[t, :, 1, :]

    return h, c_t 
