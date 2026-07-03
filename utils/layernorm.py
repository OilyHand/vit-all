import numpy as np

def to_fixed(val, wl, iwl):
    frac_bits = wl - iwl
    scale = 2 ** frac_bits
    
    min_val = - (2 ** (iwl - 1))
    max_val = (2 ** (iwl - 1)) - (1.0 / scale)
    
    val_q = np.round(val * scale) / scale
    
    val_q = np.clip(val_q, min_val, max_val)
    
    return val_q.astype(np.float32)

def to_float():
    pass

def layernorm_sim(num_rows, x_flat, weight, bias):
    x = x_flat.reshape(num_rows, 768)
    output = np.zeros_like(x)

    x_q = to_fixed(x,      16, 5)
    w_q = to_fixed(weight, 16, 5)
    b_q = to_fixed(bias,   16, 5)

    for i in range(num_rows):
        row_data = x_q[i]

        sum_val = np.sum(row_data)
        sq_sum_val = np.sum(row_data ** 2)
        
        # convert to acc_t
        sum_val = to_fixed(sum_val, 32, 13)
        sq_sum_val = to_fixed(sq_sum_val, 32, 13)
        
        # Mean = sum * (1/N)
        mean = to_fixed(sum_val * 1.0 / 768.0, 32, 13)
        
        # Var = (sq_sum * (1/N)) - (mean^2)
        term1 = to_fixed(sq_sum_val * 1.0 / 768.0, 32, 13)
        term2 = to_fixed(mean * mean, 32, 13)
        var = to_fixed(term1 - term2, 32, 13)
        
        # Rstd = 1 / sqrt(var + eps)
        rstd = 1.0 / np.sqrt(var + 1e-6)
        rstd = to_fixed(rstd, 32, 13)
        
        # (input - mean)
        diff = to_fixed(row_data - mean, 32, 13)
        
        # * rstd
        norm = to_fixed(diff * rstd, 32, 13)
        
        # * gamma
        scaled = to_fixed(norm * w_q, 32, 13)
        
        # + beta
        res = to_fixed(scaled + b_q, 32, 13)
        
        output[i] = to_fixed(res, 16, 5)

    return output.flatten()


def layernorm_impl():
    pass