
#include "nn_math.h"

int64_t fxp_mult_by_parts_with_round(int8_t fxp_a, int8_t fxp_b) {
  int64_t ret;
  int64_t int_a = fxp_a >> FXP_VALUE;
  int64_t frac_a = fxp_a - (int_a << FXP_VALUE);
  int64_t int_b = fxp_b >> FXP_VALUE;
  int64_t frac_b = fxp_b - (int_b << FXP_VALUE);

  ret = int_a * frac_b + int_b * frac_a;
  ret += (frac_a * frac_b + ((1 << FXP_VALUE) - 1)) >> FXP_VALUE;
  ret += (int_a * int_b) << FXP_VALUE;
  return ret;
}

void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int64_t *result, const unsigned int N, const unsigned int K, const unsigned int M)
{
  unsigned int n, k, m;
  unsigned int row, col;
  int64_t accumulator;
  /* int64_t accumulator_, accumulator_2; */

  for (m = 0; m < M; m++) {
    for (n = 0; n < N; n++) {
      row = n*K;
      accumulator = 0;
      for (k = 0; k < K; k++) {
        col = k*M;
        /* accumulator += fxp_mult_by_parts_with_round(mat_l[row + k], mat_r[col + m]); */
        /*  ORIGINAL */
        /* int64_t mat_row = mat_l[row + k]; */
        /* int64_t mat_col = mat_r[col + m]; */
        /* accumulator += mat_row * mat_col; */
        accumulator += mat_l[row + k] * mat_r[col + m];
      }
      result[n*M + m] = (int64_t)accumulator;
    }
  }
}

void dequantize_per_channel(int64_t *tensor_in, const int *amax_w, const int amax_x, const unsigned int N, const unsigned int C, const unsigned int K) {
  unsigned int k, n, c;

  int64_t out_value;

  for (n = 0; n < N; n++) {
    for (c = 0; c < C; c++) {
      out_value = (int64_t)(amax_w[c] * amax_x);
      for (k = 0; k < K; k++) {
        if (out_value > (1 << FXP_VALUE)) {
          tensor_in[n*C + c*K + k]  *= ((out_value + ROUND_CONST) >> FXP_VALUE);
        } else {
          tensor_in[n*C + c*K + k] = (out_value*tensor_in[n*C + c*K + k] + ROUND_CONST) >> FXP_VALUE;
        }
      }
    }
  }
}

void relu(int64_t *tensor, const unsigned int size)
{
  unsigned int i;
  for (i = 0; i < size; i++) {
    tensor[i] = MAX(tensor[i], 0);
  }
}

void sigmoid(int64_t *tensor, const unsigned int size)
{
  unsigned int i;
  int eps = 1;
  for (i = 0; i < size; i++) {
    if (tensor[i] < -eps) {
      tensor[i] = 0;
    }
    else if (tensor[i] > eps) {
      tensor[i] = 1;
    }
    else {
      tensor[i] = tensor[i] / eps;
    }
  }
}

void quantize(const int64_t *tensor_in, int8_t *tensor_q, const int scale_factor,
    const int scale_factor_inv, const unsigned int size)
{
  unsigned int i;
  int rounded_value, tensor_int, tensor_frac;
  // separation to integer and fraction parts
  int scale_factor_int = (scale_factor + ROUND_CONST) >> FXP_VALUE;
  int scale_factor_frac = scale_factor - (scale_factor_int << FXP_VALUE);
  int overflow_threshold = INT8_MAX_VALUE*scale_factor_inv;
  // element wise operation - we iterate throughout the entire length of the flattened tensor
  /* printf("scale factor: %lld\t%lld\t%lld\t%lld\t%lld\t%lld\n", scale_factor, (scale_factor + ROUND_CONST), (scale_factor + ROUND_CONST) >> FXP_VALUE, scale_factor_int, scale_factor_frac, overflow_threshold); */
  for (i = 0; i < size; i++) {
    tensor_int = (tensor_in[i] + ROUND_CONST) >> FXP_VALUE;

    /* #ifdef DEBUG_MODE */
    /*     printf("%d\t%lld\t%d\t%d\t%d\n", i, tensor_in[i], tensor_int, scale_factor_inv, overflow_threshold); */
    /* #endif */
    if (tensor_int > overflow_threshold) {
      tensor_q[i] = (int8_t)INT8_MAX_VALUE;
    } else if (tensor_int < -overflow_threshold) {
      tensor_q[i] = -(int8_t)INT8_MAX_VALUE;
    } else {
      tensor_frac = tensor_in[i] - (tensor_int << FXP_VALUE);
      // int * fxp = result is in fxp */
      rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
      // fxp * fxp = fix-point multiplication with result is in fxp */
      rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
      // convert fxp to int and add to integer parts as final value should be a rounded integer
      rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int;

      tensor_q[i] = (int8_t)rounded_value; /* store quantized value in output tensor */
    }
  }
}


void dequantize_per_row(int64_t *mat_in, const int *scale_factor_w_inv, const int scale_factor_x_inv,
    const unsigned int  N, const unsigned int  M)
{
  unsigned int  k, n;

  int64_t out_value;


  for (n = 0; n < N; n++) {
    for (k = 0; k < M; k++) {
      out_value = scale_factor_w_inv[k] * scale_factor_x_inv;
      if (out_value > (1 << FXP_VALUE))
        mat_in[n*M + k] *= ((out_value + ROUND_CONST) >> FXP_VALUE);
      else
        mat_in[n*M + k] = (out_value*mat_in[n*M + k] + ROUND_CONST) >> FXP_VALUE;
    }
  }
}

void argmax_over_cols(const int64_t *mat_in, unsigned int *indices, const unsigned int N, const unsigned int M)
{

  // calculate max of each row
  unsigned int n, m, max_idx;
  int64_t row_max, value;
  for (n = 0; n < N; n++) {
    row_max = mat_in[n*M];
    max_idx = 0;
    for (m = 0; m < M; m++) {
      value = mat_in[n*M + m];
      if (value > row_max) {
        row_max = value;
        max_idx = m; // return column
      }
    }
    indices[n] = max_idx;
  }
}

void conv1d(const int8_t *x, const int8_t *w, int64_t *y, int N, int C_in, int C_out, int H, int H_new,
    int k_size, int stride)
{
  int n_i, c_out_j, c_in_i; /* sample and channels*/
  int n; /* kernel iterations */
  int i; /* iteration*/
  // N: batch size
  // x: N * C_in * H
  // w: N * C_in * C_out * k_size
  // y: N * C_out * H_new

  for (n_i = 0; n_i < N; n_i++) {
    int N_idx_y = n_i*C_out*H_new;
    int N_idx_x = n_i*C_in*H;

    for (c_out_j = 0; c_out_j < C_out; c_out_j++) {
      int C_out_idx_y = c_out_j*H_new;
      int C_out_idx_kernel = c_out_j*C_in*k_size;

      for (i = 0; i < H_new; i++) {
        int output_idx_x = i*stride;
        int accumulator = 0;
        for (c_in_i = 0; c_in_i < C_in; c_in_i++) {
          int C_in_idx_x = c_in_i*H;
          int C_in_idx_kernel = c_in_i*k_size;
          for (n = 0; n < k_size; n++) {
            int x_value = (int)x[N_idx_x + output_idx_x + C_in_idx_x + n];
            int w_value = (int)w[C_out_idx_kernel + C_in_idx_kernel + n];
            accumulator += x_value*w_value;
            /* printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", c_out_j, i, c_in_i, n, N_idx_x + output_idx_x + C_in_idx_x + n, C_out_idx_kernel + C_in_idx_kernel + n, N_idx_y + C_out_idx_y + i, x_value, w_value, accumulator); */
          }
        }
        y[N_idx_y + C_out_idx_y + i] = (int64_t)accumulator;
      }
    }
  }
}

void pooling1d(int64_t *x, int64_t *y, int N, int C_out, int H, int H_new,
    int k_size, int stride)
{
  int n_i, c_out_j; /* sample and channels*/
  int n; /* kernel iterations */
  int i; /* iteration*/
  for (n_i = 0; n_i < N; n_i++) {
    int N_idx_y = n_i*C_out*H_new;
    int N_idx_x = n_i*C_out*H;
    for (c_out_j = 0; c_out_j < C_out; c_out_j++) {
      int C_out_idx_y = c_out_j*H_new;
      int C_out_idx_x = c_out_j*H;
      for (i = 0; i < H_new; i++) {
        int output_idx_x = i*stride;
        // MAX POOLING
        int64_t max = x[N_idx_x + C_out_idx_x + output_idx_x];
        for (n = 0; n < k_size; n++) {
          int64_t value = x[N_idx_x + C_out_idx_x + output_idx_x + n];
          if (value > max) {
            max = value;
          }
        }
        y[N_idx_y + C_out_idx_y + i] = max;
        // AVG POOLING
        /* int64_t avg = 0; */
        /* for (n = 0; n < k_size; n++) { */
        /*   avg += x[N_idx_x + C_out_idx_x + output_idx_x + n]; */
        /* } */
        /* #<{(| avg = avg * 3; |)}># */
        /* avg = (avg * 21823 + ROUND_CONST) >> FXP_VALUE; */
        /* y[N_idx_y + C_out_idx_y + i] = avg; */
      }
    }
  }
}
