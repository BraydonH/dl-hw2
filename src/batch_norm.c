#include "uwnet.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>

matrix mean(matrix x, int spatial)
{
    matrix m = make_matrix(1, x.cols/spatial);
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/spatial] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / spatial;
    }
    return m;
}

matrix variance(volatile matrix x, volatile matrix m, int spatial)
{
    volatile matrix v = make_matrix(1, x.cols/spatial);
    // TODO: 7.1 - calculate variance
    // matrix m is mean for each filter accross batch
    // #filters = x.cols / spatial;
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            volatile float delta = m.data[j / spatial] - x.data[i*x.cols + j];
            v.data[j / spatial] += pow(delta, 2);
        }
    }    
    
    for(int i = 0; i < m.cols; i++){
        v.data[i] = v.data[i] / (x.rows * spatial);
    }
    return v;
}

matrix normalize(volatile matrix x, volatile matrix m, volatile matrix v, int spatial)
{
    volatile matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - normalize array, norm = (x - mean) / sqrt(variance + eps)
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            volatile float mean = m.data[j / spatial];
            volatile float variance = v.data[j / spatial];
            if (variance == 0) {
                variance = 0.001; // default epsilon in pytorch BatchNorm
            } 
            volatile float normalized = (x.data[i * x.cols + j] - mean) / sqrt(variance);
            norm.data[i * x.cols + j] = normalized;
        }
    } 
    return norm;
    
}

matrix batch_normalize_forward(layer l, matrix x)
{
    float s = .1;
    volatile int spatial = x.cols / l.rolling_mean.cols;
    if (x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
    }
    volatile matrix m = mean(x, spatial);
    volatile matrix v = variance(x, m, spatial);

    volatile matrix x_norm = normalize(x, m, v, spatial);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);

    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    free_matrix(l.x[0]);
    l.x[0] = x;

    return x_norm;
}


matrix delta_mean(volatile matrix d, volatile matrix variance, int spatial)
{
    volatile matrix dm = make_matrix(1, variance.cols);
    // TODO: 7.3 - calculate dL/dmean
    for (int i = 0; i < d.rows; i++) {
        for (int j = 0; j < d.cols; j++) {
            volatile float v = variance.data[j / spatial];
            if (v == 0) {
                v = 0.001; // default epsilon in pytorch BatchNorm
            } 
            dm.data[j / spatial] += d.data[i * d.cols + j] *  (-1 / sqrt(v));
        }
    } 
    return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial)
{
    volatile matrix dv = make_matrix(1, variance.cols);
    // TODO: 7.4 - calculate dL/dvariance
    for (int i = 0; i < d.rows; i++) {
        for (int j = 0; j < d.cols; j++) {
            volatile float m = mean.data[j / spatial];
            volatile float v = variance.data[j / spatial];
            if (v == 0) {
                v = 0.001; // default epsilon in pytorch BatchNorm
            } 
            volatile float delta = x.data[i * x.cols + j] - m;
            dv.data[j / spatial] += d.data[i * d.cols + j] * delta *  (-0.5 * pow(v, -1.5));
        }
    } 
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial)
{
    int i, j;
    volatile matrix dx = make_matrix(d.rows, d.cols);
    // TODO: 7.5 - calculate dL/dx
    for (int i = 0; i < d.rows; i++) {
        for (int j = 0; j < d.cols; j++) {
            volatile int index = i * d.cols + j;
            volatile float m = mean.data[j / spatial];
            volatile float v = variance.data[j / spatial];
            if (v == 0) {
                v = 0.001; // default epsilon in pytorch BatchNorm
            } 
            volatile float delta = x.data[i * x.cols + j] - m;
            dx.data[index] = d.data[index] / sqrt(v) 
                            + dv.data[j / spatial] * 2 * delta / (x.rows * spatial)
                            + dm.data[j / spatial] / (x.rows * spatial);
        }
    }
    return dx;
}

matrix batch_normalize_backward(layer l, matrix d)
{
    int spatial = d.cols / l.rolling_mean.cols;
    matrix x = l.x[0];

    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix dm = delta_mean(d, v, spatial);
    matrix dv = delta_variance(d, x, m, v, spatial);
    matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}
