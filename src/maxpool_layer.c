#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (int i = 0; i < in.rows; i++) {
        for (int c = 0; c < l.channels; c++) {
            for (int y = 0; y < outh; y++) {
                for (int x = 0; x < outw; x++) {
                    for (int fidx = 0; fidx < l.size * l.size; fidx++) {
                        int dx = fidx % l.size;
                        int dy = fidx / l.size;
                        int imX = (x * l.stride) + dx;
                        int imY = (y * l.stride) + dy;
                        if (imX < l.width && imY < l.height) {
                            int inIdx = imX + (l.width * (imY + l.height * (c + l.channels * i)));
                            int outIdx = x + (outw * (y + outh * (c + l.channels * i)));
                            if (in.data[inIdx] > out.data[outIdx]) {
                                out.data[outIdx] = in.data[inIdx];
                            }
                        }
                    }
                }
            }
        }
    }
    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    for (int i = 0; i < in.rows; i++) {
        for (int c = 0; c < l.channels; c++) {
            for (int y = 0; y < outh; y++) {
                for (int x = 0; x < outw; x++) {
                    int maxInIdx = (x * l.stride) + (l.width * ((y * l.stride) + l.height * (c + l.channels * i)));
                    int max = in.data[maxInIdx];
                    int outIdx = x + (outw * (y + outh * (c + l.channels * i)));
                    for (int fidx = 1; fidx < l.size * l.size; fidx++) {
                        int dx = fidx % l.size;
                        int dy = fidx / l.size;
                        int imX = (x * l.stride) + dx;
                        int imY = (y * l.stride) + dy;
                        if (imX < l.width && imY < l.height) {
                            int inIdx = imX + (l.width * (imY + l.height * (c + l.channels * i)));
                            if (in.data[inIdx] > max) {
                                maxInIdx = inIdx;
                                max = in.data[inIdx];
                            }
                        }
                    }
                    prev_delta.data[maxInIdx] = delta.data[outIdx];                    
                }
            }
        }
    }
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

