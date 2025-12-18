

#ifndef FISHERFACE_H
#define FISHERFACE_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// ============================================================
// CONFIGURATION
// ============================================================
#define VECTOR_SIZE 10000
#define PACKED_SIZE 2500   // 10000 pixels / 4 pixels per packet
#define MAX_CLASSES 5

// ============================================================
// DATA TYPES
// ============================================================
// Internal math: 16-bit fixed point
typedef ap_fixed<16, 9> fixed_t;

// Accumulator: 32-bit
typedef ap_fixed<32, 20> result_t;

// AXI-Stream packet: 32-bit data width
// We use this for both Input (Packed Faces) and Output (Scores)
typedef ap_axiu<32, 0, 0, 0> axis_t;

// ============================================================
// FUNCTION PROTOTYPE
// ============================================================
void fisherface_accel(
    hls::stream<axis_t> &input_stream,   
    hls::stream<axis_t> &output_stream,  // <--- NEW: Output Stream
    int mode,                            
    int class_id                         
);

#endif