#include "fisherface.h"

void fisherface_accel(
    hls::stream<axis_t> &input_stream,
    hls::stream<axis_t> &output_stream,
    int mode,
    int class_id
) {
    // ============================================================
    // INTERFACE PRAGMAS
    // ============================================================
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE s_axilite port=mode bundle=control
    #pragma HLS INTERFACE s_axilite port=class_id bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // ============================================================
    // STATIC STORAGE
    // ============================================================
    static fixed_t mean_vec[VECTOR_SIZE];
    static fixed_t weight_matrix[MAX_CLASSES][VECTOR_SIZE];
    #pragma HLS ARRAY_PARTITION variable=weight_matrix dim=1 complete

    // Accumulators
    result_t sums[MAX_CLASSES];
    #pragma HLS ARRAY_PARTITION variable=sums complete

    // Init Accumulators
    for (int c = 0; c < MAX_CLASSES; c++) {
        #pragma HLS UNROLL
        sums[c] = 0;
    }

    // ============================================================
    // MODE 0: INFERENCE (Packed Input)
    // ============================================================
    if (mode == 0) {
        inference_loop: for (int i = 0; i < PACKED_SIZE; i++) {
            #pragma HLS PIPELINE II=1

            axis_t temp_axis = input_stream.read();
            ap_uint<32> raw_packed = temp_axis.data;

            // Unpack 4 pixels (8-bit each)
            for (int p = 0; p < 4; p++) {
                #pragma HLS UNROLL
                ap_uint<8> pixel_byte = raw_packed.range((p + 1) * 8 - 1, p * 8);
                int idx = (i * 4) + p;
                
                fixed_t pixel_val = (fixed_t)pixel_byte;
                fixed_t centered = pixel_val - mean_vec[idx];
                
                for (int c = 0; c < MAX_CLASSES; c++) {
                    #pragma HLS UNROLL
                    sums[c] += centered * weight_matrix[c][idx];
                }
            }
        }

        // WRITE RESULTS TO STREAM
        for (int c = 0; c < MAX_CLASSES; c++) {
            #pragma HLS PIPELINE II=1
            axis_t out_pkt;
            out_pkt.data = sums[c];
            out_pkt.keep = 0xF;
            out_pkt.strb = 0xF;  // Important: mark all bytes valid
            out_pkt.last = (c == MAX_CLASSES - 1) ? 1 : 0;
            output_stream.write(out_pkt);
        }
    }
    // ============================================================
    // MODE 1 & 2: LOADING (Unpacked Input)
    // ============================================================
    else {
        loading_loop: for (int i = 0; i < VECTOR_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            
            axis_t temp_axis = input_stream.read();
            fixed_t temp_val;
            temp_val.range(15, 0) = temp_axis.data.range(15, 0);

            if (mode == 1) {
                mean_vec[i] = temp_val;
            } 
            else if (mode == 2) {
                if (class_id >= 0 && class_id < MAX_CLASSES) {
                    weight_matrix[class_id][i] = temp_val;
                }
            }
        }
        
        // *** FIX: Add strb field to ack packet ***
        axis_t out_pkt;
        out_pkt.data = 1;      // Ack value
        out_pkt.keep = 0xF;    // All 4 bytes valid
        out_pkt.strb = 0xF;    // All 4 bytes valid (THIS WAS MISSING!)
        out_pkt.last = 1;      // End of transfer
        output_stream.write(out_pkt);
    }
}