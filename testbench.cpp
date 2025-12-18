#include "fisherface.h"
#include <iostream>

int main() {
    hls::stream<axis_t> input_stream;
    hls::stream<axis_t> output_stream;
    
    // Create test data
    float mean[VECTOR_SIZE];
    float weights[MAX_CLASSES][VECTOR_SIZE];
    float face[VECTOR_SIZE];
    
    // Initialize with simple values
    for (int i = 0; i < VECTOR_SIZE; i++) {
        mean[i] = 100.0f;  // Mean = 100
        face[i] = 150.0f;  // Face = 150, so (face - mean) = 50
        for (int c = 0; c < MAX_CLASSES; c++) {
            weights[c][i] = (c + 1) * 0.001f;  // Different weight per class
        }
    }
    
    // =========================================================
    // MODE 1: Load Mean (10000 unpacked packets)
    // =========================================================
    std::cout << "Loading mean..." << std::endl;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        axis_t pkt;
        int scaled = (int)(mean[i] * 128.0f);  // HW_SCALE
        pkt.data = scaled;
        pkt.keep = 0xF;
        pkt.strb = 0xF;
        pkt.last = (i == VECTOR_SIZE - 1) ? 1 : 0;
        input_stream.write(pkt);
    }
    fisherface_accel(input_stream, output_stream, 1, 0);
    
    // Read ack
    axis_t ack = output_stream.read();
    std::cout << "Mean loaded. Ack: " << ack.data << std::endl;
    
    // =========================================================
    // MODE 2: Load Weights (for each class)
    // =========================================================
    for (int c = 0; c < MAX_CLASSES; c++) {
        std::cout << "Loading class " << c << " weights..." << std::endl;
        for (int i = 0; i < VECTOR_SIZE; i++) {
            axis_t pkt;
            int scaled = (int)(weights[c][i] * 128.0f * 20.0f);  // TOTAL_SCALE
            pkt.data = scaled;
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = (i == VECTOR_SIZE - 1) ? 1 : 0;
            input_stream.write(pkt);
        }
        fisherface_accel(input_stream, output_stream, 2, c);
        
        // Read ack
        ack = output_stream.read();
        std::cout << "Class " << c << " loaded. Ack: " << ack.data << std::endl;
    }
    
    // =========================================================
    // MODE 0: Inference (2500 packed packets)
    // =========================================================
    std::cout << "\nRunning inference..." << std::endl;
    
    // Pack face data (4 pixels per packet)
    for (int i = 0; i < PACKED_SIZE; i++) {
        axis_t pkt;
        ap_uint<32> packed_word = 0;
        
        for (int p = 0; p < 4; p++) {
            int idx = i * 4 + p;
            int pixel_int = (int)face[idx];  // Already 0-255
            if (pixel_int > 255) pixel_int = 255;
            if (pixel_int < 0) pixel_int = 0;
            packed_word.range((p+1)*8-1, p*8) = pixel_int;
        }
        
        pkt.data = packed_word;
        pkt.keep = 0xF;
        pkt.strb = 0xF;
        pkt.last = (i == PACKED_SIZE - 1) ? 1 : 0;
        input_stream.write(pkt);
    }
    
    fisherface_accel(input_stream, output_stream, 0, 0);
    
    // Read results
    std::cout << "\nResults:" << std::endl;
    for (int c = 0; c < MAX_CLASSES; c++) {
        axis_t res = output_stream.read();
        
        // Convert from fixed point
        int raw = res.data;
        if (raw & 0x80000000) raw -= 0x100000000;  // Sign extend
        float score = raw / (4096.0f * 20.0f);  // De-scale
        
        std::cout << "  Class " << c << " Score: " << score << std::endl;
    }
    
    // Expected: Each class should have different non-zero score
    // Because weights[c] = (c+1) * 0.001
    
    return 0;
}
// ```

// ---

// ## Expected Non-Zero Output

// With the complete testbench:
// ```
// Loading mean...
// Mean loaded. Ack: 1
// Loading class 0 weights...
// Class 0 loaded. Ack: 1
// ...
// Running inference...

// Results:
//   Class 0 Score: 500.0   (approx)
//   Class 1 Score: 1000.0  (approx)
//   Class 2 Score: 1500.0  (approx)
//   Class 3 Score: 2000.0  (approx)
//   Class 4 Score: 2500.0  (approx)