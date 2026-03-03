__kernel void fast_pattern_kernel(__global const char* text, uint text_size, __constant char* patterns, uint patterns_count,
                                  __constant int* lengths, __constant int* offsets, __global uint* results,
                                  __local char* local_text, uint max_pattern_len) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    if (global_id < text_size) {
        local_text[local_id] = text[global_id];
    }
    else {
        local_text[local_id] = 0;
    }
    int tail_size = max_pattern_len - 1;

    if (local_id < tail_size) {
        int halo_global_idx = global_id + local_size;

        if (halo_global_idx < text_size) {
            local_text[local_size + local_id] = text[halo_global_idx];
        } else {
            local_text[local_size + local_id] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id >= text_size) return;

    for (uint i = 0; i < patterns_count; ++i) {
        int pattern_len = lengths[i];
        int pattern_offset = offsets[i];

        if (global_id + pattern_len > text_size) continue;

        bool match = true;

        for (int j = 0; j < pattern_len; ++j) {
            if (local_text[local_id + j] != patterns[pattern_offset + j]) {
                match = false;
                break;
            }
        }

        if (match) {
            atomic_inc(&results[i]);
        }
    }
}