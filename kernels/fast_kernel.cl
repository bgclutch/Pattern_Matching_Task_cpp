__kernel void fast_pattern_kernel(__global const char* text, uint text_size, __constant char* patterns, uint patterns_count,
                                  __constant int* lengths, __constant int* offsets, __global uint* results,
                                  __local char* local_text, uint max_pattern_len) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint local_size = get_local_size(0);

    if (global_id < text_size) {
        local_text[local_id] = text[global_id];
    }
    else {
        local_text[local_id] = 0;
    }
    int tail_size = max_pattern_len - 1;

    uint group_start_gid = global_id - local_id;

    for (int offset = local_id; offset < tail_size; offset += local_size) {
        uint tail_global_idx = group_start_gid + local_size + offset;

        if (tail_global_idx < text_size) {
            local_text[local_size + offset] = text[tail_global_idx];
        } else {
            local_text[local_size + offset] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id >= text_size)
        return;

    for (uint i = 0; i < patterns_count; ++i) {
        int pattern_len = lengths[i];
        int pattern_offset = offsets[i];

        if (global_id + pattern_len > text_size)
            continue;

        char first_char = patterns[pattern_offset];

        if (local_text[local_id] != first_char)
            continue;

        bool match = true;

        for (int j = 1; j < pattern_len; ++j) {
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