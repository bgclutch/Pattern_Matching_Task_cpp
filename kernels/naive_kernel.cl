__kernel void naive_pattern_kernel(__global const uchar* data, uint size, __global const uchar* patterns,
                              uint patternsAmount, __global const uint* lengths,
                              __global const uint* offsets, __global uint* matches) {

    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    if (global_id >= size)
        return;

    for (int i = 0; i < patternsAmount; ++i) {
        int cur_len = lengths[i];
        int cur_offset = offsets[i];

        if (global_id + cur_len > size)
            continue;

        bool match = true;
        for (int j = 0; j < cur_len; ++j) {
            if (data[global_id + j] != patterns[cur_offset + j]) {
                match = false;
                break;
            }
        }

        if (match) {
            atomic_inc(&matches[i]);
        }
    }
}