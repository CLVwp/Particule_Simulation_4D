// Exclusive prefix scan over the grid table: 2^22 u32 entries in place.
// Three kernels: one block scan per 256-entry slice, one single-group scan
// of the slice sums, one offset pass. The slice count is TABLE_SIZE / 256.

// Slice sums. One entry per 256-entry slice.
const SLICES: u32 = 16384u;



@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;

// Shared scratch for one 256-entry inclusive scan.
var<workgroup> tmp: array<u32, 256>;
var<workgroup> run_offset: u32;
var<workgroup> chunk_total: u32;

// Inclusive Hillis-Steele scan of `tmp` across the whole workgroup.
fn scan_shared(tid: u32) {
    for (var d = 1u; d < 256u; d = d << 1u) {
        workgroupBarrier();
        let add = select(0u, tmp[tid - d], tid >= d);
        workgroupBarrier();
        tmp[tid] = tmp[tid] + add;
    }
}

// Level 1: exclusive scan inside each slice, slice total into `sums`.
@compute @workgroup_size(256)
fn scan_level1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let g = wid.x * 256u + lid.x;
    tmp[lid.x] = data[g];
    scan_shared(lid.x);
    workgroupBarrier();
    let incl = tmp[lid.x];
    let excl = select(0u, tmp[lid.x - 1u], lid.x > 0u);
    data[g] = excl;
    if (lid.x == 255u) {
        sums[wid.x] = incl;
    }
}

// Level 2: exclusive scan of the slice sums in one workgroup, 64 chunks.
@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        run_offset = 0u;
    }
    for (var c = 0u; c < SLICES / 256u; c = c + 1u) {
        let idx = c * 256u + lid.x;
        workgroupBarrier();
        tmp[lid.x] = sums[idx];
        scan_shared(lid.x);
        workgroupBarrier();
        let incl = tmp[lid.x];
        if (lid.x == 255u) {
            chunk_total = incl;
        }
        workgroupBarrier();
        let excl = select(0u, tmp[lid.x - 1u], lid.x > 0u);
        sums[idx] = run_offset + excl;
        workgroupBarrier();
        if (lid.x == 0u) {
            run_offset = run_offset + chunk_total;
        }
    }
}

// Level 3: add the slice offsets back.
@compute @workgroup_size(256)
fn scan_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] + sums[gid.x / 256u];
}
