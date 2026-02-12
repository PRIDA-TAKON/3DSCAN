import numpy as np
import time

class MockPlyElement:
    def __init__(self, N):
        self.N = N
        # Create a structured array mimicking PlyElement.data
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('opacity', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
        ]
        self.data = np.zeros(N, dtype=dtype_list)

        # Fill with random data
        # Note: assigning to fields works on structured array
        for name, _ in dtype_list:
            if name.startswith('scale'):
                self.data[name] = np.random.uniform(-5, 5, N).astype(np.float32)
            elif name == 'opacity':
                self.data[name] = np.random.uniform(-10, 10, N).astype(np.float32)
            else:
                self.data[name] = np.random.randn(N).astype(np.float32)

    def __getitem__(self, key):
        return self.data[key]

def baseline_sort(vert):
    scales_keys = [k for k in vert.data.dtype.names if k.startswith("scale_")]
    opacity_key = "opacity" if "opacity" in vert.data.dtype.names else None

    start = time.time()

    scale_sum = np.sum([vert[k] for k in scales_keys], axis=0)

    sorted_indices = np.argsort(
       -np.exp(scale_sum) / (1 / (1 + np.exp(-vert[opacity_key])))
    )

    scales = np.stack([vert[k][sorted_indices] for k in scales_keys], axis=1).astype(np.float32)

    end = time.time()
    return end - start, sorted_indices, scales

def optimized_v3(vert):
    scales_keys = [k for k in vert.data.dtype.names if k.startswith("scale_")]
    opacity_key = "opacity" if "opacity" in vert.data.dtype.names else None

    start = time.time()

    # Reuse array, keep original math
    scales_T = np.array([vert[k] for k in scales_keys])
    scale_sum = np.sum(scales_T, axis=0)

    opacity = vert[opacity_key]
    # Original formula
    metric = -np.exp(scale_sum) / (1 / (1 + np.exp(-opacity)))

    sorted_indices = np.argsort(metric)

    scales = scales_T[:, sorted_indices].T.astype(np.float32)

    end = time.time()
    return end - start, sorted_indices, scales

def main():
    N = 1_000_000
    print(f"Generating {N} vertices (Structured Array)...")
    vert = MockPlyElement(N)

    print("Running Baseline...")
    base_time_total = 0
    iterations = 5
    for _ in range(iterations):
        t, base_idx, base_scales = baseline_sort(vert)
        base_time_total += t
    avg_base = base_time_total / iterations
    print(f"Baseline Avg Time: {avg_base:.4f}s")

    print("Running Optimized v3 (Reuse array, Original Math)...")
    opt3_time_total = 0
    for _ in range(iterations):
        t, opt3_idx, opt3_scales = optimized_v3(vert)
        opt3_time_total += t
    avg_opt3 = opt3_time_total / iterations
    print(f"Optimized v3 Avg Time: {avg_opt3:.4f}s")

    # Verification
    print("Verifying correctness (v3)...")
    success = True
    if np.array_equal(base_idx, opt3_idx):
        print("✅ Indices match perfectly!")
    else:
        match_count = np.sum(base_idx == opt3_idx)
        print(f"❌ Indices mismatch! Match rate: {match_count/N:.2%}")
        success = False

    if np.allclose(base_scales, opt3_scales):
        print("✅ Scales output matches!")
    else:
        print("❌ Scales output mismatch!")
        success = False

    print(f"Speedup v3: {avg_base / avg_opt3:.2f}x")

    if not success:
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
