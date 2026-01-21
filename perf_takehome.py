"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Highly optimized VLIW SIMD kernel with software pipelining.
        Target: <1400 cycles (currently ~3968 = 37x speedup, need ~106x)
        """
        n_vectors = batch_size // VLEN  # 32
        n_groups = 6  # Limited by 6 valu slots

        # Address temporaries (12 for parallel ALU)
        tmp_addr = [self.alloc_scratch(f"ta{i}") for i in range(12)]

        # Load parameters
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_addr[0], i))
            self.add("load", ("load", self.scratch[v], tmp_addr[0]))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))

        # Vector constants
        v_zero = self.alloc_scratch("vz", VLEN)
        v_one = self.alloc_scratch("vo", VLEN)
        v_two = self.alloc_scratch("vt", VLEN)
        v_n = self.alloc_scratch("vn", VLEN)

        self.instrs.append({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_n, self.scratch["n_nodes"]),
        ]})

        # Hash constants
        vh = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1 = self.alloc_scratch(f"hc{hi}a", VLEN)
            vc3 = self.alloc_scratch(f"hc{hi}b", VLEN)
            vh.append((op1, vc1, op2, op3, vc3))
        for i in range(0, 6, 3):
            vl = []
            for j in range(min(3, 6-i)):
                vl.append(("vbroadcast", vh[i+j][1], self.scratch_const(HASH_STAGES[i+j][1])))
                vl.append(("vbroadcast", vh[i+j][4], self.scratch_const(HASH_STAGES[i+j][4])))
            self.instrs.append({"valu": vl})

        # Double-buffered scratch
        vi = [self.alloc_scratch(f"vi{b}", n_groups*VLEN) for b in range(2)]
        vv = [self.alloc_scratch(f"vv{b}", n_groups*VLEN) for b in range(2)]
        vn = [self.alloc_scratch(f"vn{b}", n_groups*VLEN) for b in range(2)]
        t1 = [self.alloc_scratch(f"t1{b}", n_groups*VLEN) for b in range(2)]
        t2 = [self.alloc_scratch(f"t2{b}", n_groups*VLEN) for b in range(2)]
        ai = [[self.alloc_scratch(f"ai{b}{g}") for g in range(n_groups)] for b in range(2)]
        av = [[self.alloc_scratch(f"av{b}{g}") for g in range(n_groups)] for b in range(2)]

        def ld(buf, base, n):
            alu = []
            for g in range(n):
                off = self.scratch_const((base+g)*VLEN)
                alu.append(("+", ai[buf][g], self.scratch["inp_indices_p"], off))
                alu.append(("+", av[buf][g], self.scratch["inp_values_p"], off))
            self.instrs.append({"alu": alu})
            for g in range(0, n, 2):
                l = [("vload", vi[buf]+g*VLEN, ai[buf][g])]
                if g+1<n: l.append(("vload", vi[buf]+(g+1)*VLEN, ai[buf][g+1]))
                self.instrs.append({"load": l})
                l = [("vload", vv[buf]+g*VLEN, av[buf][g])]
                if g+1<n: l.append(("vload", vv[buf]+(g+1)*VLEN, av[buf][g+1]))
                self.instrs.append({"load": l})

        def gth(buf, n):
            tot = n * VLEN
            alu = [("+", tmp_addr[i], self.scratch["forest_values_p"], vi[buf]+i) for i in range(min(12,tot))]
            self.instrs.append({"alu": alu})
            for i in range(0, tot, 2):
                ins = {"load": [("load", vn[buf]+i, tmp_addr[i%12])]}
                if i+1<tot: ins["load"].append(("load", vn[buf]+i+1, tmp_addr[(i+1)%12]))
                nxt = i+12
                if nxt<tot:
                    a = [("+", tmp_addr[nxt%12], self.scratch["forest_values_p"], vi[buf]+nxt)]
                    if nxt+1<tot: a.append(("+", tmp_addr[(nxt+1)%12], self.scratch["forest_values_p"], vi[buf]+nxt+1))
                    ins["alu"] = a
                self.instrs.append(ins)

        def cmp(buf, n):
            self.instrs.append({"valu": [("^", vv[buf]+g*VLEN, vv[buf]+g*VLEN, vn[buf]+g*VLEN) for g in range(n)]})
            for op1, vc1, op2, op3, vc3 in vh:
                self.instrs.append({"valu": [(op1, t1[buf]+g*VLEN, vv[buf]+g*VLEN, vc1) for g in range(n)]})
                self.instrs.append({"valu": [(op3, t2[buf]+g*VLEN, vv[buf]+g*VLEN, vc3) for g in range(n)]})
                self.instrs.append({"valu": [(op2, vv[buf]+g*VLEN, t1[buf]+g*VLEN, t2[buf]+g*VLEN) for g in range(n)]})
            self.instrs.append({"valu": [("&", t1[buf]+g*VLEN, vv[buf]+g*VLEN, v_one) for g in range(n)]})
            self.instrs.append({"valu": [("+", t1[buf]+g*VLEN, t1[buf]+g*VLEN, v_one) for g in range(n)]})
            self.instrs.append({"valu": [("multiply_add", vi[buf]+g*VLEN, vi[buf]+g*VLEN, v_two, t1[buf]+g*VLEN) for g in range(n)]})
            self.instrs.append({"valu": [("<", t1[buf]+g*VLEN, vi[buf]+g*VLEN, v_n) for g in range(n)]})
            self.instrs.append({"valu": [("*", vi[buf]+g*VLEN, vi[buf]+g*VLEN, t1[buf]+g*VLEN) for g in range(n)]})

        def st(buf, n):
            for g in range(0, n, 2):
                s = [("vstore", ai[buf][g], vi[buf]+g*VLEN)]
                if g+1<n: s.append(("vstore", ai[buf][g+1], vi[buf]+(g+1)*VLEN))
                self.instrs.append({"store": s})
                s = [("vstore", av[buf][g], vv[buf]+g*VLEN)]
                if g+1<n: s.append(("vstore", av[buf][g+1], vv[buf]+(g+1)*VLEN))
                self.instrs.append({"store": s})

        def gth_cmp(lb, ln, cb, cn):
            tot = ln * VLEN
            ops = []
            ops.append({"valu": [("^", vv[cb]+g*VLEN, vv[cb]+g*VLEN, vn[cb]+g*VLEN) for g in range(cn)]})
            for op1, vc1, op2, op3, vc3 in vh:
                ops.append({"valu": [(op1, t1[cb]+g*VLEN, vv[cb]+g*VLEN, vc1) for g in range(cn)]})
                ops.append({"valu": [(op3, t2[cb]+g*VLEN, vv[cb]+g*VLEN, vc3) for g in range(cn)]})
                ops.append({"valu": [(op2, vv[cb]+g*VLEN, t1[cb]+g*VLEN, t2[cb]+g*VLEN) for g in range(cn)]})
            ops.append({"valu": [("&", t1[cb]+g*VLEN, vv[cb]+g*VLEN, v_one) for g in range(cn)]})
            ops.append({"valu": [("+", t1[cb]+g*VLEN, t1[cb]+g*VLEN, v_one) for g in range(cn)]})
            ops.append({"valu": [("multiply_add", vi[cb]+g*VLEN, vi[cb]+g*VLEN, v_two, t1[cb]+g*VLEN) for g in range(cn)]})
            ops.append({"valu": [("<", t1[cb]+g*VLEN, vi[cb]+g*VLEN, v_n) for g in range(cn)]})
            ops.append({"valu": [("*", vi[cb]+g*VLEN, vi[cb]+g*VLEN, t1[cb]+g*VLEN) for g in range(cn)]})

            alu = [("+", tmp_addr[i], self.scratch["forest_values_p"], vi[lb]+i) for i in range(min(12,tot))]
            self.instrs.append({"alu": alu})
            oi = 0
            for i in range(0, tot, 2):
                ins = {"load": [("load", vn[lb]+i, tmp_addr[i%12])]}
                if i+1<tot: ins["load"].append(("load", vn[lb]+i+1, tmp_addr[(i+1)%12]))
                nxt = i+12
                if nxt<tot:
                    a = [("+", tmp_addr[nxt%12], self.scratch["forest_values_p"], vi[lb]+nxt)]
                    if nxt+1<tot: a.append(("+", tmp_addr[(nxt+1)%12], self.scratch["forest_values_p"], vi[lb]+nxt+1))
                    ins["alu"] = a
                if oi<len(ops):
                    ins.update(ops[oi])
                    oi += 1
                self.instrs.append(ins)
            while oi<len(ops):
                self.instrs.append(ops[oi])
                oi += 1

        # Main loop
        for _ in range(rounds):
            nb = (n_vectors+n_groups-1)//n_groups
            for bi in range(nb):
                buf = bi % 2
                base = bi * n_groups
                na = min(n_groups, n_vectors-base)
                ld(buf, base, na)
                if bi == 0:
                    gth(buf, na)
                else:
                    pb = (buf+1)%2
                    pbase = (bi-1)*n_groups
                    pn = min(n_groups, n_vectors-pbase)
                    gth_cmp(buf, na, pb, pn)
                    st(pb, pn)
            li = nb-1
            lb = li%2
            lbase = li*n_groups
            ln = min(n_groups, n_vectors-lbase)
            cmp(lb, ln)
            st(lb, ln)

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
