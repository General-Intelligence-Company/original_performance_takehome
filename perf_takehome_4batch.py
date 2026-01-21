"""
4-batch (2 groups of 2) kernel with true hash interleaving.

With 2 batches per group:
- op1+op3 = 4 valu slots
- op2 = 2 valu slots
- A_op2 (2) + B_op1+op3 (4) = 6 slots total!

This achieves ~1.5 cycles per hash stage instead of 2.
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
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

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

    def alloc_vconst(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        NUM_BATCHES = 4
        A = [0, 1]  # Group A batches
        B = [2, 3]  # Group B batches

        v_idx_r = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_val_r = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_node_val_r = [self.alloc_scratch(f"v_node_val_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_tmp1_r = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_tmp2_r = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(NUM_BATCHES)]
        addr_temps_r = [[self.alloc_scratch(f"addr_tmp_{i}_{j}") for j in range(VLEN)] for i in range(NUM_BATCHES)]
        idx_addr_r = [self.alloc_scratch(f"idx_addr_{i}") for i in range(NUM_BATCHES)]
        val_addr_r = [self.alloc_scratch(f"val_addr_{i}") for i in range(NUM_BATCHES)]
        v_cond_r = [self.alloc_scratch(f"v_cond_{i}", VLEN) for i in range(NUM_BATCHES)]

        v_one = self.alloc_vconst(1)

        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_consts.append((self.alloc_vconst(val1), self.alloc_vconst(val3)))

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        n_vec_iters = batch_size // VLEN
        batch_offset_consts = [self.scratch_const(vi * VLEN) for vi in range(n_vec_iters)]

        self.add("flow", ("pause",))

        is_first_outer_iter = True

        for vec_iter in range(0, n_vec_iters, NUM_BATCHES):
            offsets = [batch_offset_consts[min(vec_iter + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)]
            next_vec_iter = vec_iter + NUM_BATCHES
            is_last_outer_iter = (next_vec_iter >= n_vec_iters)
            next_offsets = [batch_offset_consts[min(next_vec_iter + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)] if not is_last_outer_iter else None

            # Load initial idx/val
            if is_first_outer_iter:
                self.instrs.append({
                    "alu": [("+", idx_addr_r[i], self.scratch["inp_indices_p"], offsets[i]) for i in range(NUM_BATCHES)] +
                           [("+", val_addr_r[i], self.scratch["inp_values_p"], offsets[i]) for i in range(NUM_BATCHES)]
                })

            for i in range(NUM_BATCHES):
                self.instrs.append({"load": [("vload", v_idx_r[i], idx_addr_r[i]), ("vload", v_val_r[i], val_addr_r[i])]})

            # Compute gather addresses for group A (2 batches * 8 = 16 addresses = 2 ALU cycles)
            for start in range(0, VLEN * 2, 12):
                ops = []
                for idx in range(start, min(start + 12, VLEN * 2)):
                    b = idx // VLEN
                    e = idx % VLEN
                    ops.append(("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e))
                self.instrs.append({"alu": ops})

            for round_idx in range(rounds):
                is_last_round = (round_idx == rounds - 1)
                is_first_round = (round_idx == 0)

                # Load A node values (2 batches * 4 loads = 8 loads = 4 cycles)
                # During this, compute B addresses (16 addresses = 2 cycles worth)
                b_addr_ops = [(b, e) for b in B for e in range(VLEN)]
                addr_idx = 0

                for b in A:
                    for e in range(0, VLEN, 2):
                        instr = {"load": [("load", v_node_val_r[b] + e, addr_temps_r[b][e]),
                                          ("load", v_node_val_r[b] + e + 1, addr_temps_r[b][e + 1])]}
                        if addr_idx < len(b_addr_ops):
                            alu_ops = []
                            for _ in range(min(12, len(b_addr_ops) - addr_idx)):
                                gb, ge = b_addr_ops[addr_idx]
                                alu_ops.append(("+", addr_temps_r[gb][ge], self.scratch["forest_values_p"], v_idx_r[gb] + ge))
                                addr_idx += 1
                            if alu_ops:
                                instr["alu"] = alu_ops
                        self.instrs.append(instr)

                # XOR A
                self.instrs.append({"valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in A]})

                # === INTERLEAVED HASH ===
                # Pattern for each hash stage:
                # Cycle 1: A_op1+op3 (4 slots)
                # Cycle 2: A_op2 (2 slots) + B_op1+op3 (4 slots) = 6 slots
                # But wait, B doesn't have XOR done yet!

                # Let me restructure:
                # 1. Load B nodes during A hash (8 loads = 4 cycles, we have 12 hash cycles for A)
                # 2. XOR B during A index/bounds
                # 3. Hash B stages 0-5 with A_next work

                # Load B node values during A's hash
                b_load_ops = [(b, e) for b in B for e in range(0, VLEN, 2)]
                load_idx = 0

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1, c3 = hash_consts[hi]

                    # A_op1+op3
                    instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in A] +
                                     [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in A]}
                    if load_idx < len(b_load_ops):
                        lb, le = b_load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # A_op2
                    instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in A]}
                    if load_idx < len(b_load_ops):
                        lb, le = b_load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                # Index A + XOR B
                self.instrs.append({"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in A] +
                                            [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in B]})
                self.instrs.append({"valu": [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in A] +
                                            [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in A]})

                # Index A step 3 + Hash B stage 0 op1
                c1_0, c3_0 = hash_consts[0]
                op1_0, _, op2_0, op3_0, _ = HASH_STAGES[0]
                self.instrs.append({"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in A] +
                                            [(op1_0, v_tmp1_r[b], v_val_r[b], c1_0) for b in B]})

                # Bounds A step 1 + Hash B stage 0 op3
                self.instrs.append({"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in A] +
                                            [(op3_0, v_tmp2_r[b], v_val_r[b], c3_0) for b in B]})

                # Bounds A step 2 + Hash B stage 0 op2
                self.instrs.append({"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in A] +
                                            [(op2_0, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in B]})

                # Hash B stages 1-5 + preload A_next
                a_addr_ops = [(b, e) for b in A for e in range(VLEN)] if not is_last_round else []
                a_load_ops = [(b, e) for b in A for e in range(0, VLEN, 2)] if not is_last_round else []
                addr_idx = 0
                load_idx = 0

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    if hi == 0:
                        continue
                    c1, c3 = hash_consts[hi]

                    # B_op1+op3
                    instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in B] +
                                     [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in B]}
                    if addr_idx < len(a_addr_ops):
                        alu_ops = []
                        for _ in range(min(12, len(a_addr_ops) - addr_idx)):
                            ab, ae = a_addr_ops[addr_idx]
                            alu_ops.append(("+", addr_temps_r[ab][ae], self.scratch["forest_values_p"], v_idx_r[ab] + ae))
                            addr_idx += 1
                        if alu_ops:
                            instr["alu"] = alu_ops
                    elif load_idx < len(a_load_ops):
                        lb, le = a_load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # B_op2
                    instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in B]}
                    if addr_idx < len(a_addr_ops):
                        alu_ops = []
                        for _ in range(min(12, len(a_addr_ops) - addr_idx)):
                            ab, ae = a_addr_ops[addr_idx]
                            alu_ops.append(("+", addr_temps_r[ab][ae], self.scratch["forest_values_p"], v_idx_r[ab] + ae))
                            addr_idx += 1
                        if alu_ops:
                            instr["alu"] = alu_ops
                    elif load_idx < len(a_load_ops):
                        lb, le = a_load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                # Index B (continue A_next loads)
                instr = {"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in B] +
                                 [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in B]}
                if load_idx < len(a_load_ops):
                    lb, le = a_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                    ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                    load_idx += 1
                self.instrs.append(instr)

                instr = {"valu": [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in B]}
                if load_idx < len(a_load_ops):
                    lb, le = a_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                    ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                    load_idx += 1
                self.instrs.append(instr)

                instr = {"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in B]}
                if load_idx < len(a_load_ops):
                    lb, le = a_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                    ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                    load_idx += 1
                self.instrs.append(instr)

                # Bounds B
                instr = {"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in B]}
                if load_idx < len(a_load_ops):
                    lb, le = a_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                    ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                    load_idx += 1
                self.instrs.append(instr)

                instr = {"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in B]}
                self.instrs.append(instr)

            # Store results
            for b in range(NUM_BATCHES):
                self.instrs.append({"store": [("vstore", idx_addr_r[b], v_idx_r[b]),
                                              ("vstore", val_addr_r[b], v_val_r[b])]})

            if not is_last_outer_iter:
                self.instrs.append({
                    "alu": [("+", idx_addr_r[i], self.scratch["inp_indices_p"], next_offsets[i]) for i in range(NUM_BATCHES)] +
                           [("+", val_addr_r[i], self.scratch["inp_values_p"], next_offsets[i]) for i in range(NUM_BATCHES)]
                })

            is_first_outer_iter = False

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
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    do_kernel_test(10, 16, 256)
