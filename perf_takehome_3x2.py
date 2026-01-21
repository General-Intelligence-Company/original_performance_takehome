"""
3x2 Pipelined kernel: 3 groups of 2 batches = 6 batches total.

Key insight: With 2 batches per group:
- op1+op3 = 2+2 = 4 valu slots
- op2 = 2 valu slots
- A_op2 (2) + B_op1+op3 (4) = 6 slots total!

This achieves ~1 cycle per hash-stage-step instead of 2, saving ~3 cycles per round.
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

        NUM_BATCHES = 6
        BATCHES_PER_GROUP = 2
        NUM_GROUPS = 3

        def grp(g):
            return list(range(g * BATCHES_PER_GROUP, (g + 1) * BATCHES_PER_GROUP))

        A, B, C = grp(0), grp(1), grp(2)  # [0,1], [2,3], [4,5]

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

            # Compute all gather addresses (6 batches * 8 = 48 addresses = 4 ALU cycles)
            all_addr_ops = [(b, e) for b in range(NUM_BATCHES) for e in range(VLEN)]
            for start in range(0, len(all_addr_ops), 12):
                ops = []
                for idx in range(start, min(start + 12, len(all_addr_ops))):
                    b, e = all_addr_ops[idx]
                    ops.append(("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e))
                self.instrs.append({"alu": ops})

            # Load all node values (6 batches * 4 load pairs = 24 loads = 12 cycles)
            for b in range(NUM_BATCHES):
                for e in range(0, VLEN, 2):
                    self.instrs.append({"load": [("load", v_node_val_r[b] + e, addr_temps_r[b][e]),
                                                  ("load", v_node_val_r[b] + e + 1, addr_temps_r[b][e + 1])]})

            for round_idx in range(rounds):
                is_last_round = (round_idx == rounds - 1)

                # XOR all groups at once (1 cycle)
                self.instrs.append({"valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in range(NUM_BATCHES)]})

                # === PIPELINED HASH ===
                # Pattern: A_op1+op3, then A_op2+B_op1+op3, B_op2+C_op1+op3, C_op2+A_op1+op3[next_stage], ...

                groups = [A, B, C]
                num_stages = len(HASH_STAGES)

                # We'll emit cycles in a pipeline fashion
                # State: for each group, which stage are they at?
                # Initially all at -1 (not started)

                # Simpler approach: unroll manually for 6 stages × 3 groups = 18 stage-ops
                # But they're pipelined, so it takes fewer cycles

                # Total stage-ops: 6 stages × 3 groups = 18
                # Per cycle in steady state: 2 stage-ops (one op2, one op1+op3)
                # Ramp-up: 1 cycle (A_op1+op3)
                # Ramp-down: 2 cycles (B_op2+C_op1+op3, C_op2)

                # Actually let me just emit it cycle by cycle

                # Cycle 0: A stage 0 op1+op3
                c1_0, c3_0 = hash_consts[0]
                op1_0, _, op2_0, op3_0, _ = HASH_STAGES[0]
                self.instrs.append({"valu": [(op1_0, v_tmp1_r[b], v_val_r[b], c1_0) for b in A] +
                                            [(op3_0, v_tmp2_r[b], v_val_r[b], c3_0) for b in A]})

                # Cycle 1: A stage 0 op2 + B stage 0 op1+op3
                self.instrs.append({"valu": [(op2_0, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in A] +
                                            [(op1_0, v_tmp1_r[b], v_val_r[b], c1_0) for b in B] +
                                            [(op3_0, v_tmp2_r[b], v_val_r[b], c3_0) for b in B]})

                # Cycle 2: B stage 0 op2 + C stage 0 op1+op3
                self.instrs.append({"valu": [(op2_0, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in B] +
                                            [(op1_0, v_tmp1_r[b], v_val_r[b], c1_0) for b in C] +
                                            [(op3_0, v_tmp2_r[b], v_val_r[b], c3_0) for b in C]})

                # Now continue pattern for stages 1-5
                for stage in range(1, num_stages):
                    c1, c3 = hash_consts[stage]
                    op1, _, op2, op3, _ = HASH_STAGES[stage]
                    c1_prev, c3_prev = hash_consts[stage - 1] if stage > 0 else (None, None)
                    op1_prev, _, op2_prev, op3_prev, _ = HASH_STAGES[stage - 1] if stage > 0 else (None, None, None, None, None)

                    # Cycle: C stage (stage-1) op2 + A stage (stage) op1+op3
                    self.instrs.append({"valu": [(op2_prev, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in C] +
                                                [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in A] +
                                                [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in A]})

                    # Cycle: A stage op2 + B stage op1+op3
                    self.instrs.append({"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in A] +
                                                [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in B] +
                                                [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in B]})

                    # Cycle: B stage op2 + C stage op1+op3
                    self.instrs.append({"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in B] +
                                                [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in C] +
                                                [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in C]})

                # Final ramp-down: C stage 5 op2
                c1_5, c3_5 = hash_consts[5]
                op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
                self.instrs.append({"valu": [(op2_5, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in C]})

                # === INDEX/BOUNDS for all groups ===
                # Can we pipeline these too? Let's do them sequentially for now
                for g in [A, B, C]:
                    # Index step 1: bit = val & 1
                    self.instrs.append({"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in g]})
                    # Index step 2: shift idx
                    self.instrs.append({"valu": [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in g]})
                    # Index step 3: child offset
                    self.instrs.append({"valu": [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in g]})
                    # Index step 4: combine
                    self.instrs.append({"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in g]})
                    # Bounds step 1
                    self.instrs.append({"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in g]})
                    # Bounds step 2
                    self.instrs.append({"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in g]})

                # Compute next round addresses and load
                if not is_last_round:
                    for start in range(0, len(all_addr_ops), 12):
                        ops = []
                        for idx in range(start, min(start + 12, len(all_addr_ops))):
                            b, e = all_addr_ops[idx]
                            ops.append(("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e))
                        self.instrs.append({"alu": ops})

                    for b in range(NUM_BATCHES):
                        for e in range(0, VLEN, 2):
                            self.instrs.append({"load": [("load", v_node_val_r[b] + e, addr_temps_r[b][e]),
                                                          ("load", v_node_val_r[b] + e + 1, addr_temps_r[b][e + 1])]})

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
