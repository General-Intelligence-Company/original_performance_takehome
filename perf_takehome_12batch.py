"""
12-batch (4 groups of 3) variant for testing.
Properly pipelined with careful timing.
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
        """
        12-batch (4 groups of 3) kernel with aggressive pipelining.

        Structure per round:
        - 4 groups, each with 3 batches
        - Pipeline: While group N hashes, group N+1 loads
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        NUM_BATCHES = 12
        BATCHES_PER_GROUP = 3
        NUM_GROUPS = 4

        def group(g):
            return list(range(g * BATCHES_PER_GROUP, (g + 1) * BATCHES_PER_GROUP))

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

        for vec_iter in range(0, n_vec_iters, NUM_BATCHES):
            offsets = [batch_offset_consts[min(vec_iter + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)]

            # Load initial idx/val for all batches
            self.instrs.append({
                "alu": [("+", idx_addr_r[i], self.scratch["inp_indices_p"], offsets[i]) for i in range(NUM_BATCHES)]
            })
            self.instrs.append({
                "alu": [("+", val_addr_r[i], self.scratch["inp_values_p"], offsets[i]) for i in range(NUM_BATCHES)]
            })
            for i in range(NUM_BATCHES):
                self.instrs.append({"load": [("vload", v_idx_r[i], idx_addr_r[i]), ("vload", v_val_r[i], val_addr_r[i])]})

            # Compute initial addresses for group 0
            for start in range(0, VLEN * BATCHES_PER_GROUP, 12):
                ops = []
                for idx in range(start, min(start + 12, VLEN * BATCHES_PER_GROUP)):
                    b = idx // VLEN
                    e = idx % VLEN
                    ops.append(("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e))
                if ops:
                    self.instrs.append({"alu": ops})

            for round_idx in range(rounds):
                is_last_round = (round_idx == rounds - 1)
                is_first_round = (round_idx == 0)

                for g in range(NUM_GROUPS):
                    curr = group(g)  # Current group batches
                    next_g = (g + 1) % NUM_GROUPS
                    next_batches = group(next_g)
                    is_last_group = (g == NUM_GROUPS - 1)

                    # Determine what to load next
                    # For groups 0,1,2: load next group's nodes
                    # For group 3: if not last round, load group 0's nodes for next round
                    if is_last_group and is_last_round:
                        # Nothing to preload
                        load_batches = []
                    elif is_last_group:
                        # Load group 0 for next round
                        load_batches = group(0)
                    else:
                        # Load next group
                        load_batches = next_batches

                    # First, we need current group's nodes loaded
                    # For g=0, round=0: load here
                    # For g>0 or round>0: already loaded in previous iteration
                    if is_first_round and g == 0:
                        # Load group 0 nodes
                        for b in curr:
                            for e in range(0, VLEN, 2):
                                self.instrs.append({"load": [
                                    ("load", v_node_val_r[b] + e, addr_temps_r[b][e]),
                                    ("load", v_node_val_r[b] + e + 1, addr_temps_r[b][e + 1])
                                ]})

                    # Compute addresses for next group BEFORE we start XOR
                    # This is critical: addresses must be ready before loads start
                    if load_batches:
                        addr_ops = [("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e)
                                    for b in load_batches for e in range(VLEN)]
                        # 24 ops, need 2 ALU cycles
                        self.instrs.append({"alu": addr_ops[:12]})
                        self.instrs.append({"alu": addr_ops[12:]})

                    # XOR current group
                    self.instrs.append({"valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in curr]})

                    # Hash current group + Load next group
                    load_ops = [(b, e) for b in load_batches for e in range(0, VLEN, 2)] if load_batches else []
                    load_idx = 0

                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        c1, c3 = hash_consts[hi]

                        # op1 + op3 cycle
                        instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in curr] +
                                        [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in curr]}
                        if load_idx < len(load_ops):
                            lb, le = load_ops[load_idx]
                            instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                            ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                            load_idx += 1
                        self.instrs.append(instr)

                        # op2 cycle
                        instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in curr]}
                        if load_idx < len(load_ops):
                            lb, le = load_ops[load_idx]
                            instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                            ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                            load_idx += 1
                        self.instrs.append(instr)

                    # Index/Bounds for current group
                    # Try to finish remaining loads during index

                    # Step 1: bit = val & 1
                    instr = {"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in curr]}
                    if load_idx < len(load_ops):
                        lb, le = load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Steps 2a+2b
                    instr = {"valu": [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in curr] +
                                    [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in curr]}
                    if load_idx < len(load_ops):
                        lb, le = load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Step 3
                    instr = {"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in curr]}
                    if load_idx < len(load_ops):
                        lb, le = load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Bounds step 1
                    instr = {"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in curr]}
                    if load_idx < len(load_ops):
                        lb, le = load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Bounds step 2
                    instr = {"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in curr]}
                    if load_idx < len(load_ops):
                        lb, le = load_ops[load_idx]
                        instr["load"] = [("load", v_node_val_r[lb] + le, addr_temps_r[lb][le]),
                                        ("load", v_node_val_r[lb] + le + 1, addr_temps_r[lb][le + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

            # Store results
            self.instrs.append({
                "alu": [("+", idx_addr_r[i], self.scratch["inp_indices_p"], offsets[i]) for i in range(NUM_BATCHES)]
            })
            self.instrs.append({
                "alu": [("+", val_addr_r[i], self.scratch["inp_values_p"], offsets[i]) for i in range(NUM_BATCHES)]
            })
            for b in range(NUM_BATCHES):
                self.instrs.append({"store": [("vstore", idx_addr_r[b], v_idx_r[b]),
                                              ("vstore", val_addr_r[b], v_val_r[b])]})

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
