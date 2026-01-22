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
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
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

    def alloc_vconst(self, val, name=None):
        """Allocate a vector constant (broadcast scalar to VLEN elements)"""
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

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
        Highly optimized VLIW SIMD kernel with aggressive pipelining.
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

        NUM_BATCHES = 6
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
        v_zero = self.alloc_vconst(0)

        # For early round optimization: scratch for selection
        v_node_left = self.alloc_scratch("v_node_left", VLEN)
        v_node_right = self.alloc_scratch("v_node_right", VLEN)
        v_diff = self.alloc_scratch("v_diff", VLEN)  # left ^ right
        v_mask = [self.alloc_scratch(f"v_mask_{i}", VLEN) for i in range(NUM_BATCHES)]
        addr_1 = self.alloc_scratch("addr_1")  # For early round optimization
        addr_2 = self.alloc_scratch("addr_2")
        s_one = self.scratch_const(1, "s_one")  # Scalar constant 1
        s_two = self.scratch_const(2, "s_two")  # Scalar constant 2

        hash_consts = []
        madd_stages = {}  # Stages that can use multiply_add: {stage_idx: (mult_addr, const_addr)}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_consts.append((self.alloc_vconst(val1), self.alloc_vconst(val3)))
            # For stages where: (val + const1) + (val << shift) = val * (1 + 2^shift) + const1
            if op1 == '+' and op2 == '+' and op3 == '<<':
                mult = 1 + (1 << val3)
                madd_stages[hi] = (self.alloc_vconst(mult), self.alloc_vconst(val1))

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        n_vec_iters = batch_size // VLEN
        batch_offset_consts = [self.scratch_const(vi * VLEN) for vi in range(n_vec_iters)]

        self.add("flow", ("pause",))

        # Track if we need to compute addresses (first iter) or they're already computed (subsequent iters)
        is_first_outer_iter = True

        for vec_iter in range(0, n_vec_iters, NUM_BATCHES):
            offsets = [batch_offset_consts[min(vec_iter + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)]
            next_vec_iter = vec_iter + NUM_BATCHES
            is_last_outer_iter = (next_vec_iter >= n_vec_iters)
            next_offsets = [batch_offset_consts[min(next_vec_iter + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)] if not is_last_outer_iter else None

            # Load initial idx/val
            if is_first_outer_iter:
                # First iteration: compute addresses then load
                self.instrs.append({
                    "alu": [("+", idx_addr_r[i], self.scratch["inp_indices_p"], offsets[i]) for i in range(NUM_BATCHES)] +
                           [("+", val_addr_r[i], self.scratch["inp_values_p"], offsets[i]) for i in range(NUM_BATCHES)]
                })
            # For non-first iterations, addresses were computed during previous iteration's stores

            # Load initial idx/val and compute gather addresses in parallel
            # vloads use load slot, addr computation uses ALU slot
            addr_ops = [(b, e) for b in range(3) for e in range(VLEN)]  # 24 ops for group A
            addr_idx = 0
            for i in range(NUM_BATCHES):
                instr = {"load": [("vload", v_idx_r[i], idx_addr_r[i]), ("vload", v_val_r[i], val_addr_r[i])]}
                # Overlap ALU addr computation with vloads (after first batch loaded idx values)
                if i >= 1 and addr_idx < len(addr_ops):
                    alu_ops = []
                    for _ in range(min(12, len(addr_ops) - addr_idx)):
                        b, e = addr_ops[addr_idx]
                        # Note: v_idx_r[b] might not be loaded yet if b > i-1
                        # We can only use indices from already-loaded batches
                        if b < i:  # Batch b already loaded
                            alu_ops.append(("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e))
                            addr_idx += 1
                    if alu_ops:
                        instr["alu"] = alu_ops
                self.instrs.append(instr)

            # Complete remaining gather addresses that couldn't overlap
            while addr_idx < len(addr_ops):
                ops = []
                for _ in range(min(12, len(addr_ops) - addr_idx)):
                    b, e = addr_ops[addr_idx]
                    ops.append(("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e))
                    addr_idx += 1
                self.instrs.append({"alu": ops})

            # Round 11 (forest_height+1) is where all indices wrap back to 0
            wraparound_round = forest_height + 1 if forest_height + 1 < rounds else -1

            for round_idx in range(rounds):
                is_last_round = (round_idx == rounds - 1)
                is_first_round = (round_idx == 0)
                # Only use all-zero optimization for round 0 (not wraparound) since
                # wraparound round has A nodes already loaded from previous round
                is_all_zero = is_first_round

                if is_all_zero:
                    # OPTIMIZATION: Round 0 all indices are 0, load once, broadcast, XOR all
                    # Load forest_values[0] (1 cycle)
                    self.instrs.append({"load": [("load", v_node_val_r[0], self.scratch["forest_values_p"])]})
                    # Broadcast to all 6 batch vectors (1 cycle)
                    self.instrs.append({"valu": [("vbroadcast", v_node_val_r[b], v_node_val_r[0]) for b in range(NUM_BATCHES)]})
                    # XOR all 6 batches at once (1 cycle)
                    self.instrs.append({"valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in range(NUM_BATCHES)]})
                else:
                    self.instrs.append({
                        "valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in range(3)],
                        "alu": [("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e)
                                for b in range(3, 6) for e in range(4)]
                    })

                # For all-zero rounds, we don't need to load B nodes (already broadcast)
                # For round 1 in FIRST outer iteration only: use early round optimization
                # (Only works when all indices start at 0, which is true for the first outer iter)
                # DISABLED: Early round optimization adds more cycles than it saves
                is_early_round_1 = False
                if is_all_zero:
                    node_load_b = []
                elif is_early_round_1:
                    # Round 1: B indices are either 1 or 2
                    # We'll load those 2 values and select based on B's indices
                    node_load_b = []  # Don't load individual B nodes
                else:
                    node_load_b = [(b, e) for b in range(3, 6) for e in range(0, VLEN, 2)]
                load_idx = 0
                need_remaining_b_addr = not is_first_round and not is_all_zero and not is_early_round_1

                # For early round 1: load values at indices 1 and 2 during first Hash A cycle
                early_round_selection_done = False

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1, c3 = hash_consts[hi]

                    if hi in madd_stages:
                        # multiply_add: 1 cycle instead of 2 for stages 0, 2, 4
                        mult_addr, const_addr = madd_stages[hi]
                        instr = {"valu": [("multiply_add", v_val_r[b], v_val_r[b], mult_addr, const_addr) for b in range(3)]}
                        if load_idx < len(node_load_b):
                            b_l, e_l = node_load_b[load_idx]
                            instr["load"] = [("load", v_node_val_r[b_l] + e_l, addr_temps_r[b_l][e_l]),
                                            ("load", v_node_val_r[b_l] + e_l + 1, addr_temps_r[b_l][e_l + 1])]
                            load_idx += 1
                        if need_remaining_b_addr and hi == 0:
                            instr["alu"] = [("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e)
                                            for b in range(3, 6) for e in range(4, VLEN)]
                            need_remaining_b_addr = False
                        self.instrs.append(instr)
                        continue

                    # Standard 2-cycle hash stage (for stages 1, 3, 5)
                    instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in range(3)] +
                                     [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in range(3)]}
                    if load_idx < len(node_load_b):
                        b_l, e_l = node_load_b[load_idx]
                        instr["load"] = [("load", v_node_val_r[b_l] + e_l, addr_temps_r[b_l][e_l]),
                                        ("load", v_node_val_r[b_l] + e_l + 1, addr_temps_r[b_l][e_l + 1])]
                        load_idx += 1
                    if need_remaining_b_addr and hi == 0:
                        instr["alu"] = [("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e)
                                        for b in range(3, 6) for e in range(4, VLEN)]
                        need_remaining_b_addr = False
                    self.instrs.append(instr)

                    instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in range(3)]}
                    if load_idx < len(node_load_b):
                        b_l, e_l = node_load_b[load_idx]
                        instr["load"] = [("load", v_node_val_r[b_l] + e_l, addr_temps_r[b_l][e_l]),
                                        ("load", v_node_val_r[b_l] + e_l + 1, addr_temps_r[b_l][e_l + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                # Complete any remaining B node loads that didn't fit in Hash A
                # OR for early round 1: do selection-based node value creation
                index_a_step1_done = False
                index_a_step2_done = False
                index_a_step3_done = False
                if is_early_round_1:
                    # Early Round 1 Optimization:
                    # B indices are either 1 or 2. Load those 2 values and select.

                    # Compute addresses: forest_values_p + 1 and forest_values_p + 2
                    # Using scalar constants s_one and s_two
                    self.instrs.append({"alu": [("+", addr_1, self.scratch["forest_values_p"], s_one),
                                                ("+", addr_2, self.scratch["forest_values_p"], s_two)]})

                    # Load values at indices 1 and 2 (scalar load)
                    self.instrs.append({"load": [("load", v_node_left, addr_1),
                                                 ("load", v_node_right, addr_2)]})

                    # Broadcast to vector form
                    self.instrs.append({"valu": [("vbroadcast", v_node_left, v_node_left),
                                                 ("vbroadcast", v_node_right, v_node_right)]})

                    # Compute diff = left ^ right (once, used for all batches)
                    self.instrs.append({"valu": [("^", v_diff, v_node_left, v_node_right)]})

                    # For each B batch, select based on index
                    # cond = (idx == 1)  -> 1 if left child (idx 1), 0 if right child (idx 2)
                    # mask = 0 - cond   -> 0xFFFFFFFF if left, 0 if right
                    # result = right ^ (diff & mask)  -> left if mask=all_1s, right if mask=0
                    # VALU has 6 slots, so we can do 2 batches per cycle (2 ops each)
                    # Step 1: cond = (idx == 1) for all 3 batches - 3 ops
                    self.instrs.append({"valu": [("==", v_cond_r[b], v_idx_r[b], v_one) for b in range(3, 6)]})
                    # Step 2: mask = 0 - cond for all 3 batches - 3 ops
                    self.instrs.append({"valu": [("-", v_mask[b], v_zero, v_cond_r[b]) for b in range(3, 6)]})
                    # Step 3: masked = diff & mask for all 3 batches - 3 ops
                    self.instrs.append({"valu": [("&", v_mask[b], v_diff, v_mask[b]) for b in range(3, 6)]})
                    # Step 4: result = right ^ masked for all 3 batches - 3 ops
                    self.instrs.append({"valu": [("^", v_node_val_r[b], v_node_right, v_mask[b]) for b in range(3, 6)]})
                else:
                    # Overlap remaining B loads with Index A steps 1, 2, and 3
                    # Index A step 1: tmp1 = val & 1 (extract LSB)
                    # Index A step 2: tmp2 = idx << 1 (shift index, no dependency on step 1)
                    # Index A step 3: node = tmp1 + 1 (depends on step 1)
                    index_a_step1_done = False
                    index_a_step2_done = False
                    index_a_step3_done = False
                    while load_idx < len(node_load_b):
                        b_l, e_l = node_load_b[load_idx]
                        instr = {"load": [("load", v_node_val_r[b_l] + e_l, addr_temps_r[b_l][e_l]),
                                         ("load", v_node_val_r[b_l] + e_l + 1, addr_temps_r[b_l][e_l + 1])]}
                        # Overlap Index A step 1 with first remaining B load cycle
                        if not index_a_step1_done:
                            instr["valu"] = [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in range(3)]
                            index_a_step1_done = True
                        # Overlap Index A step 2 with second remaining B load cycle
                        elif not index_a_step2_done:
                            instr["valu"] = [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in range(3)]
                            index_a_step2_done = True
                        # Overlap Index A step 3 with third remaining B load cycle (step 1 must be done)
                        elif not index_a_step3_done:
                            instr["valu"] = [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in range(3)]
                            index_a_step3_done = True
                        self.instrs.append(instr)
                        load_idx += 1

                # Index A step 1 + XOR B (but skip XOR B for all-zero rounds - already done)
                # Also skip Index A step 1 if it was done during extra B loads
                if is_all_zero:
                    self.instrs.append({"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in range(3)]})
                elif is_early_round_1:
                    # Early round: Index A step 1 not done during extra B loads (no extra loads)
                    self.instrs.append({"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in range(3)] +
                                                [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in range(3, 6)]})
                elif index_a_step1_done:
                    # Index A step 1 already done, just do XOR B
                    self.instrs.append({"valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in range(3, 6)]})
                else:
                    self.instrs.append({"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in range(3)] +
                                                [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in range(3, 6)]})
                # Index A steps 2-3 (skip steps already done during extra B loads)
                if index_a_step2_done and index_a_step3_done:
                    # Both steps already done - skip this instruction entirely
                    pass
                elif index_a_step2_done:
                    # Only step 3: node = tmp1 + 1
                    self.instrs.append({"valu": [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in range(3)]})
                elif index_a_step3_done:
                    # Only step 2: tmp2 = idx << 1 (unusual case)
                    self.instrs.append({"valu": [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in range(3)]})
                else:
                    # Both steps 2 and 3
                    self.instrs.append({"valu": [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in range(3)] +
                                                [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in range(3)]})

                # Index A step 4 + Hash B stage 0 (multiply_add since stage 0 is madd-able)
                mult_addr_0, const_addr_0 = madd_stages[0]
                self.instrs.append({"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in range(3)] +
                                            [("multiply_add", v_val_r[b], v_val_r[b], mult_addr_0, const_addr_0) for b in range(3, 6)]})

                # Bounds A step 1 + Hash B stage 1 op1 (stage 1 is XOR-based, not multiply_add)
                op1_1, _, _, op3_1, _ = HASH_STAGES[1]
                c1_1, c3_1 = hash_consts[1]
                self.instrs.append({"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in range(3)] +
                                            [(op1_1, v_tmp1_r[b], v_val_r[b], c1_1) for b in range(3, 6)]})

                # Bounds A step 2 + Hash B stage 1 op3
                self.instrs.append({"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in range(3)] +
                                            [(op3_1, v_tmp2_r[b], v_val_r[b], c3_1) for b in range(3, 6)]})

                next_addr_ops = [(b, e) for b in range(3) for e in range(VLEN)] if not is_last_round else []
                next_load_ops = [(b, e) for b in range(3) for e in range(0, VLEN, 2)] if not is_last_round else []
                addr_idx = 0
                load_idx = 0

                # Hash B stages 1-5 (stage 0 already done, stage 1 op1/op3 done above with Bounds A)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    if hi == 0:
                        continue  # Skip stage 0, already done
                    c1, c3 = hash_consts[hi]

                    # Use multiply_add for stages 2 and 4 (op1=+, op2=+, op3=<<)
                    if hi in madd_stages:
                        mult_addr, const_addr = madd_stages[hi]
                        instr = {"valu": [("multiply_add", v_val_r[b], v_val_r[b], mult_addr, const_addr) for b in range(3, 6)]}
                        # INTERLEAVED scheduling: do ALU AND loads in same cycle
                        # Compute addresses
                        if addr_idx < len(next_addr_ops):
                            alu_ops = []
                            for _ in range(min(12, len(next_addr_ops) - addr_idx)):
                                nb, ne = next_addr_ops[addr_idx]
                                alu_ops.append(("+", addr_temps_r[nb][ne], self.scratch["forest_values_p"], v_idx_r[nb] + ne))
                                addr_idx += 1
                            if alu_ops:
                                instr["alu"] = alu_ops
                        # Also do loads if addresses from PREVIOUS cycle are ready
                        # (addresses computed in this cycle won't be ready until next cycle)
                        if load_idx < len(next_load_ops):
                            nb, ne = next_load_ops[load_idx]
                            # Address for (nb, ne) is at position nb*VLEN + ne in next_addr_ops
                            # It's ready if we computed it BEFORE this cycle
                            # Since we compute 12 addresses per cycle, and this is cumulative,
                            # addr_idx BEFORE this cycle was addr_idx - len(alu_ops computed above)
                            prev_addr_idx = addr_idx - (12 if addr_idx >= 12 else addr_idx)
                            addr_linear = nb * VLEN + ne
                            # Need both ne and ne+1 addresses to be ready (computed in previous cycles)
                            if addr_linear + 1 < prev_addr_idx:
                                instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                                ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                                load_idx += 1
                        self.instrs.append(instr)
                        continue  # Skip 2-cycle path for this stage

                    # For stage 1, op1/op3 were already done overlapped with Bounds A
                    # So we only need to emit op2
                    if hi == 1:
                        instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in range(3, 6)]}
                        if addr_idx < len(next_addr_ops):
                            alu_ops = []
                            alu_count = min(12, len(next_addr_ops) - addr_idx)
                            for _ in range(alu_count):
                                nb, ne = next_addr_ops[addr_idx]
                                alu_ops.append(("+", addr_temps_r[nb][ne], self.scratch["forest_values_p"], v_idx_r[nb] + ne))
                                addr_idx += 1
                            if alu_ops:
                                instr["alu"] = alu_ops
                        else:
                            alu_count = 0
                        if load_idx < len(next_load_ops):
                            nb, ne = next_load_ops[load_idx]
                            prev_addr_idx = addr_idx - alu_count
                            addr_linear = nb * VLEN + ne
                            if addr_linear + 1 < prev_addr_idx:
                                instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                                ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                                load_idx += 1
                        self.instrs.append(instr)
                        continue

                    # Standard 2-cycle path for stages 3, 5
                    instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in range(3, 6)] +
                                     [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in range(3, 6)]}
                    if addr_idx < len(next_addr_ops):
                        alu_ops = []
                        alu_count = min(12, len(next_addr_ops) - addr_idx)
                        for _ in range(alu_count):
                            nb, ne = next_addr_ops[addr_idx]
                            alu_ops.append(("+", addr_temps_r[nb][ne], self.scratch["forest_values_p"], v_idx_r[nb] + ne))
                            addr_idx += 1
                        if alu_ops:
                            instr["alu"] = alu_ops
                    else:
                        alu_count = 0
                    if load_idx < len(next_load_ops):
                        nb, ne = next_load_ops[load_idx]
                        prev_addr_idx = addr_idx - alu_count
                        addr_linear = nb * VLEN + ne
                        if addr_linear + 1 < prev_addr_idx:
                            instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                            ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                            load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in range(3, 6)]}
                    if addr_idx < len(next_addr_ops):
                        alu_ops = []
                        alu_count = min(12, len(next_addr_ops) - addr_idx)
                        for _ in range(alu_count):
                            nb, ne = next_addr_ops[addr_idx]
                            alu_ops.append(("+", addr_temps_r[nb][ne], self.scratch["forest_values_p"], v_idx_r[nb] + ne))
                            addr_idx += 1
                        if alu_ops:
                            instr["alu"] = alu_ops
                    else:
                        alu_count = 0
                    if load_idx < len(next_load_ops):
                        nb, ne = next_load_ops[load_idx]
                        prev_addr_idx = addr_idx - alu_count
                        addr_linear = nb * VLEN + ne
                        if addr_linear + 1 < prev_addr_idx:
                            instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                            ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                            load_idx += 1
                    self.instrs.append(instr)

                instr = {"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in range(3, 6)] +
                                 [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in range(3, 6)]}
                if load_idx < len(next_load_ops):
                    nb, ne = next_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                    ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                    load_idx += 1
                self.instrs.append(instr)

                instr = {"valu": [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in range(3, 6)]}
                if load_idx < len(next_load_ops):
                    nb, ne = next_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                    ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                    load_idx += 1
                self.instrs.append(instr)

                instr = {"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in range(3, 6)]}
                if load_idx < len(next_load_ops):
                    nb, ne = next_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                    ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                    load_idx += 1
                elif is_last_round:
                    # Overlap store batch 0 with Index B step 3
                    instr["store"] = [("vstore", idx_addr_r[0], v_idx_r[0]), ("vstore", val_addr_r[0], v_val_r[0])]
                self.instrs.append(instr)

                instr = {"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in range(3, 6)]}
                if load_idx < len(next_load_ops):
                    nb, ne = next_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                    ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                    load_idx += 1
                elif is_last_round:
                    # Overlap store batch 1 with Index B step 4
                    instr["store"] = [("vstore", idx_addr_r[1], v_idx_r[1]), ("vstore", val_addr_r[1], v_val_r[1])]
                self.instrs.append(instr)

                instr = {"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in range(3, 6)]}
                if load_idx < len(next_load_ops):
                    nb, ne = next_load_ops[load_idx]
                    instr["load"] = [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                    ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]
                    load_idx += 1
                elif is_last_round:
                    # Overlap store batch 2 with Index B step 5
                    instr["store"] = [("vstore", idx_addr_r[2], v_idx_r[2]), ("vstore", val_addr_r[2], v_val_r[2])]
                self.instrs.append(instr)

                # Extra load cycle if needed (when using multiply_add we save cycles but need more load slots)
                while load_idx < len(next_load_ops):
                    nb, ne = next_load_ops[load_idx]
                    instr = {"load": [("load", v_node_val_r[nb] + ne, addr_temps_r[nb][ne]),
                                     ("load", v_node_val_r[nb] + ne + 1, addr_temps_r[nb][ne + 1])]}
                    load_idx += 1
                    self.instrs.append(instr)

            # Store results - for last round of rounds, batches 0-2 already stored above
            # Only need to store batches 3-5
            # If not last outer iteration, overlap ALU for next iter's addresses with LAST store
            for b in range(3, NUM_BATCHES - 1):
                self.instrs.append({"store": [("vstore", idx_addr_r[b], v_idx_r[b]),
                                              ("vstore", val_addr_r[b], v_val_r[b])]})

            if not is_last_outer_iter:
                # Compute next iteration's addresses during last store (batch 5)
                self.instrs.append({
                    "alu": [("+", idx_addr_r[i], self.scratch["inp_indices_p"], next_offsets[i]) for i in range(NUM_BATCHES)] +
                           [("+", val_addr_r[i], self.scratch["inp_values_p"], next_offsets[i]) for i in range(NUM_BATCHES)],
                    "store": [("vstore", idx_addr_r[5], v_idx_r[5]), ("vstore", val_addr_r[5], v_val_r[5])]
                })
            else:
                # Last outer iteration: just store, no next iter to precompute
                self.instrs.append({"store": [("vstore", idx_addr_r[5], v_idx_r[5]),
                                              ("vstore", val_addr_r[5], v_val_r[5])]})

            is_first_outer_iter = False

        self.instrs.append({"flow": [("pause",)]})

    def build_kernel_12batch(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        12-batch kernel with 4 groups (A/B/C/D) for better load/VALU overlap.

        Groups (3 batches each):
        - A: batches 0, 1, 2
        - B: batches 3, 4, 5
        - C: batches 6, 7, 8
        - D: batches 9, 10, 11

        With 12 batches, we have 12*8 = 96 elements per outer iteration.
        More VALU work means better overlap with loads.
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
        NUM_GROUPS = 4
        BATCHES_PER_GROUP = 3

        # Allocate registers for all 12 batches
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

        # Hash constants and multiply_add stages
        hash_consts = []
        madd_stages = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_consts.append((self.alloc_vconst(val1), self.alloc_vconst(val3)))
            if op1 == '+' and op2 == '+' and op3 == '<<':
                mult = 1 + (1 << val3)
                madd_stages[hi] = (self.alloc_vconst(mult), self.alloc_vconst(val1))

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        n_vec_iters = batch_size // VLEN
        batch_offset_consts = [self.scratch_const(vi * VLEN) for vi in range(n_vec_iters)]

        self.add("flow", ("pause",))

        # Group definitions: A=0-2, B=3-5, C=6-8, D=9-11
        groups = [list(range(g*3, g*3+3)) for g in range(NUM_GROUPS)]

        # Outer loop: process 12 batches (96 elements) at a time
        for vec_iter in range(0, n_vec_iters, NUM_BATCHES):
            is_first_outer_iter = (vec_iter == 0)
            is_last_outer_iter = (vec_iter + NUM_BATCHES >= n_vec_iters)

            offsets = [batch_offset_consts[min(vec_iter + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)]
            next_offsets = [batch_offset_consts[min(vec_iter + NUM_BATCHES + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)]

            if is_first_outer_iter:
                # Compute idx/val addresses
                self.instrs.append({
                    "alu": [("+", idx_addr_r[i], self.scratch["inp_indices_p"], offsets[i]) for i in range(12)]
                })
                self.instrs.append({
                    "alu": [("+", val_addr_r[i], self.scratch["inp_values_p"], offsets[i]) for i in range(12)]
                })

            # Load initial idx/val for all 12 batches (12 vloads = 6 cycles at 2 per cycle)
            for i in range(0, NUM_BATCHES, 2):
                self.instrs.append({"load": [("vload", v_idx_r[i], idx_addr_r[i]),
                                             ("vload", v_idx_r[i+1], idx_addr_r[i+1])]})
            for i in range(0, NUM_BATCHES, 2):
                self.instrs.append({"load": [("vload", v_val_r[i], val_addr_r[i]),
                                             ("vload", v_val_r[i+1], val_addr_r[i+1])]})

            for round_idx in range(rounds):
                is_last_round = (round_idx == rounds - 1)
                is_first_round = (round_idx == 0)

                if is_first_round:
                    # Round 0: all indices are 0, load once and broadcast
                    self.instrs.append({"load": [("load", v_node_val_r[0], self.scratch["forest_values_p"])]})
                    self.instrs.append({"valu": [("vbroadcast", v_node_val_r[b], v_node_val_r[0]) for b in range(6)]})
                    self.instrs.append({"valu": [("vbroadcast", v_node_val_r[b], v_node_val_r[0]) for b in range(6, NUM_BATCHES)]})
                    # XOR all 12 batches (2 cycles of 6 VALU each)
                    self.instrs.append({"valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in range(6)]})
                    self.instrs.append({"valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in range(6, NUM_BATCHES)]})
                else:
                    # Non-zero rounds: process groups A, B, C, D in a pipelined fashion
                    # Group A's node values were loaded at end of previous round
                    # Need to load B, C, D's node values during Hash A/B

                    # XOR Group A + start computing B's addresses
                    self.instrs.append({
                        "valu": [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in groups[0]],
                        "alu": [("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e)
                                for b in groups[1] for e in range(4)]
                    })
                    # Compute remaining B addresses (12 ops)
                    self.instrs.append({
                        "alu": [("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e)
                                for b in groups[1] for e in range(4, VLEN)]
                    })
                    # Compute first C addresses (12 ops)
                    self.instrs.append({
                        "alu": [("+", addr_temps_r[b][e], self.scratch["forest_values_p"], v_idx_r[b] + e)
                                for b in groups[2] for e in range(4)]
                    })

                    # B node loads to do: 3 batches * 4 load ops = 12 load ops
                    # C node loads: 12 load ops
                    # D node loads: 12 load ops
                    # Total: 36 load ops for groups B, C, D

                    b_loads = [(b, e) for b in groups[1] for e in range(0, VLEN, 2)]
                    c_loads = [(b, e) for b in groups[2] for e in range(0, VLEN, 2)]
                    d_loads = [(b, e) for b in groups[3] for e in range(0, VLEN, 2)]
                    all_bcd_loads = b_loads + c_loads + d_loads
                    load_idx = 0

                    # Hash A (6 stages, groups 0-2) with B loads overlapped
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        c1, c3 = hash_consts[hi]

                        if hi in madd_stages:
                            mult_addr, const_addr = madd_stages[hi]
                            instr = {"valu": [("multiply_add", v_val_r[b], v_val_r[b], mult_addr, const_addr)
                                              for b in groups[0]]}
                        else:
                            instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in groups[0]] +
                                             [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in groups[0]]}

                        if load_idx < len(all_bcd_loads):
                            bl, el = all_bcd_loads[load_idx]
                            instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                            ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                            load_idx += 1
                        self.instrs.append(instr)

                        if hi not in madd_stages:
                            instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in groups[0]]}
                            if load_idx < len(all_bcd_loads):
                                bl, el = all_bcd_loads[load_idx]
                                instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                                ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                                load_idx += 1
                            self.instrs.append(instr)

                    # Index A + XOR B + continue C/D address computation
                    c_addr_ops = [(b, e) for b in groups[2] for e in range(4, VLEN)]
                    d_addr_ops = [(b, e) for b in groups[3] for e in range(VLEN)]
                    all_cd_addr = c_addr_ops + d_addr_ops
                    addr_idx = 0

                    # Index A step 1 + XOR B
                    instr = {"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in groups[0]] +
                                     [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in groups[1]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Index A steps 2-3
                    instr = {"valu": [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in groups[0]] +
                                     [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in groups[0]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    if addr_idx < len(all_cd_addr):
                        alu_ops = []
                        for _ in range(min(12, len(all_cd_addr) - addr_idx)):
                            ab, ae = all_cd_addr[addr_idx]
                            alu_ops.append(("+", addr_temps_r[ab][ae], self.scratch["forest_values_p"], v_idx_r[ab] + ae))
                            addr_idx += 1
                        instr["alu"] = alu_ops
                    self.instrs.append(instr)

                    # Index A step 4 + Hash B stage 0
                    mult_addr_0, const_addr_0 = madd_stages[0]
                    instr = {"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in groups[0]] +
                                     [("multiply_add", v_val_r[b], v_val_r[b], mult_addr_0, const_addr_0) for b in groups[1]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    if addr_idx < len(all_cd_addr):
                        alu_ops = []
                        for _ in range(min(12, len(all_cd_addr) - addr_idx)):
                            ab, ae = all_cd_addr[addr_idx]
                            alu_ops.append(("+", addr_temps_r[ab][ae], self.scratch["forest_values_p"], v_idx_r[ab] + ae))
                            addr_idx += 1
                        instr["alu"] = alu_ops
                    self.instrs.append(instr)

                    # Bounds A + Hash B stage 1 op1
                    op1_1, _, _, op3_1, _ = HASH_STAGES[1]
                    c1_1, c3_1 = hash_consts[1]
                    instr = {"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in groups[0]] +
                                     [(op1_1, v_tmp1_r[b], v_val_r[b], c1_1) for b in groups[1]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Bounds A step 2 + Hash B stage 1 op3
                    instr = {"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in groups[0]] +
                                     [(op3_1, v_tmp2_r[b], v_val_r[b], c3_1) for b in groups[1]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Hash B stages 1-5 + Index B + XOR C + Hash C + Index C + XOR D + Hash D + Index D
                    # This is where we really benefit from more batches!

                    # Continue Hash B (stages 1 op2 through 5)
                    for hi in range(1, 6):
                        op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                        c1, c3 = hash_consts[hi]

                        if hi == 1:
                            # Just op2 (op1/op3 done above)
                            instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in groups[1]]}
                        elif hi in madd_stages:
                            mult_addr, const_addr = madd_stages[hi]
                            instr = {"valu": [("multiply_add", v_val_r[b], v_val_r[b], mult_addr, const_addr) for b in groups[1]]}
                        else:
                            instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in groups[1]] +
                                             [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in groups[1]]}

                        if load_idx < len(all_bcd_loads):
                            bl, el = all_bcd_loads[load_idx]
                            instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                            ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                            load_idx += 1
                        self.instrs.append(instr)

                        if hi not in madd_stages and hi != 1:
                            instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in groups[1]]}
                            if load_idx < len(all_bcd_loads):
                                bl, el = all_bcd_loads[load_idx]
                                instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                                ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                                load_idx += 1
                            self.instrs.append(instr)

                    # Index B + XOR C
                    instr = {"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in groups[1]] +
                                     [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in groups[2]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in groups[1]] +
                                     [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in groups[1]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Index B step 4 + Hash C stage 0
                    instr = {"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in groups[1]] +
                                     [("multiply_add", v_val_r[b], v_val_r[b], mult_addr_0, const_addr_0) for b in groups[2]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Bounds B + Hash C stage 1 ops
                    instr = {"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in groups[1]] +
                                     [(op1_1, v_tmp1_r[b], v_val_r[b], c1_1) for b in groups[2]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in groups[1]] +
                                     [(op3_1, v_tmp2_r[b], v_val_r[b], c3_1) for b in groups[2]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Hash C stages 1-5 (similar pattern)
                    for hi in range(1, 6):
                        op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                        c1, c3 = hash_consts[hi]

                        if hi == 1:
                            instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in groups[2]]}
                        elif hi in madd_stages:
                            mult_addr, const_addr = madd_stages[hi]
                            instr = {"valu": [("multiply_add", v_val_r[b], v_val_r[b], mult_addr, const_addr) for b in groups[2]]}
                        else:
                            instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in groups[2]] +
                                             [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in groups[2]]}

                        if load_idx < len(all_bcd_loads):
                            bl, el = all_bcd_loads[load_idx]
                            instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                            ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                            load_idx += 1
                        self.instrs.append(instr)

                        if hi not in madd_stages and hi != 1:
                            instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in groups[2]]}
                            if load_idx < len(all_bcd_loads):
                                bl, el = all_bcd_loads[load_idx]
                                instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                                ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                                load_idx += 1
                            self.instrs.append(instr)

                    # Index C + XOR D
                    instr = {"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in groups[2]] +
                                     [("^", v_val_r[b], v_val_r[b], v_node_val_r[b]) for b in groups[3]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in groups[2]] +
                                     [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in groups[2]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Index C step 4 + Hash D stage 0
                    instr = {"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in groups[2]] +
                                     [("multiply_add", v_val_r[b], v_val_r[b], mult_addr_0, const_addr_0) for b in groups[3]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Bounds C + Hash D stage 1 ops
                    instr = {"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in groups[2]] +
                                     [(op1_1, v_tmp1_r[b], v_val_r[b], c1_1) for b in groups[3]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in groups[2]] +
                                     [(op3_1, v_tmp2_r[b], v_val_r[b], c3_1) for b in groups[3]]}
                    if load_idx < len(all_bcd_loads):
                        bl, el = all_bcd_loads[load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Hash D stages 1-5
                    # For next round, we need to load A's node values
                    next_a_addr_ops = [(b, e) for b in groups[0] for e in range(VLEN)] if not is_last_round else []
                    next_a_load_ops = [(b, e) for b in groups[0] for e in range(0, VLEN, 2)] if not is_last_round else []
                    a_addr_idx = 0
                    a_load_idx = 0

                    for hi in range(1, 6):
                        op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                        c1, c3 = hash_consts[hi]

                        if hi == 1:
                            instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in groups[3]]}
                        elif hi in madd_stages:
                            mult_addr, const_addr = madd_stages[hi]
                            instr = {"valu": [("multiply_add", v_val_r[b], v_val_r[b], mult_addr, const_addr) for b in groups[3]]}
                        else:
                            instr = {"valu": [(op1, v_tmp1_r[b], v_val_r[b], c1) for b in groups[3]] +
                                             [(op3, v_tmp2_r[b], v_val_r[b], c3) for b in groups[3]]}

                        # Overlap with next round A address computation
                        if a_addr_idx < len(next_a_addr_ops):
                            alu_ops = []
                            for _ in range(min(12, len(next_a_addr_ops) - a_addr_idx)):
                                ab, ae = next_a_addr_ops[a_addr_idx]
                                alu_ops.append(("+", addr_temps_r[ab][ae], self.scratch["forest_values_p"], v_idx_r[ab] + ae))
                                a_addr_idx += 1
                            instr["alu"] = alu_ops
                        elif a_load_idx < len(next_a_load_ops):
                            bl, el = next_a_load_ops[a_load_idx]
                            instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                            ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                            a_load_idx += 1
                        self.instrs.append(instr)

                        if hi not in madd_stages and hi != 1:
                            instr = {"valu": [(op2, v_val_r[b], v_tmp1_r[b], v_tmp2_r[b]) for b in groups[3]]}
                            if a_addr_idx < len(next_a_addr_ops):
                                alu_ops = []
                                for _ in range(min(12, len(next_a_addr_ops) - a_addr_idx)):
                                    ab, ae = next_a_addr_ops[a_addr_idx]
                                    alu_ops.append(("+", addr_temps_r[ab][ae], self.scratch["forest_values_p"], v_idx_r[ab] + ae))
                                    a_addr_idx += 1
                                instr["alu"] = alu_ops
                            elif a_load_idx < len(next_a_load_ops):
                                bl, el = next_a_load_ops[a_load_idx]
                                instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                                ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                                a_load_idx += 1
                            self.instrs.append(instr)

                    # Index D with next round A loads
                    instr = {"valu": [("&", v_tmp1_r[b], v_val_r[b], v_one) for b in groups[3]] +
                                     [("<<", v_tmp2_r[b], v_idx_r[b], v_one) for b in groups[3]]}
                    if a_load_idx < len(next_a_load_ops):
                        bl, el = next_a_load_ops[a_load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        a_load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("+", v_node_val_r[b], v_tmp1_r[b], v_one) for b in groups[3]]}
                    if a_load_idx < len(next_a_load_ops):
                        bl, el = next_a_load_ops[a_load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        a_load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("+", v_idx_r[b], v_tmp2_r[b], v_node_val_r[b]) for b in groups[3]]}
                    if a_load_idx < len(next_a_load_ops):
                        bl, el = next_a_load_ops[a_load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        a_load_idx += 1
                    self.instrs.append(instr)

                    # Bounds D
                    instr = {"valu": [("<", v_cond_r[b], v_idx_r[b], v_n_nodes) for b in groups[3]]}
                    if a_load_idx < len(next_a_load_ops):
                        bl, el = next_a_load_ops[a_load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        a_load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("*", v_idx_r[b], v_idx_r[b], v_cond_r[b]) for b in groups[3]]}
                    if a_load_idx < len(next_a_load_ops):
                        bl, el = next_a_load_ops[a_load_idx]
                        instr["load"] = [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                        ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]
                        a_load_idx += 1
                    self.instrs.append(instr)

                    # Any remaining A loads
                    while a_load_idx < len(next_a_load_ops):
                        bl, el = next_a_load_ops[a_load_idx]
                        self.instrs.append({"load": [("load", v_node_val_r[bl] + el, addr_temps_r[bl][el]),
                                                    ("load", v_node_val_r[bl] + el + 1, addr_temps_r[bl][el + 1])]})
                        a_load_idx += 1

                # After all rounds, do Hash and Index for groups A, B, C, D on their final values
                # (This was already done in the loop above for non-first rounds)

            # Store results for all 12 batches (12 vstores = 6 cycles)
            for i in range(0, NUM_BATCHES, 2):
                self.instrs.append({"store": [("vstore", idx_addr_r[i], v_idx_r[i]),
                                              ("vstore", val_addr_r[i], v_val_r[i])]})
                self.instrs.append({"store": [("vstore", idx_addr_r[i+1], v_idx_r[i+1]),
                                              ("vstore", val_addr_r[i+1], v_val_r[i+1])]})

            # Compute next iteration addresses if not last
            if not is_last_outer_iter:
                self.instrs.append({
                    "alu": [("+", idx_addr_r[i], self.scratch["inp_indices_p"], next_offsets[i]) for i in range(12)]
                })
                self.instrs.append({
                    "alu": [("+", val_addr_r[i], self.scratch["inp_values_p"], next_offsets[i]) for i in range(12)]
                })

        self.instrs.append({"flow": [("pause",)]})

    def build_kernel_staggered_pipeline(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Staggered multi-round pipeline: Multiple batch groups at different rounds.

        Key insight: If Group A is at round N step X, and Group B is at round N-1 step Y,
        they have no data dependency! We can execute them in parallel.

        Structure: 2 super-groups (SG0, SG1), each with 3 batches.
        - SG0 processes rounds 0, 2, 4, 6, ...
        - SG1 processes rounds 1, 3, 5, 7, ...
        When SG0 finishes round 0 and moves to round 2, SG1 starts round 1.
        This keeps both super-groups active with different data.
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

        # 2 super-groups of 3 batches each = 6 batches total per outer iter
        NUM_BATCHES = 6
        SG_SIZE = 3  # batches per super-group

        # Allocate registers for all batches
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_node = [self.alloc_scratch(f"v_node_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_cond = [self.alloc_scratch(f"v_cond_{i}", VLEN) for i in range(NUM_BATCHES)]
        addr_temps = [[self.alloc_scratch(f"addr_tmp_{i}_{j}") for j in range(VLEN)] for i in range(NUM_BATCHES)]
        idx_addr = [self.alloc_scratch(f"idx_addr_{i}") for i in range(NUM_BATCHES)]
        val_addr = [self.alloc_scratch(f"val_addr_{i}") for i in range(NUM_BATCHES)]

        v_one = self.alloc_vconst(1)

        hash_consts = []
        madd_stages = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_consts.append((self.alloc_vconst(val1), self.alloc_vconst(val3)))
            if op1 == '+' and op2 == '+' and op3 == '<<':
                mult = 1 + (1 << val3)
                madd_stages[hi] = (self.alloc_vconst(mult), self.alloc_vconst(val1))

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        n_vec_iters = batch_size // VLEN
        batch_offset_consts = [self.scratch_const(vi * VLEN) for vi in range(n_vec_iters)]

        self.add("flow", ("pause",))

        # Super-groups: SG0 = batches 0-2, SG1 = batches 3-5
        SG0 = list(range(SG_SIZE))
        SG1 = list(range(SG_SIZE, NUM_BATCHES))

        def emit_xor(batches):
            """Emit XOR for given batches"""
            self.instrs.append({"valu": [("^", v_val[b], v_val[b], v_node[b]) for b in batches]})

        def emit_hash_stage(batches, hi, extra_load=None, extra_alu=None):
            """Emit hash stage hi for batches, optionally with overlapped load/alu"""
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            c1, c3 = hash_consts[hi]

            if hi in madd_stages:
                mult_addr, const_addr = madd_stages[hi]
                instr = {"valu": [("multiply_add", v_val[b], v_val[b], mult_addr, const_addr) for b in batches]}
                if extra_load:
                    instr["load"] = extra_load
                if extra_alu:
                    instr["alu"] = extra_alu
                self.instrs.append(instr)
            else:
                instr = {"valu": [(op1, v_tmp1[b], v_val[b], c1) for b in batches] +
                                 [(op3, v_tmp2[b], v_val[b], c3) for b in batches]}
                if extra_load:
                    instr["load"] = extra_load
                if extra_alu:
                    instr["alu"] = extra_alu
                self.instrs.append(instr)
                self.instrs.append({"valu": [(op2, v_val[b], v_tmp1[b], v_tmp2[b]) for b in batches]})

        def emit_index(batches, extra_load=None):
            """Emit index computation for batches"""
            self.instrs.append({"valu": [("&", v_tmp1[b], v_val[b], v_one) for b in batches]})
            instr = {"valu": [("<<", v_tmp2[b], v_idx[b], v_one) for b in batches] +
                             [("+", v_node[b], v_tmp1[b], v_one) for b in batches]}
            if extra_load:
                instr["load"] = extra_load
            self.instrs.append(instr)
            self.instrs.append({"valu": [("+", v_idx[b], v_tmp2[b], v_node[b]) for b in batches]})
            self.instrs.append({"valu": [("<", v_cond[b], v_idx[b], v_n_nodes) for b in batches]})
            self.instrs.append({"valu": [("*", v_idx[b], v_idx[b], v_cond[b]) for b in batches]})

        def emit_gather_addrs(batches):
            """Emit ALU ops to compute gather addresses"""
            for b in batches:
                for e in range(0, VLEN, 4):
                    self.instrs.append({"alu": [("+", addr_temps[b][e+j], self.scratch["forest_values_p"], v_idx[b] + e + j) for j in range(min(4, VLEN-e))]})

        def emit_gather_loads(batches):
            """Emit loads for node values"""
            for b in batches:
                for e in range(0, VLEN, 2):
                    self.instrs.append({"load": [("load", v_node[b] + e, addr_temps[b][e]),
                                                 ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]})

        for vec_iter in range(0, n_vec_iters, NUM_BATCHES):
            offsets = [batch_offset_consts[min(vec_iter + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)]

            # Load initial idx/val for all batches
            self.instrs.append({
                "alu": [("+", idx_addr[i], self.scratch["inp_indices_p"], offsets[i]) for i in range(NUM_BATCHES)] +
                       [("+", val_addr[i], self.scratch["inp_values_p"], offsets[i]) for i in range(NUM_BATCHES)]
            })
            for i in range(NUM_BATCHES):
                self.instrs.append({"load": [("vload", v_idx[i], idx_addr[i]), ("vload", v_val[i], val_addr[i])]})

            # Process rounds with staggered pipeline
            # SG0 does even rounds first, SG1 follows with odd rounds

            for round_idx in range(rounds):
                is_first = (round_idx == 0)
                is_last = (round_idx == rounds - 1)

                if is_first:
                    # Round 0: All indices are 0, broadcast optimization
                    self.instrs.append({"load": [("load", v_node[0], self.scratch["forest_values_p"])]})
                    self.instrs.append({"valu": [("vbroadcast", v_node[b], v_node[0]) for b in range(NUM_BATCHES)]})
                    emit_xor(list(range(NUM_BATCHES)))

                    # Hash all 6 together
                    for hi in range(len(HASH_STAGES)):
                        emit_hash_stage(list(range(NUM_BATCHES)), hi)

                    # Index all 6 together
                    emit_index(list(range(NUM_BATCHES)))

                    # Prepare addresses for next round (SG0 only - it will start round 1)
                    emit_gather_addrs(SG0)
                    emit_gather_loads(SG0)
                else:
                    # Interleaved processing: SG0 does current, SG1 loads/prepares
                    # XOR SG0 with its loaded node values
                    emit_xor(SG0)

                    # Compute SG1's gather addresses while hashing SG0
                    addr_ops = [(b, e) for b in SG1 for e in range(VLEN)]
                    load_ops = [(b, e) for b in SG1 for e in range(0, VLEN, 2)]
                    addr_idx = 0
                    load_idx = 0

                    for hi in range(len(HASH_STAGES)):
                        op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                        c1, c3 = hash_consts[hi]

                        # Overlap ALU/load with SG0's hash
                        extra_alu = None
                        extra_load = None

                        if addr_idx < len(addr_ops):
                            alu_ops = []
                            for _ in range(min(12, len(addr_ops) - addr_idx)):
                                b, e = addr_ops[addr_idx]
                                alu_ops.append(("+", addr_temps[b][e], self.scratch["forest_values_p"], v_idx[b] + e))
                                addr_idx += 1
                            extra_alu = alu_ops
                        elif load_idx < len(load_ops):
                            b, e = load_ops[load_idx]
                            extra_load = [("load", v_node[b] + e, addr_temps[b][e]),
                                         ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]
                            load_idx += 1

                        emit_hash_stage(SG0, hi, extra_load=extra_load, extra_alu=extra_alu)

                    # Finish any remaining SG1 loads
                    while load_idx < len(load_ops):
                        b, e = load_ops[load_idx]
                        self.instrs.append({"load": [("load", v_node[b] + e, addr_temps[b][e]),
                                                    ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]})
                        load_idx += 1

                    # Index SG0 while XORing and starting hash on SG1
                    emit_xor(SG1)

                    # SG0 index
                    self.instrs.append({"valu": [("&", v_tmp1[b], v_val[b], v_one) for b in SG0]})
                    self.instrs.append({"valu": [("<<", v_tmp2[b], v_idx[b], v_one) for b in SG0] +
                                                [("+", v_node[b], v_tmp1[b], v_one) for b in SG0]})

                    # Start SG1 hash stage 0 while finishing SG0 index
                    self.instrs.append({"valu": [("+", v_idx[b], v_tmp2[b], v_node[b]) for b in SG0] +
                                                [("multiply_add", v_val[b], v_val[b], madd_stages[0][0], madd_stages[0][1]) for b in SG1]})

                    # Continue SG0 bounds check while SG1 hash stage 1
                    op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
                    c1_1, c3_1 = hash_consts[1]
                    self.instrs.append({"valu": [("<", v_cond[b], v_idx[b], v_n_nodes) for b in SG0] +
                                                [(op1_1, v_tmp1[b], v_val[b], c1_1) for b in SG1]})
                    self.instrs.append({"valu": [("*", v_idx[b], v_idx[b], v_cond[b]) for b in SG0] +
                                                [(op3_1, v_tmp2[b], v_val[b], c3_1) for b in SG1]})

                    # SG1 hash stage 1 op2, while preparing next round SG0 addresses
                    next_addr_ops = [(b, e) for b in SG0 for e in range(VLEN)] if not is_last else []
                    next_load_ops = [(b, e) for b in SG0 for e in range(0, VLEN, 2)] if not is_last else []
                    addr_idx = 0
                    load_idx = 0

                    instr = {"valu": [(op2_1, v_val[b], v_tmp1[b], v_tmp2[b]) for b in SG1]}
                    if addr_idx < len(next_addr_ops):
                        alu_ops = []
                        for _ in range(min(12, len(next_addr_ops) - addr_idx)):
                            b, e = next_addr_ops[addr_idx]
                            alu_ops.append(("+", addr_temps[b][e], self.scratch["forest_values_p"], v_idx[b] + e))
                            addr_idx += 1
                        instr["alu"] = alu_ops
                    self.instrs.append(instr)

                    # Complete SG1 hash stages 2-5 with next round SG0 address/load overlap
                    for hi in range(2, len(HASH_STAGES)):
                        extra_alu = None
                        extra_load = None

                        if addr_idx < len(next_addr_ops):
                            alu_ops = []
                            for _ in range(min(12, len(next_addr_ops) - addr_idx)):
                                b, e = next_addr_ops[addr_idx]
                                alu_ops.append(("+", addr_temps[b][e], self.scratch["forest_values_p"], v_idx[b] + e))
                                addr_idx += 1
                            extra_alu = alu_ops
                        elif load_idx < len(next_load_ops):
                            b, e = next_load_ops[load_idx]
                            extra_load = [("load", v_node[b] + e, addr_temps[b][e]),
                                         ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]
                            load_idx += 1

                        emit_hash_stage(SG1, hi, extra_load=extra_load, extra_alu=extra_alu)

                    # SG1 index with remaining SG0 loads
                    self.instrs.append({"valu": [("&", v_tmp1[b], v_val[b], v_one) for b in SG1]})

                    instr = {"valu": [("<<", v_tmp2[b], v_idx[b], v_one) for b in SG1] +
                                     [("+", v_node[b], v_tmp1[b], v_one) for b in SG1]}
                    if load_idx < len(next_load_ops):
                        b, e = next_load_ops[load_idx]
                        instr["load"] = [("load", v_node[b] + e, addr_temps[b][e]),
                                        ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("+", v_idx[b], v_tmp2[b], v_node[b]) for b in SG1]}
                    if load_idx < len(next_load_ops):
                        b, e = next_load_ops[load_idx]
                        instr["load"] = [("load", v_node[b] + e, addr_temps[b][e]),
                                        ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("<", v_cond[b], v_idx[b], v_n_nodes) for b in SG1]}
                    if load_idx < len(next_load_ops):
                        b, e = next_load_ops[load_idx]
                        instr["load"] = [("load", v_node[b] + e, addr_temps[b][e]),
                                        ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("*", v_idx[b], v_idx[b], v_cond[b]) for b in SG1]}
                    if load_idx < len(next_load_ops):
                        b, e = next_load_ops[load_idx]
                        instr["load"] = [("load", v_node[b] + e, addr_temps[b][e]),
                                        ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]
                        load_idx += 1
                    self.instrs.append(instr)

                    # Finish remaining SG0 loads
                    while load_idx < len(next_load_ops):
                        b, e = next_load_ops[load_idx]
                        self.instrs.append({"load": [("load", v_node[b] + e, addr_temps[b][e]),
                                                    ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]})
                        load_idx += 1

                    # Swap SG0 and SG1 for next round
                    SG0, SG1 = SG1, SG0

            # Store results
            for b in range(NUM_BATCHES):
                self.instrs.append({"store": [("vstore", idx_addr[b], v_idx[b]),
                                              ("vstore", val_addr[b], v_val[b])]})

        self.instrs.append({"flow": [("pause",)]})

    def build_kernel_true_stagger(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        True staggered pipeline: A and B run at DIFFERENT stages simultaneously.
        When A is at stage N, B is at stage N-1.
        This uses all 6 VALU slots: 3 from A's stage + 3 from B's stage.
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

        NUM_BATCHES = 6
        A = list(range(3))  # batches 0, 1, 2
        B = list(range(3, 6))  # batches 3, 4, 5

        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_node = [self.alloc_scratch(f"v_node_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(NUM_BATCHES)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(NUM_BATCHES)]
        addr_temps = [[self.alloc_scratch(f"addr_tmp_{i}_{j}") for j in range(VLEN)] for i in range(NUM_BATCHES)]
        idx_addr = [self.alloc_scratch(f"idx_addr_{i}") for i in range(NUM_BATCHES)]
        val_addr = [self.alloc_scratch(f"val_addr_{i}") for i in range(NUM_BATCHES)]
        v_cond = [self.alloc_scratch(f"v_cond_{i}", VLEN) for i in range(NUM_BATCHES)]

        v_one = self.alloc_vconst(1)

        hash_consts = []
        madd_stages = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_consts.append((self.alloc_vconst(val1), self.alloc_vconst(val3)))
            if op1 == '+' and op2 == '+' and op3 == '<<':
                mult = 1 + (1 << val3)
                madd_stages[hi] = (self.alloc_vconst(mult), self.alloc_vconst(val1))

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        n_vec_iters = batch_size // VLEN
        batch_offset_consts = [self.scratch_const(vi * VLEN) for vi in range(n_vec_iters)]

        self.add("flow", ("pause",))

        # Hash stage info for staggering
        # Stages 0,2,4: multiply_add (1 cycle, 3 ops for 3 batches)
        # Stages 1,3,5: 2-cycle (cycle 1: 6 ops for op1+op3, cycle 2: 3 ops for op2)

        def get_stage_ops(batches, hi, phase=0):
            """Get VALU ops for a hash stage.
            For multiply_add: single phase (phase=0)
            For 2-cycle: phase=0 is op1+op3, phase=1 is op2
            Returns list of (op, dest, src1, src2) or (op, dest, src1, src2, src3) for madd
            """
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            c1, c3 = hash_consts[hi]

            if hi in madd_stages:
                mult_addr, const_addr = madd_stages[hi]
                return [("multiply_add", v_val[b], v_val[b], mult_addr, const_addr) for b in batches]
            else:
                if phase == 0:
                    # op1 and op3 together
                    return ([(op1, v_tmp1[b], v_val[b], c1) for b in batches] +
                            [(op3, v_tmp2[b], v_val[b], c3) for b in batches])
                else:
                    # op2
                    return [(op2, v_val[b], v_tmp1[b], v_tmp2[b]) for b in batches]

        def is_madd(hi):
            return hi in madd_stages

        for vec_iter in range(0, n_vec_iters, NUM_BATCHES):
            offsets = [batch_offset_consts[min(vec_iter + i, n_vec_iters - 1)] for i in range(NUM_BATCHES)]

            # Load initial idx/val
            self.instrs.append({
                "alu": [("+", idx_addr[i], self.scratch["inp_indices_p"], offsets[i]) for i in range(NUM_BATCHES)] +
                       [("+", val_addr[i], self.scratch["inp_values_p"], offsets[i]) for i in range(NUM_BATCHES)]
            })
            for i in range(NUM_BATCHES):
                self.instrs.append({"load": [("vload", v_idx[i], idx_addr[i]), ("vload", v_val[i], val_addr[i])]})

            for round_idx in range(rounds):
                is_first = (round_idx == 0)
                is_last = (round_idx == rounds - 1)

                if is_first:
                    # Round 0: All indices are 0
                    self.instrs.append({"load": [("load", v_node[0], self.scratch["forest_values_p"])]})
                    self.instrs.append({"valu": [("vbroadcast", v_node[b], v_node[0]) for b in range(NUM_BATCHES)]})
                    # XOR all
                    self.instrs.append({"valu": [("^", v_val[b], v_val[b], v_node[b]) for b in range(NUM_BATCHES)]})

                    # Hash all 6 together - BUT respect 6 VALU slot limit
                    for hi in range(len(HASH_STAGES)):
                        op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                        c1, c3 = hash_consts[hi]
                        if is_madd(hi):
                            # multiply_add: 6 ops for 6 batches = 1 cycle
                            mult_addr, const_addr = madd_stages[hi]
                            self.instrs.append({"valu": [("multiply_add", v_val[b], v_val[b], mult_addr, const_addr) for b in range(NUM_BATCHES)]})
                        else:
                            # 2-cycle stage: need 3 cycles for 6 batches
                            # Cycle 1: op1 for all 6
                            self.instrs.append({"valu": [(op1, v_tmp1[b], v_val[b], c1) for b in range(NUM_BATCHES)]})
                            # Cycle 2: op3 for all 6
                            self.instrs.append({"valu": [(op3, v_tmp2[b], v_val[b], c3) for b in range(NUM_BATCHES)]})
                            # Cycle 3: op2 for all 6
                            self.instrs.append({"valu": [(op2, v_val[b], v_tmp1[b], v_tmp2[b]) for b in range(NUM_BATCHES)]})

                    # Index all 6
                    self.instrs.append({"valu": [("&", v_tmp1[b], v_val[b], v_one) for b in range(NUM_BATCHES)]})
                    self.instrs.append({"valu": [("<<", v_tmp2[b], v_idx[b], v_one) for b in range(NUM_BATCHES)]})
                    self.instrs.append({"valu": [("+", v_node[b], v_tmp1[b], v_one) for b in range(NUM_BATCHES)]})
                    self.instrs.append({"valu": [("+", v_idx[b], v_tmp2[b], v_node[b]) for b in range(NUM_BATCHES)]})
                    self.instrs.append({"valu": [("<", v_cond[b], v_idx[b], v_n_nodes) for b in range(NUM_BATCHES)]})
                    self.instrs.append({"valu": [("*", v_idx[b], v_idx[b], v_cond[b]) for b in range(NUM_BATCHES)]})

                    # Compute addresses and load node values for A (round 1)
                    for b in A:
                        for e in range(0, VLEN, 4):
                            self.instrs.append({"alu": [("+", addr_temps[b][e+j], self.scratch["forest_values_p"], v_idx[b] + e + j) for j in range(min(4, VLEN-e))]})
                    for b in A:
                        for e in range(0, VLEN, 2):
                            self.instrs.append({"load": [("load", v_node[b] + e, addr_temps[b][e]),
                                                        ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]})

                else:
                    # Non-zero rounds: TRUE STAGGERED PIPELINE
                    # A starts at stage 0, B is one stage behind (still loading or at previous stage)

                    # Compute B addresses while A does XOR
                    b_addr_ops = [(b, e) for b in B for e in range(VLEN)]
                    b_load_ops = [(b, e) for b in B for e in range(0, VLEN, 2)]
                    b_addr_idx = 0
                    b_load_idx = 0

                    # XOR A + compute first B addresses
                    alu_ops = []
                    for _ in range(min(12, len(b_addr_ops))):
                        b, e = b_addr_ops[b_addr_idx]
                        alu_ops.append(("+", addr_temps[b][e], self.scratch["forest_values_p"], v_idx[b] + e))
                        b_addr_idx += 1
                    self.instrs.append({
                        "valu": [("^", v_val[b], v_val[b], v_node[b]) for b in A],
                        "alu": alu_ops
                    })

                    # Finish B addresses
                    while b_addr_idx < len(b_addr_ops):
                        alu_ops = []
                        for _ in range(min(12, len(b_addr_ops) - b_addr_idx)):
                            b, e = b_addr_ops[b_addr_idx]
                            alu_ops.append(("+", addr_temps[b][e], self.scratch["forest_values_p"], v_idx[b] + e))
                            b_addr_idx += 1
                        self.instrs.append({"alu": alu_ops})

                    # TRUE STAGGER: A at stage N, B loads/trails
                    # We interleave A's hash stages with B's loads
                    # A stage 0 (madd) + B load
                    # A stage 1 phase 0 + B load
                    # A stage 1 phase 1 + B load
                    # ... etc until B finishes loading
                    # Then A stage X + B XOR
                    # Then A stage X+1 + B stage 0
                    # etc.

                    # Track where we are in A's pipeline
                    # A has 9 VALU cycles: 3 madd + 6 for 2-cycle stages
                    # B needs 12 load cycles

                    # Build A's hash stages as a sequence of 3-op VALU instructions
                    a_hash_ops = []
                    for hi in range(len(HASH_STAGES)):
                        if is_madd(hi):
                            a_hash_ops.append(get_stage_ops(A, hi))
                        else:
                            a_hash_ops.append(get_stage_ops(A, hi, 0))  # op1+op3 = 6 ops for 3 batches
                            a_hash_ops.append(get_stage_ops(A, hi, 1))  # op2 = 3 ops

                    # Emit A's hash with B's loads overlapped
                    a_idx = 0
                    while a_idx < len(a_hash_ops) and b_load_idx < len(b_load_ops):
                        valu_ops = a_hash_ops[a_idx]
                        b, e = b_load_ops[b_load_idx]
                        self.instrs.append({
                            "valu": valu_ops,
                            "load": [("load", v_node[b] + e, addr_temps[b][e]),
                                    ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]
                        })
                        a_idx += 1
                        b_load_idx += 1

                    # Finish remaining A hash ops
                    while a_idx < len(a_hash_ops):
                        self.instrs.append({"valu": a_hash_ops[a_idx]})
                        a_idx += 1

                    # Finish remaining B loads
                    while b_load_idx < len(b_load_ops):
                        b, e = b_load_ops[b_load_idx]
                        self.instrs.append({"load": [("load", v_node[b] + e, addr_temps[b][e]),
                                                    ("load", v_node[b] + e + 1, addr_temps[b][e + 1])]})
                        b_load_idx += 1

                    # Now B has its node values loaded, A has finished hash
                    # XOR B (3 ops) + A index step 1 (3 ops) = 6 ops
                    self.instrs.append({
                        "valu": [("^", v_val[b], v_val[b], v_node[b]) for b in B] +
                                [("&", v_tmp1[b], v_val[b], v_one) for b in A]
                    })

                    # A index step 2-3 + B hash stage 0
                    self.instrs.append({
                        "valu": [("<<", v_tmp2[b], v_idx[b], v_one) for b in A] +
                                [("+", v_node[b], v_tmp1[b], v_one) for b in A]
                    })

                    # Now TRUE STAGGER: A index step 4 + B stage 0
                    # B stage 0 is multiply_add (3 ops)
                    # A step 4 is + (3 ops)
                    self.instrs.append({
                        "valu": [("+", v_idx[b], v_tmp2[b], v_node[b]) for b in A] +
                                [("multiply_add", v_val[b], v_val[b], madd_stages[0][0], madd_stages[0][1]) for b in B]
                    })

                    # A bounds check + B stage 1 phase 0
                    op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
                    c1_1, c3_1 = hash_consts[1]
                    self.instrs.append({
                        "valu": [("<", v_cond[b], v_idx[b], v_n_nodes) for b in A] +
                                [(op1_1, v_tmp1[b], v_val[b], c1_1) for b in B]
                    })
                    self.instrs.append({
                        "valu": [("*", v_idx[b], v_idx[b], v_cond[b]) for b in A] +
                                [(op3_1, v_tmp2[b], v_val[b], c3_1) for b in B]
                    })

                    # B stage 1 phase 1 - only 3 ops, need to overlap with something
                    # Start loading next round's A values
                    next_a_addr_ops = [(b, e) for b in A for e in range(VLEN)] if not is_last else []
                    next_a_load_ops = [(b, e) for b in A for e in range(0, VLEN, 2)] if not is_last else []
                    next_addr_idx = 0
                    next_load_idx = 0

                    # B stage 1 op2 + start next A addresses
                    instr = {"valu": [(op2_1, v_val[b], v_tmp1[b], v_tmp2[b]) for b in B]}
                    if next_addr_idx < len(next_a_addr_ops):
                        alu_ops = []
                        for _ in range(min(12, len(next_a_addr_ops) - next_addr_idx)):
                            nb, ne = next_a_addr_ops[next_addr_idx]
                            alu_ops.append(("+", addr_temps[nb][ne], self.scratch["forest_values_p"], v_idx[nb] + ne))
                            next_addr_idx += 1
                        instr["alu"] = alu_ops
                    self.instrs.append(instr)

                    # B stages 2-5 + next A addr/loads
                    for hi in range(2, len(HASH_STAGES)):
                        if is_madd(hi):
                            instr = {"valu": get_stage_ops(B, hi)}
                            if next_addr_idx < len(next_a_addr_ops):
                                alu_ops = []
                                for _ in range(min(12, len(next_a_addr_ops) - next_addr_idx)):
                                    nb, ne = next_a_addr_ops[next_addr_idx]
                                    alu_ops.append(("+", addr_temps[nb][ne], self.scratch["forest_values_p"], v_idx[nb] + ne))
                                    next_addr_idx += 1
                                instr["alu"] = alu_ops
                            elif next_load_idx < len(next_a_load_ops):
                                nb, ne = next_a_load_ops[next_load_idx]
                                instr["load"] = [("load", v_node[nb] + ne, addr_temps[nb][ne]),
                                                ("load", v_node[nb] + ne + 1, addr_temps[nb][ne + 1])]
                                next_load_idx += 1
                            self.instrs.append(instr)
                        else:
                            # 2-cycle stage
                            instr = {"valu": get_stage_ops(B, hi, 0)}
                            if next_addr_idx < len(next_a_addr_ops):
                                alu_ops = []
                                for _ in range(min(12, len(next_a_addr_ops) - next_addr_idx)):
                                    nb, ne = next_a_addr_ops[next_addr_idx]
                                    alu_ops.append(("+", addr_temps[nb][ne], self.scratch["forest_values_p"], v_idx[nb] + ne))
                                    next_addr_idx += 1
                                instr["alu"] = alu_ops
                            elif next_load_idx < len(next_a_load_ops):
                                nb, ne = next_a_load_ops[next_load_idx]
                                instr["load"] = [("load", v_node[nb] + ne, addr_temps[nb][ne]),
                                                ("load", v_node[nb] + ne + 1, addr_temps[nb][ne + 1])]
                                next_load_idx += 1
                            self.instrs.append(instr)

                            instr = {"valu": get_stage_ops(B, hi, 1)}
                            if next_addr_idx < len(next_a_addr_ops):
                                alu_ops = []
                                for _ in range(min(12, len(next_a_addr_ops) - next_addr_idx)):
                                    nb, ne = next_a_addr_ops[next_addr_idx]
                                    alu_ops.append(("+", addr_temps[nb][ne], self.scratch["forest_values_p"], v_idx[nb] + ne))
                                    next_addr_idx += 1
                                instr["alu"] = alu_ops
                            elif next_load_idx < len(next_a_load_ops):
                                nb, ne = next_a_load_ops[next_load_idx]
                                instr["load"] = [("load", v_node[nb] + ne, addr_temps[nb][ne]),
                                                ("load", v_node[nb] + ne + 1, addr_temps[nb][ne + 1])]
                                next_load_idx += 1
                            self.instrs.append(instr)

                    # B index + remaining next A loads
                    self.instrs.append({"valu": [("&", v_tmp1[b], v_val[b], v_one) for b in B]})

                    instr = {"valu": [("<<", v_tmp2[b], v_idx[b], v_one) for b in B] +
                                     [("+", v_node[b], v_tmp1[b], v_one) for b in B]}
                    if next_load_idx < len(next_a_load_ops):
                        nb, ne = next_a_load_ops[next_load_idx]
                        instr["load"] = [("load", v_node[nb] + ne, addr_temps[nb][ne]),
                                        ("load", v_node[nb] + ne + 1, addr_temps[nb][ne + 1])]
                        next_load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("+", v_idx[b], v_tmp2[b], v_node[b]) for b in B]}
                    if next_load_idx < len(next_a_load_ops):
                        nb, ne = next_a_load_ops[next_load_idx]
                        instr["load"] = [("load", v_node[nb] + ne, addr_temps[nb][ne]),
                                        ("load", v_node[nb] + ne + 1, addr_temps[nb][ne + 1])]
                        next_load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("<", v_cond[b], v_idx[b], v_n_nodes) for b in B]}
                    if next_load_idx < len(next_a_load_ops):
                        nb, ne = next_a_load_ops[next_load_idx]
                        instr["load"] = [("load", v_node[nb] + ne, addr_temps[nb][ne]),
                                        ("load", v_node[nb] + ne + 1, addr_temps[nb][ne + 1])]
                        next_load_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [("*", v_idx[b], v_idx[b], v_cond[b]) for b in B]}
                    if next_load_idx < len(next_a_load_ops):
                        nb, ne = next_a_load_ops[next_load_idx]
                        instr["load"] = [("load", v_node[nb] + ne, addr_temps[nb][ne]),
                                        ("load", v_node[nb] + ne + 1, addr_temps[nb][ne + 1])]
                        next_load_idx += 1
                    self.instrs.append(instr)

                    # Finish remaining next A loads
                    while next_load_idx < len(next_a_load_ops):
                        nb, ne = next_a_load_ops[next_load_idx]
                        self.instrs.append({"load": [("load", v_node[nb] + ne, addr_temps[nb][ne]),
                                                    ("load", v_node[nb] + ne + 1, addr_temps[nb][ne + 1])]})
                        next_load_idx += 1

                    # Swap A and B for next round
                    A, B = B, A

            # Store results
            for b in range(NUM_BATCHES):
                self.instrs.append({"store": [("vstore", idx_addr[b], v_idx[b]),
                                              ("vstore", val_addr[b], v_val[b])]})

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
    # kb.build_kernel_12batch(forest.height, len(forest.values), len(inp.indices), rounds)

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

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
