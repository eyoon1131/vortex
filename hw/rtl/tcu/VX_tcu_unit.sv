// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

module VX_tcu_unit import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output tcu_perf_t       tcu_perf,
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // Bank-parallel LMEM read port
    VX_mem_bus_if.master     tcu_lmem_if,
`endif

`ifdef TCU_META_ENABLE
    // TCU_LD memory client connection to VX_lsu_scheduler at VX_core.
    VX_lsu_sched_if.master  tcu_mem_if,
`endif

`ifdef TCU_TMEM_ENABLE
    input wire [`VX_CFG_NUM_WARPS-1:0][NW_WIDTH-1:0] cta_rank_table,
`endif

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = `VX_CFG_NUM_TCU_BLOCKS;
    localparam NUM_LANES  = `VX_CFG_NUM_TCU_LANES;

    `STATIC_ASSERT (BLOCK_SIZE == `VX_CFG_ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (NUM_LANES == `VX_CFG_NUM_THREADS, ("must be full warp execution"));
    `SCOPE_IO_SWITCH (BLOCK_SIZE);

    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (tcu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    // -----------------------------------------------------------------------
    // Split each per_block_execute_if between two consumers:
    //   - VX_tcu_agu: handles INST_TCU_LD (warp-level memory load).
    //   - VX_tcu_core: handles every MMA op_type.
    // The ready signal is muxed by op_type so only one consumer drives at a time.
    // -----------------------------------------------------------------------
    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) core_execute_if[BLOCK_SIZE]();

    VX_result_if #(
        .data_t (tcu_result_t)
    ) core_result_if[BLOCK_SIZE]();

`ifdef TCU_META_ENABLE
    wire [BLOCK_SIZE-1:0]    agu_ld_valid;
    wire [BLOCK_SIZE-1:0]    agu_ld_ready;
    tcu_execute_t            agu_ld_data [BLOCK_SIZE];

    wire [BLOCK_SIZE-1:0]    agu_result_valid;
    tcu_result_t             agu_result_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0]    agu_result_ready;
`endif

`ifdef TCU_TMEM_ENABLE
    // TMEM_ALLOC/DEALLOC/ST/LD bypass tcu_core entirely (no FEDP compute)
    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) tmem_execute_if[BLOCK_SIZE]();
`endif

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_split
    `ifdef TCU_TMEM_ENABLE
        wire is_tmem_mgmt = (per_block_execute_if[bi].data.op_type == INST_TCU_TMEM_ALLOC)
                         || (per_block_execute_if[bi].data.op_type == INST_TCU_TMEM_DEALLOC)
                         || (per_block_execute_if[bi].data.op_type == INST_TCU_TMEM_ST)
                         || (per_block_execute_if[bi].data.op_type == INST_TCU_TMEM_LD);
        assign tmem_execute_if[bi].valid = per_block_execute_if[bi].valid && is_tmem_mgmt;
        assign tmem_execute_if[bi].data  = per_block_execute_if[bi].data;
    `endif
    `ifdef TCU_META_ENABLE
        wire is_tcu_ld = (per_block_execute_if[bi].data.op_type == INST_TCU_LD);

        // To AGU when TCU_LD
        assign agu_ld_valid[bi]    = per_block_execute_if[bi].valid && is_tcu_ld;
        assign agu_ld_data[bi]     = per_block_execute_if[bi].data;

        // To tcu_core when NOT TCU_LD (and not TMEM management, if enabled)
        assign core_execute_if[bi].valid = per_block_execute_if[bi].valid && !is_tcu_ld
        `ifdef TCU_TMEM_ENABLE
            && !is_tmem_mgmt
        `endif
            ;
        assign core_execute_if[bi].data  = per_block_execute_if[bi].data;

        // Parent .ready: route to AGU on TCU_LD, TMEM bypass on TMEM
        // management, otherwise tcu_core
        assign per_block_execute_if[bi].ready = is_tcu_ld
            ? agu_ld_ready[bi]
        `ifdef TCU_TMEM_ENABLE
            : is_tmem_mgmt
            ? tmem_execute_if[bi].ready
        `endif
            : core_execute_if[bi].ready;
    `else
        // No sparse: pass-through to tcu_core (or TMEM bypass, if enabled)
        assign core_execute_if[bi].valid = per_block_execute_if[bi].valid
        `ifdef TCU_TMEM_ENABLE
            && !is_tmem_mgmt
        `endif
            ;
        assign core_execute_if[bi].data  = per_block_execute_if[bi].data;
        assign per_block_execute_if[bi].ready =
        `ifdef TCU_TMEM_ENABLE
            is_tmem_mgmt ? tmem_execute_if[bi].ready :
        `endif
            core_execute_if[bi].ready;
    `endif
    end

`ifdef TCU_TMEM_ENABLE
    VX_result_if #(
        .data_t (tcu_result_t)
    ) tmem_result_if[BLOCK_SIZE]();
`endif

    // -----------------------------------------------------------------------
    // Result_if merge: AGU / TMEM-bypass / tcu_core results are mutually
    // exclusive in time per block; OR-mux with priority arbiter (AGU wins,
    // then TMEM bypass: both are rare single-cycle ops vs. tcu_core's
    // multi-cycle FEDP compute).
    // -----------------------------------------------------------------------
`ifdef TCU_META_ENABLE
    // AGU wins same-cycle conflicts; tcu_core stalls (ready=0) and retries next cycle.
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result_merge
    `ifdef TCU_TMEM_ENABLE
        assign per_block_result_if[bi].valid = agu_result_valid[bi] || tmem_result_if[bi].valid || core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = agu_result_valid[bi]  ? agu_result_data[bi]
                                              : tmem_result_if[bi].valid ? tmem_result_if[bi].data
                                              : core_result_if[bi].data;
        assign agu_result_ready[bi]      = per_block_result_if[bi].ready;
        assign tmem_result_if[bi].ready  = per_block_result_if[bi].ready && !agu_result_valid[bi];
        assign core_result_if[bi].ready  = per_block_result_if[bi].ready && !agu_result_valid[bi] && !tmem_result_if[bi].valid;
    `else
        assign per_block_result_if[bi].valid = agu_result_valid[bi] || core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = agu_result_valid[bi]
            ? agu_result_data[bi]
            : core_result_if[bi].data;
        assign agu_result_ready[bi]    = per_block_result_if[bi].ready;
        assign core_result_if[bi].ready = per_block_result_if[bi].ready && !agu_result_valid[bi];
    `endif
    end
`else
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_result_passthru
    `ifdef TCU_TMEM_ENABLE
        assign per_block_result_if[bi].valid = tmem_result_if[bi].valid || core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = tmem_result_if[bi].valid ? tmem_result_if[bi].data : core_result_if[bi].data;
        assign tmem_result_if[bi].ready = per_block_result_if[bi].ready;
        assign core_result_if[bi].ready = per_block_result_if[bi].ready && !tmem_result_if[bi].valid;
    `else
        assign per_block_result_if[bi].valid = core_result_if[bi].valid;
        assign per_block_result_if[bi].data  = core_result_if[bi].data;
        assign core_result_if[bi].ready      = per_block_result_if[bi].ready;
    `endif
    end
`endif

    // -----------------------------------------------------------------------
    // WGMMA feature (orchestrator): VX_tcu_tbuf + VX_tcu_lockstep + perf.
    // -----------------------------------------------------------------------

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [BLOCK_SIZE-1:0]                                          exec_valid_w;
    wire [BLOCK_SIZE-1:0]                                          exec_ready_w;
    tcu_execute_t                                                  exec_data_w [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_obs
        assign exec_valid_w[bi]    = core_execute_if[bi].valid;
        assign exec_ready_w[bi]    = core_execute_if[bi].ready;
        assign exec_data_w[bi]     = core_execute_if[bi].data;
    end

    wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data;
    wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] tbuf_rs2_data;
    wire [BLOCK_SIZE-1:0]                                         tbuf_ready_eff;

    VX_tcu_wgmma #(
        .INSTANCE_ID (`SFORMATF(("%s-wgmma", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) wgmma (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tcu_perf       (tcu_perf),
    `endif
        .exec_valid     (exec_valid_w),
        .exec_ready     (exec_ready_w),
        .exec_data      (exec_data_w),
        .tcu_lmem_if    (tcu_lmem_if),
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data),
        .tbuf_ready_eff (tbuf_ready_eff)
    );

`else // !VX_CFG_TCU_WGMMA_ENABLE

`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_stalls     = '0;
    assign tcu_perf.tbuf_cache_hits = '0;
    assign tcu_perf.lmem_reads      = '0;
    assign tcu_perf.wgmma_instrs    = '0;
    assign tcu_perf.wgmma_stalls    = '0;
`endif

`endif // VX_CFG_TCU_WGMMA_ENABLE

    // -----------------------------------------------------------------------
    // VX_tcu_agu — warp-level AGU for TCU_LD instructions.
    // Drives meta_wr signals broadcast to every tcu_core so wmma_sp on
    // any block sees the loaded metadata.
    // -----------------------------------------------------------------------
`ifdef TCU_META_ENABLE
    wire                                              agu_meta_wr_en;
    wire [NW_WIDTH-1:0]                               agu_meta_wr_wid;
    wire [4:0]                                        agu_meta_wr_idx;
    wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0]        agu_meta_wr_data;

    VX_tcu_agu #(
        .INSTANCE_ID (`SFORMATF(("%s-agu", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE),
        .NUM_LANES   (NUM_LANES)
    ) agu (
        .clk                (clk),
        .reset              (reset),
        .per_block_ld_valid (agu_ld_valid),
        .per_block_ld_data  (agu_ld_data),
        .per_block_ld_ready (agu_ld_ready),
        .client_if          (tcu_mem_if),
        .meta_wr_en         (agu_meta_wr_en),
        .meta_wr_wid        (agu_meta_wr_wid),
        .meta_wr_idx        (agu_meta_wr_idx),
        .meta_wr_data       (agu_meta_wr_data),
        .result_valid       (agu_result_valid),
        .result_data        (agu_result_data),
        .result_ready       (agu_result_ready)
    );
`endif

`ifdef TCU_TMEM_ENABLE
    // -----------------------------------------------------------------------
    // TMEM storage and management.
    //
    // PLACEHOLDER: this is tensor_mem's original single-tenant allocator
    // (one global allocation, no per-CTA handle) ported onto the current
    // TCU structure. NOT the real multi-tenant allocator (CTA-scoped
    // free-list + CAM, matching the SimX side). Deliberately simple to
    // verify the UMMA FEDP compute path + RAW-hazard interlock.
    // Redo before any multi-CTA-concurrent UMMA testing:
    //   - ALLOC/DEALLOC/ST all share one always_ff below with NO real
    //     arbitration across blocks — if more than one block requests one
    //     of these in the same cycle, whichever has the highest block_idx
    //     wins (last write in the loop), the rest are silently dropped
    //     rather than retried.
    //   - handle is always implicitly 0 (tmem_ncols tracks one allocation,
    //     no per-CTA base column)
    // -----------------------------------------------------------------------

    logic [31:0] tmem_data [TCU_TMEM_LANES][TCU_TMEM_COLS];
    logic        tmem_allocated;
    logic [7:0]  tmem_ncols;
    `UNUSED_VAR (tmem_allocated)
    `UNUSED_VAR (tmem_ncols)

    // Per-block outputs from tcu_core (UMMA compute writeback)
    wire                                    tmem_wr_en        [BLOCK_SIZE];
    wire [TCU_TMEM_LANE_BITS-1:0]           tmem_wr_lane_base [BLOCK_SIZE];
    wire [TCU_TMEM_COL_BITS-1:0]            tmem_wr_col_base  [BLOCK_SIZE];
    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] tmem_wr_data      [BLOCK_SIZE];

    wire        [BLOCK_SIZE-1:0] tmem_mgmt_valid;
    tcu_execute_t                tmem_mgmt_data [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_extract
        assign tmem_mgmt_valid[bi] = tmem_execute_if[bi].valid;
        assign tmem_mgmt_data[bi]  = tmem_execute_if[bi].data;
    end

    // TMEM_LD is combinational, never conflicts across blocks
    wire [31:0] tmem_ld_rd_data [BLOCK_SIZE][`VX_CFG_NUM_THREADS];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_ld
        wire is_tmem_ld = tmem_execute_if[bi].valid
                        && (tmem_execute_if[bi].data.op_type == INST_TCU_TMEM_LD);
        wire [TCU_TMEM_LANE_BITS-1:0] ld_lane_base =
            TCU_TMEM_LANE_BITS'(tmem_execute_if[bi].data.rs1_data[0][31:16]);
        wire [TCU_TMEM_COL_BITS-1:0] ld_col =
            TCU_TMEM_COL_BITS'(tmem_execute_if[bi].data.rs1_data[0][15:0]);
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_tmem_ld_t
            assign tmem_ld_rd_data[bi][t] = is_tmem_ld
                ? tmem_data[TCU_TMEM_LANE_BITS'(ld_lane_base) + TCU_TMEM_LANE_BITS'(t)][ld_col]
                : '0;
        end
    end

    // ALLOC/DEALLOC/ST + UMMA writeback: combined into one process so
    // there's a single driver for tmem_data/tmem_allocated/tmem_ncols
    always_ff @(posedge clk) begin
        if (reset) begin
            tmem_allocated <= 1'b0;
            tmem_ncols     <= '0;
        end else begin
            // UMMA compute writeback — never collides across blocks in
            // practice (lane ranges are cta_rank-derived and disjoint
            // within an active warpgroup)
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                if (tmem_wr_en[b]) begin
                    for (int i = 0; i < TCU_TC_M; ++i) begin
                        for (int j = 0; j < TCU_TC_N; ++j) begin
                            tmem_data[TCU_TMEM_LANE_BITS'(tmem_wr_lane_base[b]) + TCU_TMEM_LANE_BITS'(i)]
                                     [TCU_TMEM_COL_BITS'(tmem_wr_col_base[b])  + TCU_TMEM_COL_BITS'(j)]
                                <= tmem_wr_data[b][i][j];
                        end
                    end
                end
            end

            // ALLOC/DEALLOC/ST: single global allocation, no arbitration
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                if (tmem_mgmt_valid[b]) begin
                    case (tmem_mgmt_data[b].op_type)
                        INST_TCU_TMEM_ALLOC: begin
                            tmem_allocated <= 1'b1;
                            tmem_ncols     <= tmem_mgmt_data[b].rs1_data[0][7:0];
                            for (int l = 0; l < TCU_TMEM_LANES; ++l)
                                for (int c = 0; c < TCU_TMEM_COLS; ++c)
                                    tmem_data[l][c] <= '0;
                        end
                        INST_TCU_TMEM_DEALLOC: begin
                            tmem_allocated <= 1'b0;
                            tmem_ncols     <= '0;
                        end
                        INST_TCU_TMEM_ST: begin
                            for (int t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                                if (tmem_mgmt_data[b].header.tmask[t]) begin
                                    automatic logic [TCU_TMEM_LANE_BITS-1:0] st_lane_base =
                                        TCU_TMEM_LANE_BITS'(tmem_mgmt_data[b].rs1_data[0][31:16]);
                                    automatic logic [TCU_TMEM_COL_BITS-1:0] st_col =
                                        TCU_TMEM_COL_BITS'(tmem_mgmt_data[b].rs1_data[0][15:0]);
                                    tmem_data[TCU_TMEM_LANE_BITS'(st_lane_base) + TCU_TMEM_LANE_BITS'(t)][st_col]
                                        <= tmem_mgmt_data[b].rs2_data[t][31:0];
                                end
                            end
                        end
                        default: ;
                    endcase
                end
            end
        end
    end

    // Bypass execute<->result handshake: same-cycle response, no queueing
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_result
        assign tmem_execute_if[bi].ready = tmem_result_if[bi].ready;
        assign tmem_result_if[bi].valid  = tmem_execute_if[bi].valid;
        assign tmem_result_if[bi].data.header = tmem_execute_if[bi].data.header;
        wire is_tmem_ld_r = tmem_execute_if[bi].data.op_type == INST_TCU_TMEM_LD;
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_tmem_result_t
            if (`VX_CFG_XLEN > 32) begin : g_nanbox
                assign tmem_result_if[bi].data.data[t] = is_tmem_ld_r
                    ? {32'hffffffff, tmem_ld_rd_data[bi][t]} : '0;
            end else begin : g_pass
                assign tmem_result_if[bi].data.data[t] = is_tmem_ld_r
                    ? `VX_CFG_XLEN'(tmem_ld_rd_data[bi][t]) : '0;
            end
        end
    end

`ifdef PERF_ENABLE
    // Stub for now
    assign tcu_perf.umma_instrs = '0;
    assign tcu_perf.tmem_reads  = '0;
    assign tcu_perf.tmem_writes = '0;
`endif
`endif // TCU_TMEM_ENABLE

    // -----------------------------------------------------------------------
    // TCU core instances
    // -----------------------------------------------------------------------

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
    `ifdef TCU_TMEM_ENABLE
        // This block's currently-executing warp's CTA-local rank
        wire [NW_WIDTH-1:0] block_cta_rank =
            cta_rank_table[core_execute_if[block_idx].data.header.wid];
    `endif
        VX_tcu_core #(
            .INSTANCE_ID (`SFORMATF(("%s-fused%0d", INSTANCE_ID, block_idx)))
        ) tcu_core (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
            .tbuf_rs1_data (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data (tbuf_rs2_data[block_idx]),
            .tbuf_ready    (tbuf_ready_eff[block_idx]),
        `endif
        `ifdef TCU_TMEM_ENABLE
            .cta_rank         (block_cta_rank),
            .tmem_data        (tmem_data),
            .tmem_wr_en       (tmem_wr_en[block_idx]),
            .tmem_wr_lane_base(tmem_wr_lane_base[block_idx]),
            .tmem_wr_col_base (tmem_wr_col_base[block_idx]),
            .tmem_wr_data     (tmem_wr_data[block_idx]),
        `endif
        `ifdef TCU_META_ENABLE
            .ext_meta_wr_en   (agu_meta_wr_en),
            .ext_meta_wr_wid  (agu_meta_wr_wid),
            .ext_meta_wr_idx  (agu_meta_wr_idx),
            .ext_meta_wr_data (agu_meta_wr_data),
        `endif
            .execute_if (core_execute_if[block_idx]),
            .result_if  (core_result_if[block_idx])
        );
    end

    // -----------------------------------------------------------------------
    // Lane gather
    // -----------------------------------------------------------------------

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
