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

`ifdef TCU_WGMMA_ENABLE
    // Bank-parallel LMEM read port
    VX_tcu_lmem_if.master   tcu_lmem_if,
`endif

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = `NUM_TCU_BLOCKS;
    localparam NUM_LANES  = `NUM_TCU_LANES;

    `STATIC_ASSERT(BLOCK_SIZE == 1, ("TMEM only supports BLOCK_SIZE=1"))
    `STATIC_ASSERT (BLOCK_SIZE == `ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (NUM_LANES == `NUM_THREADS, ("must be full warp execution"));
    `SCOPE_IO_SWITCH (BLOCK_SIZE);

`ifdef TCU_TMEM_ENABLE
    `STATIC_ASSERT(TCU_TMEM_LANES <= 128, ("TMEM lanes exceed cap"))
`endif

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
    // WGMMA tile buffers + LMEM port arbitration
    // -----------------------------------------------------------------------

`ifdef TCU_WGMMA_ENABLE
    localparam BANK_ADDR_WIDTH = `LMEM_LOG_SIZE - $clog2(`XLEN / 8) - $clog2(`LMEM_NUM_BANKS);

    // Per-block tile buffer outputs
    wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] tbuf_rs1_data  [BLOCK_SIZE];
    wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] tbuf_rs2_data [BLOCK_SIZE];
`ifdef TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] tbuf_sp_meta  [BLOCK_SIZE];
`endif
    wire                                tbuf_ready     [BLOCK_SIZE];

    // Per-block LMEM read port signals
    wire                         per_blk_rd_valid [BLOCK_SIZE];
    wire                         per_blk_rd_ready [BLOCK_SIZE];
    wire [BANK_ADDR_WIDTH-1:0]   per_blk_rd_addr  [BLOCK_SIZE];

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0]     tbuf_fetch_stalls_b [BLOCK_SIZE];
    wire [PERF_CTR_BITS-1:0]     lmem_reads_b        [BLOCK_SIZE];
`endif

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_tile_bufs
        // Per-block lmem interface: tile_buf drives valid/addr; data/data_valid
        // are broadcast from the module-level arbitrated interface.
        VX_tcu_lmem_if #(
            .DATA_WIDTH(`LMEM_NUM_BANKS * `XLEN),
            .ADDR_WIDTH(BANK_ADDR_WIDTH)
        ) blk_lmem_if();

        assign blk_lmem_if.rsp_data       = tcu_lmem_if.rsp_data;
        assign blk_lmem_if.rsp_valid      = tcu_lmem_if.rsp_valid;
        assign per_blk_rd_valid[block_idx]= blk_lmem_if.req_valid;
        assign blk_lmem_if.req_ready      = per_blk_rd_ready[block_idx];
        assign per_blk_rd_addr[block_idx] = blk_lmem_if.req_addr;

    `ifdef TCU_TMEM_ENABLE
        wire is_umma_b          = (core_execute_if[block_idx].data.op_type == INST_TCU_UMMA);
        wire is_wgmma_or_umma_b = (core_execute_if[block_idx].data.op_type == INST_TCU_WGMMA)
                               || is_umma_b;
        wire req_valid_b        = core_execute_if[block_idx].valid && is_wgmma_or_umma_b;
        wire req_fire_b         = core_execute_if[block_idx].valid
                               && core_execute_if[block_idx].ready
                               && is_wgmma_or_umma_b;
    `else
        wire is_umma_b      = 1'b0;
        wire is_wgmma_b     = (per_block_execute_if[block_idx].data.op_type == INST_TCU_WGMMA);
        wire req_valid_b    = per_block_execute_if[block_idx].valid && is_wgmma_b;
        wire req_fire_b     = per_block_execute_if[block_idx].valid
                           && per_block_execute_if[block_idx].ready
                           && is_wgmma_b;
    `endif

        VX_tcu_tbuf #(
            .INSTANCE_ID    (`SFORMATF(("%s-tbuf%0d", INSTANCE_ID, block_idx))),
            .TCU_TBUF_SIZE  (`NUM_WARPS),
            .NUM_BANKS      (`LMEM_NUM_BANKS),
            .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
        ) tile_buf (
            .clk              (clk),
            .reset            (reset),
        `ifdef PERF_ENABLE
            .tbuf_fetch_stalls(tbuf_fetch_stalls_b[block_idx]),
            .lmem_reads       (lmem_reads_b[block_idx]),
        `endif
            .req_valid        (req_valid_b),
            .req_fire         (req_fire_b),
            .req_wid          (per_block_execute_if[block_idx].data.header.wid),
            .req_is_sparse    (per_block_execute_if[block_idx].data.op_args.tcu.is_sparse),
            .req_is_umma      (is_umma_b),
            .req_step_m       (per_block_execute_if[block_idx].data.op_args.tcu.step_m),
            .req_step_n       (per_block_execute_if[block_idx].data.op_args.tcu.step_n),
            .req_step_k       (per_block_execute_if[block_idx].data.op_args.tcu.step_k),
            .req_fmt_s        (per_block_execute_if[block_idx].data.op_args.tcu.fmt_s),
            .req_a_from_smem  (per_block_execute_if[block_idx].data.op_args.tcu.a_from_smem),
            .req_cd_nregs     (per_block_execute_if[block_idx].data.op_args.tcu.cd_nregs),
            .req_desc_a       (per_block_execute_if[block_idx].data.rs1_data[0]),
            .req_desc_b       (per_block_execute_if[block_idx].data.rs2_data[0]),
            .tcu_lmem_if      (blk_lmem_if),
            // Tile data outputs
            .tbuf_rs1_data    (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data    (tbuf_rs2_data[block_idx]),
        `ifdef TCU_SPARSE_ENABLE
            .tbuf_sp_meta     (tbuf_sp_meta[block_idx]),
        `endif
            .tbuf_ready       (tbuf_ready[block_idx])
        );
    end

    // -------------------------------------------------------------------
    // LMEM port arbitration
    // -------------------------------------------------------------------
    //   For BLOCK_SIZE==1 (typical), this is a direct wire-through.
    //   For BLOCK_SIZE>1, a simple priority arbiter grants the port
    //   to one tile buffer at a time.

    if (BLOCK_SIZE == 1) begin : g_lmem_direct
        assign tcu_lmem_if.req_valid = per_blk_rd_valid[0];
        assign per_blk_rd_ready[0] = tcu_lmem_if.req_ready;
        assign tcu_lmem_if.req_addr = per_blk_rd_addr[0];
    end else begin : g_lmem_arb
        // Priority arbiter: lowest block index wins
        logic [$clog2(BLOCK_SIZE)-1:0] grant_idx;
        logic                          grant_valid;

        always_comb begin
            grant_idx   = '0;
            grant_valid = 1'b0;
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                if (per_blk_rd_valid[b] && !grant_valid) begin
                    grant_idx   = $clog2(BLOCK_SIZE)'(b);
                    grant_valid = 1'b1;
                end
            end
        end

        assign tcu_lmem_if.req_valid = grant_valid;
        assign tcu_lmem_if.req_addr = per_blk_rd_addr[grant_idx];

        for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_rd_ready
            assign per_blk_rd_ready[b] = tcu_lmem_if.req_ready
                                      && grant_valid
                                      && (grant_idx == $clog2(BLOCK_SIZE)'(b));
        end
    end

    // -------------------------------------------------------------------
    // Performance counters
    // -------------------------------------------------------------------

`ifdef PERF_ENABLE
    logic [PERF_CTR_BITS-1:0] tbuf_fetch_stalls_sum;
    logic [PERF_CTR_BITS-1:0] lmem_reads_sum;
    always_comb begin
        tbuf_fetch_stalls_sum = '0;
        lmem_reads_sum        = '0;
        for (int bi = 0; bi < BLOCK_SIZE; bi++) begin
            tbuf_fetch_stalls_sum += tbuf_fetch_stalls_b[bi];
            lmem_reads_sum        += lmem_reads_b[bi];
        end
    end
    assign tcu_perf.tbuf_fetch_stalls = tbuf_fetch_stalls_sum;
    assign tcu_perf.lmem_reads        = lmem_reads_sum;

    // wgmma_instrs / wgmma_stalls: derived from per_block_execute_if.
    logic wgmma_fire_b  [BLOCK_SIZE];
    logic wgmma_stall_b [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_wgmma_perf
        wire is_wgmma_p = (per_block_execute_if[bi].data.op_type == INST_TCU_WGMMA);
        assign wgmma_fire_b [bi] = per_block_execute_if[bi].valid && per_block_execute_if[bi].ready && is_wgmma_p;
        assign wgmma_stall_b[bi] = per_block_execute_if[bi].valid && !per_block_execute_if[bi].ready && is_wgmma_p;
    end

    logic [PERF_CTR_BITS-1:0] wgmma_instrs_ctr_r;
    logic [PERF_CTR_BITS-1:0] wgmma_stalls_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            wgmma_instrs_ctr_r <= '0;
            wgmma_stalls_ctr_r <= '0;
        end else begin
            for (int bi = 0; bi < BLOCK_SIZE; bi++) begin
                if (wgmma_fire_b[bi])  wgmma_instrs_ctr_r <= wgmma_instrs_ctr_r + PERF_CTR_BITS'(1);
                if (wgmma_stall_b[bi]) wgmma_stalls_ctr_r <= wgmma_stalls_ctr_r + PERF_CTR_BITS'(1);
            end
        end
    end
    assign tcu_perf.wgmma_instrs = wgmma_instrs_ctr_r;
    assign tcu_perf.wgmma_stalls = wgmma_stalls_ctr_r;

`ifdef TCU_TMEM_ENABLE
    // umma_instrs / umma_stalls
    logic umma_fire_b  [BLOCK_SIZE];
    logic umma_stall_b [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_umma_perf
        wire is_umma_p = (per_block_execute_if[bi].data.op_type == INST_TCU_UMMA);
        assign umma_fire_b [bi] = per_block_execute_if[bi].valid && per_block_execute_if[bi].ready && is_umma_p;
        assign umma_stall_b[bi] = per_block_execute_if[bi].valid && !per_block_execute_if[bi].ready && is_umma_p;
    end

    logic [PERF_CTR_BITS-1:0] umma_instrs_ctr_r;
    logic [PERF_CTR_BITS-1:0] umma_stalls_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            umma_instrs_ctr_r <= '0;
            umma_stalls_ctr_r <= '0;
        end else begin
            for (int bi = 0; bi < BLOCK_SIZE; bi++) begin
                if (umma_fire_b[bi])  umma_instrs_ctr_r <= umma_instrs_ctr_r + PERF_CTR_BITS'(1);
                if (umma_stall_b[bi]) umma_stalls_ctr_r <= umma_stalls_ctr_r + PERF_CTR_BITS'(1);
            end
        end
    end
    assign tcu_perf.umma_instrs = umma_instrs_ctr_r;
    assign tcu_perf.umma_stalls = umma_stalls_ctr_r;
`endif // TCU_TMEM_ENABLE
`endif // PERF_ENABLE

`else // !TCU_WGMMA_ENABLE

`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_fetch_stalls = '0;
    assign tcu_perf.lmem_reads        = '0;
    assign tcu_perf.wgmma_instrs      = '0;
    assign tcu_perf.wgmma_stalls      = '0;
    assign tcu_perf.umma_instrs       = '0;
    assign tcu_perf.umma_stalls       = '0;
`endif

`endif // TCU_WGMMA_ENABLE

`ifdef TCU_TMEM_ENABLE
    // -----------------------------------------------------------------------
    // TMEM storage and management
    // -----------------------------------------------------------------------

    logic [31:0]    tmem_data [TCU_TMEM_LANES][TCU_TMEM_COLS];
    logic           tmem_allocated;
    logic [7:0]     tmem_ncols;

    `UNUSED_VAR(tmem_allocated);
    `UNUSED_VAR(tmem_ncols);

    // TMEM read output
    wire [31:0] tmem_rd_data [BLOCK_SIZE][`NUM_THREADS];
    wire        tmem_rd_valid[BLOCK_SIZE];

    // Per-block outputs from tcu_core
    wire                                    tmem_wr_en        [BLOCK_SIZE];
    wire [TCU_TMEM_LANE_BITS-1:0]           tmem_wr_lane_base [BLOCK_SIZE];
    wire [TCU_TMEM_COL_BITS-1:0]            tmem_wr_col_base  [BLOCK_SIZE];
    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] tmem_wr_data      [BLOCK_SIZE];

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_tmem_mgmt
        always_ff @(posedge clk) begin
            if (reset) begin
                tmem_allocated <= 0;
                tmem_ncols     <= '0;
            end else begin
                if (per_block_execute_if[block_idx].valid && per_block_execute_if[block_idx].ready) begin
                    case (per_block_execute_if[block_idx].data.op_type)
                        INST_TCU_TMEM_ALLOC: begin
                            tmem_allocated <= 1'b1;
                            tmem_ncols     <= per_block_execute_if[block_idx].data.rs1_data[0][7:0];
                            for (int l = 0; l < TCU_TMEM_LANES; ++l)
                                for (int c = 0; c < TCU_TMEM_COLS; ++c)
                                    tmem_data[l][c] <= '0;
                        end
                        INST_TCU_TMEM_DEALLOC: begin
                            tmem_allocated <= 1'b0;
                            tmem_ncols     <= '0;
                        end
                        INST_TCU_TMEM_ST: begin
                            for (int t = 0; t < `NUM_THREADS; ++t) begin
                                if (per_block_execute_if[block_idx].data.header.tmask[t]) begin
                                    automatic logic [TCU_TMEM_LANE_BITS-1:0] lane_base =
                                        TCU_TMEM_LANE_BITS'(per_block_execute_if[block_idx].data.rs1_data[0][31:16]);
                                    automatic logic [TCU_TMEM_COL_BITS-1:0] col =
                                        TCU_TMEM_COL_BITS'(per_block_execute_if[block_idx].data.rs1_data[0][15:0]);
                                    tmem_data[TCU_TMEM_LANE_BITS'(lane_base) + TCU_TMEM_LANE_BITS'(t)][col] <=
                                        per_block_execute_if[block_idx].data.rs2_data[t][31:0];
                                end
                            end
                        end
                        default:;
                    endcase
                end
            end
        end
    end

    // TMEM_LD
    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_tmem_ld
        wire is_tmem_ld = per_block_execute_if[block_idx].valid 
                       && (per_block_execute_if[block_idx].data.op_type == INST_TCU_TMEM_LD);
        for (genvar t = 0; t < `NUM_THREADS; ++t) begin : g_tmem_ld_t
            wire [TCU_TMEM_LANE_BITS-1:0] lane_base = 
                TCU_TMEM_LANE_BITS'(per_block_execute_if[block_idx].data.rs1_data[0][31:16]);
            wire [TCU_TMEM_COL_BITS-1:0] col = 
                TCU_TMEM_COL_BITS'(per_block_execute_if[block_idx].data.rs1_data[0][15:0]);
            assign tmem_rd_data[block_idx][t] = is_tmem_ld ? tmem_data[lane_base + t][col] : '0;
        end
        assign tmem_rd_valid[block_idx] = is_tmem_ld;
    end

    // UMMA d_val writeback to TMEM
    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_tmem_wr
        always_ff @(posedge clk) begin
            if (tmem_wr_en[block_idx]) begin
                for (int i = 0; i < TCU_TC_M; ++i) begin
                    for (int j = 0; j < TCU_TC_N; ++j) begin
                        tmem_data   [TCU_TMEM_LANE_BITS'(tmem_wr_lane_base[block_idx]) + TCU_TMEM_LANE_BITS'(i)]
                                    [TCU_TMEM_COL_BITS'(tmem_wr_col_base[block_idx])   + TCU_TMEM_COL_BITS'(j)]
                            <= tmem_wr_data[block_idx][i][j];
                    end
                end
            end
        end
    end

    // -----------------------------------------------------------------------
    // TMEM bypass, routes ALLOC/DEALLOC/ST/LD around tcu_core
    // -----------------------------------------------------------------------

    VX_execute_if #(.data_t(tcu_execute_t)) core_execute_if[BLOCK_SIZE]();
    VX_result_if  #(.data_t(tcu_result_t))  tmem_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_tmem_bypass
        wire is_tmem_mgmt = (per_block_execute_if[block_idx].data.op_type == INST_TCU_TMEM_ALLOC)
                         || (per_block_execute_if[block_idx].data.op_type == INST_TCU_TMEM_DEALLOC)
                         || (per_block_execute_if[block_idx].data.op_type == INST_TCU_TMEM_ST)
                         || (per_block_execute_if[block_idx].data.op_type == INST_TCU_TMEM_LD);

        // Route TMEM management instructions directly to bypass result, everything else to tcu_core
        assign core_execute_if[block_idx].valid = per_block_execute_if[block_idx].valid && ~is_tmem_mgmt;
        assign core_execute_if[block_idx].data = per_block_execute_if[block_idx].data;

        // TMEM ALLOC/DEALLOC/ST ready immediately, results fire same cycle
        assign tmem_result_if[block_idx].valid = per_block_execute_if[block_idx].valid && is_tmem_mgmt;
        assign tmem_result_if[block_idx].data.header = per_block_execute_if[block_idx].data.header;

        // TMEM LD puts result data into result
        for (genvar t = 0; t < `NUM_THREADS; ++t) begin : g_tmem_ld_result
            if (`XLEN > 32) begin : g_nanbox
                assign tmem_result_if[block_idx].data.data[t] = tmem_rd_valid[block_idx]
                                                              ? {32'hffffffff, tmem_rd_data[block_idx][t]}
                                                              : '0;
            end else begin : g_pass
                assign tmem_result_if[block_idx].data.data[t] = tmem_rd_valid[block_idx]
                                                              ? `XLEN'(tmem_rd_data[block_idx][t])
                                                              : '0;
            end
        end

        assign per_block_execute_if[block_idx].ready = is_tmem_mgmt
                                                     ? tmem_result_if[block_idx].ready
                                                     : core_execute_if[block_idx].ready;
    end
`endif // TCU_TMEM_ENABLE

    // -----------------------------------------------------------------------
    // TCU core instances
    // -----------------------------------------------------------------------

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
        VX_tcu_core #(
            .INSTANCE_ID (`SFORMATF(("%s-fused%0d", INSTANCE_ID, block_idx)))
        ) tcu_core (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
        `ifdef TCU_WGMMA_ENABLE
            .tbuf_rs1_data (tbuf_rs1_data[block_idx]),
            .tbuf_rs2_data (tbuf_rs2_data[block_idx]),
        `ifdef TCU_SPARSE_ENABLE
            .tbuf_sp_meta  (tbuf_sp_meta[block_idx]),
        `endif
            .tbuf_ready (tbuf_ready[block_idx]),
        `endif
        `ifdef TCU_TMEM_ENABLE
            .tmem_data          (tmem_data),
            .tmem_wr_en         (tmem_wr_en[block_idx]),
            .tmem_wr_lane_base  (tmem_wr_lane_base[block_idx]),
            .tmem_wr_col_base   (tmem_wr_col_base[block_idx]),
            .tmem_wr_data       (tmem_wr_data[block_idx]),
            .execute_if         (core_execute_if[block_idx]),
        `else
            .execute_if         (per_block_execute_if[block_idx]),
        `endif
            .result_if          (per_block_result_if[block_idx])
        );
    end

    // -----------------------------------------------------------------------
    // Merge result streams (core and bypass)
    // -----------------------------------------------------------------------

`ifdef TCU_TMEM_ENABLE
    VX_result_if #(.data_t(tcu_result_t)) merged_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_merge
        assign merged_result_if[block_idx].valid = per_block_result_if[block_idx].valid 
                                                || tmem_result_if[block_idx].valid;
        assign merged_result_if[block_idx].data = tmem_result_if[block_idx].valid 
                                                ? tmem_result_if[block_idx].data
                                                : per_block_result_if[block_idx].data;
        assign per_block_result_if[block_idx].ready = merged_result_if[block_idx].ready 
                                                   && ~tmem_result_if[block_idx].valid;
        assign tmem_result_if[block_idx].ready = merged_result_if[block_idx].ready;
    end
`endif

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
    `ifdef TCU_TMEM_ENABLE
        .result_if (merged_result_if),
    `else
        .result_if (per_block_result_if),
    `endif
        .commit_if (commit_if)
    );

endmodule
