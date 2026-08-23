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

`ifdef TCU_TMEM_ENABLE

// TMEM (Tensor Memory) storage and management
//
// TMEM_ALLOC/TMEM_DEALLOC are per-CTA allocation, delegated to
// VX_tcu_tmem_alloc: one block's ALLOC-or-DEALLOC request is arbitrated to
// a single winner per cycle. alloc() only ever reserves a column range —
// it does not initialize TMEM contents, so the allocator has no dependency
// on tmem_data

module VX_tcu_tmem import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter         BLOCK_SIZE  = `VX_CFG_NUM_TCU_BLOCKS
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output tcu_perf_t tcu_perf,
`endif

    // TMEM_ALLOC/DEALLOC/ST/LD management ops (one execute/result_if per
    // block)
    input  wire [BLOCK_SIZE-1:0] mgmt_valid,
    input  tcu_execute_t         mgmt_data  [BLOCK_SIZE],
    output wire [BLOCK_SIZE-1:0] mgmt_ready,

    output wire [BLOCK_SIZE-1:0] result_valid,
    output tcu_result_t          result_data  [BLOCK_SIZE],
    input  wire [BLOCK_SIZE-1:0] result_ready,

    // UMMA compute read (operand C) — request is this op's tile origin
    // (lane_base/col_base, same value for every cell); response is the
    // whole TC_M x TC_N tile
    input  wire [TCU_TMEM_LANE_BITS-1:0]            rd_lane_base [BLOCK_SIZE],
    input  wire [TCU_TMEM_COL_BITS-1:0]             rd_col_base  [BLOCK_SIZE],
    output wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0]  rd_data      [BLOCK_SIZE],

    // UMMA compute writeback, driven off the FEDP result once it retires
    input wire                                     wr_en        [BLOCK_SIZE],
    input wire [TCU_TMEM_LANE_BITS-1:0]            wr_lane_base [BLOCK_SIZE],
    input wire [TCU_TMEM_COL_BITS-1:0]             wr_col_base  [BLOCK_SIZE],
    input wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0]  wr_data      [BLOCK_SIZE]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    logic [31:0] tmem_data [TCU_TMEM_LANES][TCU_TMEM_COLS];

    // -----------------------------------------------------------------------
    // UMMA compute read: decode once per (block, tile row), reuse across
    // that row's TC_N columns, forward only the resulting tile.
    // -----------------------------------------------------------------------
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_rd
        for (genvar ii = 0; ii < TCU_TC_M; ++ii) begin : g_rd_row
            wire [TCU_TMEM_LANE_BITS-1:0] rd_row = rd_lane_base[bi] + TCU_TMEM_LANE_BITS'(ii);
            for (genvar jj = 0; jj < TCU_TC_N; ++jj) begin : g_rd_col
                wire [TCU_TMEM_COL_BITS-1:0] rd_col = rd_col_base[bi] + TCU_TMEM_COL_BITS'(jj);
                assign rd_data[bi][ii][jj] = tmem_data[rd_row][rd_col];
            end
        end
    end

    // TMEM_LD is combinational, never conflicts across blocks
    wire [31:0] tmem_ld_rd_data [BLOCK_SIZE][`VX_CFG_NUM_THREADS];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_ld
        wire is_tmem_ld = mgmt_valid[bi]
                        && (mgmt_data[bi].op_type == INST_TCU_TMEM_LD);
        wire [TCU_TMEM_LANE_BITS-1:0] ld_lane_base =
            TCU_TMEM_LANE_BITS'(mgmt_data[bi].rs1_data[0][31:16]);
        wire [TCU_TMEM_COL_BITS-1:0] ld_col =
            TCU_TMEM_COL_BITS'(mgmt_data[bi].rs1_data[0][15:0]);
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_tmem_ld_t
            assign tmem_ld_rd_data[bi][t] = is_tmem_ld
                ? tmem_data[TCU_TMEM_LANE_BITS'(ld_lane_base) + TCU_TMEM_LANE_BITS'(t)][ld_col]
                : '0;
        end
    end

    // UMMA compute writeback + TMEM_ST: both write their own tmem_data
    // cells independently, never touching shared allocator state, so
    // neither needs arbitration across blocks
    always_ff @(posedge clk) begin
        // UMMA compute writeback
        for (int b = 0; b < BLOCK_SIZE; ++b) begin
            if (wr_en[b]) begin
                for (int i = 0; i < TCU_TC_M; ++i) begin
                    for (int j = 0; j < TCU_TC_N; ++j) begin
                        tmem_data[TCU_TMEM_LANE_BITS'(wr_lane_base[b]) + TCU_TMEM_LANE_BITS'(i)]
                                 [TCU_TMEM_COL_BITS'(wr_col_base[b])  + TCU_TMEM_COL_BITS'(j)]
                            <= wr_data[b][i][j];
                    end
                end
            end
        end

        // TMEM_ST
        for (int b = 0; b < BLOCK_SIZE; ++b) begin
            if (mgmt_valid[b] && (mgmt_data[b].op_type == INST_TCU_TMEM_ST)) begin
                for (int t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin
                    if (mgmt_data[b].header.tmask[t]) begin
                        automatic logic [TCU_TMEM_LANE_BITS-1:0] st_lane_base =
                            TCU_TMEM_LANE_BITS'(mgmt_data[b].rs1_data[0][31:16]);
                        automatic logic [TCU_TMEM_COL_BITS-1:0] st_col =
                            TCU_TMEM_COL_BITS'(mgmt_data[b].rs1_data[0][15:0]);
                        tmem_data[TCU_TMEM_LANE_BITS'(st_lane_base) + TCU_TMEM_LANE_BITS'(t)][st_col]
                            <= mgmt_data[b].rs2_data[t][31:0];
                    end
                end
            end
        end
    end

    // -----------------------------------------------------------------------
    // ALLOC/DEALLOC arbitration: one block's request is granted per cycle
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] is_alloc_req;
    wire [BLOCK_SIZE-1:0] is_dealloc_req;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_mgmt_decode
        assign is_alloc_req[bi]   = mgmt_valid[bi] && (mgmt_data[bi].op_type == INST_TCU_TMEM_ALLOC);
        assign is_dealloc_req[bi] = mgmt_valid[bi] && (mgmt_data[bi].op_type == INST_TCU_TMEM_DEALLOC);
    end
    wire [BLOCK_SIZE-1:0] is_alloc_or_dealloc_req = is_alloc_req | is_dealloc_req;

    wire [BLOCK_SIZE-1:0]          alloc_grant_onehot;
    wire [`LOG2UP(BLOCK_SIZE)-1:0] alloc_grant_idx;
    wire                           alloc_grant_valid;
    VX_priority_encoder #(
        .N (BLOCK_SIZE)
    ) alloc_arb (
        .data_in    (is_alloc_or_dealloc_req),
        .onehot_out (alloc_grant_onehot),
        .index_out  (alloc_grant_idx),
        .valid_out  (alloc_grant_valid)
    );

    // Granted request's fields, extracted once from the winning block.
    // req_ncols is meaningful only for ALLOC, req_handle only for DEALLOC
    wire                          alloc_req_valid      = alloc_grant_valid;
    wire                          alloc_req_is_dealloc = is_dealloc_req[alloc_grant_idx];
    wire [NCTA_WIDTH-1:0]         alloc_req_cta_id     = mgmt_data[alloc_grant_idx].header.cta_id;
    wire [7:0]                    alloc_req_ncols      = mgmt_data[alloc_grant_idx].rs1_data[0][7:0];
    wire [TCU_TMEM_COL_BITS-1:0]  alloc_req_handle     = TCU_TMEM_COL_BITS'(mgmt_data[alloc_grant_idx].rs1_data[0]);

    wire                          alloc_resp_valid;
    wire [TCU_TMEM_COL_BITS-1:0]  alloc_resp_handle;

    VX_tcu_tmem_alloc #(
        .INSTANCE_ID (`SFORMATF(("%s-alloc", INSTANCE_ID)))
    ) alloc (
        .clk            (clk),
        .reset          (reset),
        .req_valid      (alloc_req_valid),
        .req_is_dealloc (alloc_req_is_dealloc),
        .req_cta_id     (alloc_req_cta_id),
        .req_ncols      (alloc_req_ncols),
        .req_handle     (alloc_req_handle),
        .resp_valid     (alloc_resp_valid),
        .resp_handle    (alloc_resp_handle)
    );

    // Bypass execute<->result handshake: same-cycle response, no queueing.
    // ALLOC/DEALLOC only fires for the block the arbiter granted this cycle,
    // gated on VX_tcu_tmem_alloc's resp_valid - every other requesting block
    // retries next cycle
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_result
        wire mgmt_fire = is_alloc_or_dealloc_req[bi]
            ? (alloc_grant_onehot[bi] && alloc_resp_valid)
            : mgmt_valid[bi];

        assign mgmt_ready[bi]         = mgmt_fire && result_ready[bi];
        assign result_valid[bi]       = mgmt_fire;
        assign result_data[bi].header = mgmt_data[bi].header;
        wire is_tmem_ld_r    = mgmt_data[bi].op_type == INST_TCU_TMEM_LD;
        wire is_tmem_alloc_r = mgmt_data[bi].op_type == INST_TCU_TMEM_ALLOC;
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_tmem_result_t
            if (`VX_CFG_XLEN > 32) begin : g_nanbox
                assign result_data[bi].data[t] = is_tmem_ld_r
                    ? {32'hffffffff, tmem_ld_rd_data[bi][t]}
                    : is_tmem_alloc_r ? `VX_CFG_XLEN'(alloc_resp_handle) : '0;
            end else begin : g_pass
                assign result_data[bi].data[t] = is_tmem_ld_r
                    ? `VX_CFG_XLEN'(tmem_ld_rd_data[bi][t])
                    : is_tmem_alloc_r ? `VX_CFG_XLEN'(alloc_resp_handle) : '0;
            end
        end
    end

`ifdef PERF_ENABLE
    // Stub for now
    assign tcu_perf.umma_instrs = '0;
    assign tcu_perf.tmem_reads  = '0;
    assign tcu_perf.tmem_writes = '0;
`endif

endmodule

`endif // TCU_TMEM_ENABLE
