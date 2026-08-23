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
// PLACEHOLDER: this is tensor_mem's original single-tenant allocator (one
// global allocation, no per-CTA handle), ported onto the current TCU
// structure — NOT the real multi-tenant allocator (CTA-scoped free-list +
// CAM, matching the SimX side). Deliberately simple, to verify the UMMA
// FEDP compute path + RAW-hazard interlock first. Redo before any
// multi-CTA-concurrent UMMA testing:
//   - ALLOC/DEALLOC/ST all share one always_ff below with NO real
//     arbitration across blocks — if more than one block requests one of
//     these in the same cycle, whichever has the highest block index
//     wins (last write in the loop), the rest are silently dropped rather
//     than retried.
//   - handle is always implicitly 0 (tmem_ncols tracks one allocation, no
//     per-CTA base column). DEALLOC clears bookkeeping only, not storage.

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
    logic        tmem_allocated;
    logic [7:0]  tmem_ncols;
    `UNUSED_VAR (tmem_allocated)
    `UNUSED_VAR (tmem_ncols)

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

    // ALLOC/DEALLOC/ST + UMMA writeback: combined into one process so
    // there's a single driver for tmem_data/tmem_allocated/tmem_ncols
    always_ff @(posedge clk) begin
        if (reset) begin
            tmem_allocated <= 1'b0;
            tmem_ncols     <= '0;
        end else begin
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

            // ALLOC/DEALLOC/ST
            for (int b = 0; b < BLOCK_SIZE; ++b) begin
                if (mgmt_valid[b]) begin
                    case (mgmt_data[b].op_type)
                        INST_TCU_TMEM_ALLOC: begin
                            tmem_allocated <= 1'b1;
                            tmem_ncols     <= mgmt_data[b].rs1_data[0][7:0];
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
                        default: ;
                    endcase
                end
            end
        end
    end

    // Bypass execute<->result handshake: same-cycle response, no queueing
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tmem_result
        assign mgmt_ready[bi]         = result_ready[bi];
        assign result_valid[bi]       = mgmt_valid[bi];
        assign result_data[bi].header = mgmt_data[bi].header;
        wire is_tmem_ld_r = mgmt_data[bi].op_type == INST_TCU_TMEM_LD;
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_tmem_result_t
            if (`VX_CFG_XLEN > 32) begin : g_nanbox
                assign result_data[bi].data[t] = is_tmem_ld_r
                    ? {32'hffffffff, tmem_ld_rd_data[bi][t]} : '0;
            end else begin : g_pass
                assign result_data[bi].data[t] = is_tmem_ld_r
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

endmodule

`endif // TCU_TMEM_ENABLE
