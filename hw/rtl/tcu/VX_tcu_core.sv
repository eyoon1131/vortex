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

module VX_tcu_core import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire          clk,
    input wire          reset,

`ifdef TCU_WGMMA_ENABLE
    input wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0]        tbuf_rs1_data,
    input wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] tbuf_rs2_data,
`ifdef TCU_SPARSE_ENABLE
    input wire [TCU_MAX_META_BLOCK_WIDTH-1:0] tbuf_sp_meta,
`endif
    input wire          tbuf_ready,
`endif

`ifdef TCU_TMEM_ENABLE
    // TMEM read/write ports
    input wire [31:0]                               tmem_data[TCU_TMEM_LANES][TCU_TMEM_COLS],
    input wire [TCU_TMEM_LANE_BITS-1:0]             tmem_warp_rank,
    output wire                                     tmem_wr_en,
    output wire [TCU_TMEM_LANE_BITS-1:0]            tmem_wr_lane_base,
    output wire [TCU_TMEM_COL_BITS-1:0]             tmem_wr_col_base,
    output wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0]  tmem_wr_data,
`endif

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_result_if.master result_if
);
    `UNUSED_SPARAM (INSTANCE_ID);

    localparam PIPE_LATENCY = FEDP_LATENCY + 1;
    localparam MDATA_QUEUE_DEPTH = 1 << $clog2(PIPE_LATENCY);

    localparam LG_A_BS    = $clog2(TCU_A_BLOCK_SIZE);
    localparam LG_B_BS    = $clog2(TCU_B_BLOCK_SIZE);
    localparam OFF_W      = $clog2(TCU_BLOCK_CAP);

`ifdef TCU_SPARSE_ENABLE
    localparam LG_B_BS_SP = $clog2(TCU_B_BLOCK_SIZE_SP);
    wire is_sparse = execute_if.data.op_args.tcu.is_sparse;
    wire is_meta_store = (execute_if.data.op_type == INST_TCU_META_STORE);
`endif

    // -----------------------------------------------------------------------
    // Operand data mux: WGMMA uses tile buffer, WMMA uses register file
    // -----------------------------------------------------------------------

    wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] rs1_data;
    wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] rs2_data;

`ifdef TCU_WGMMA_ENABLE
    wire is_wgmma = (execute_if.data.op_type == INST_TCU_WGMMA);
`ifdef TCU_TMEM_ENABLE
    wire is_umma  = (execute_if.data.op_type == INST_TCU_UMMA);
`endif
    wire wg_a_smem = execute_if.data.op_args.tcu.a_from_smem;
    // A source: tile buffer (smem) or register file
    assign rs1_data = ((is_wgmma && wg_a_smem) 
`ifdef TCU_TMEM_ENABLE
                    || is_umma
`endif
                    ) ? tbuf_rs1_data : execute_if.data.rs1_data;
    // B source: always tile buffer (smem) for WGMMA/UMMA
    assign rs2_data = (is_wgmma
`ifdef TCU_TMEM_ENABLE
                    || is_umma
`endif
                    ) ? tbuf_rs2_data[TCU_BLOCK_CAP-1:0] : execute_if.data.rs2_data;
`else
    assign rs1_data = execute_if.data.rs1_data;
    assign rs2_data = execute_if.data.rs2_data;
`endif

    wire [2:0] step_m = execute_if.data.op_args.tcu.step_m;
    wire [6:0] step_n = execute_if.data.op_args.tcu.step_n;
    wire [2:0] step_k = execute_if.data.op_args.tcu.step_k;

    wire [4:0] fmt_s = execute_if.data.op_args.tcu.fmt_s;
    wire [4:0] fmt_d = execute_if.data.op_args.tcu.fmt_d;

    wire execute_fire = execute_if.valid && execute_if.ready;

    // -----------------------------------------------------------------------
    // Sparse metadata: VX_tcu_meta (for WMMA_SP) + optional tile-buffer mux
    // -----------------------------------------------------------------------

`ifdef TCU_SPARSE_ENABLE
    wire [`LOG2UP(`NUM_WARPS)-1:0] wid = execute_if.data.header.wid;
    wire meta_wr_en = execute_fire && is_meta_store;

    // meta_store: force rd=0 in mdata_queue header
    tcu_header_t mdata_queue_in;
    always_comb begin
        mdata_queue_in = execute_if.data.header;
        if (is_meta_store) begin
            mdata_queue_in.rd = '0;
        end
    end
`else
    tcu_header_t mdata_queue_in;
    always_comb begin
        mdata_queue_in = execute_if.data.header;
    end
`endif

`ifdef TCU_TMEM_ENABLE
    `UNUSED_VAR ({step_k, fmt_s, fmt_d, execute_if.data});
`else
    `UNUSED_VAR ({step_m, step_n, step_k, fmt_s, fmt_d, execute_if.data});
`endif

    // -----------------------------------------------------------------------
    // Pipeline control
    // -----------------------------------------------------------------------

    wire mdata_queue_full;

    wire result_fire = result_if.valid && result_if.ready;
    wire fedp_enable, fedp_done;

    reg [PIPE_LATENCY-1:0] fedp_delay_pipe;
    always @(posedge clk) begin
        if (reset) begin
            fedp_delay_pipe <= '0;
        end else begin
            if (fedp_enable) begin
                fedp_delay_pipe <= fedp_delay_pipe >> 1;
            end
            if (execute_fire) begin
                fedp_delay_pipe[PIPE_LATENCY-1] <= 1;
            end
        end
    end
    assign fedp_done = fedp_delay_pipe[0];

    assign result_if.valid  = fedp_done;
    assign fedp_enable      = ~result_if.valid || result_if.ready;
`ifdef TCU_WGMMA_ENABLE
`ifdef TCU_TMEM_ENABLE
    assign execute_if.ready = ~mdata_queue_full && fedp_enable 
                           && (~(is_wgmma || is_umma) || tbuf_ready);
`else
    assign execute_if.ready = ~mdata_queue_full && fedp_enable && (~is_wgmma || tbuf_ready);
`endif
`else
    assign execute_if.ready = ~mdata_queue_full && fedp_enable;
`endif

    VX_fifo_queue #(
        .DATAW ($bits(tcu_header_t)),
        .DEPTH (MDATA_QUEUE_DEPTH),
        .OUT_REG (1)
    ) mdata_queue (
        .clk    (clk),
        .reset  (reset),
        .push   (execute_fire),
        .pop    (result_fire),
        .data_in(mdata_queue_in),
        .data_out(result_if.data.header),
        `UNUSED_PIN(empty),
        `UNUSED_PIN(alm_empty),
        .full   (mdata_queue_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    // -----------------------------------------------------------------------
    // Operand offset computation
    // -----------------------------------------------------------------------

    wire [OFF_W-1:0] a_off = (OFF_W'(step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;
`ifdef TCU_SPARSE_ENABLE
    wire [OFF_W-1:0] b_off = is_sparse
        ? (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS_SP-1)) << LG_B_BS_SP
        : (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`else
    wire [OFF_W-1:0] b_off = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`endif

    // -----------------------------------------------------------------------
    // Unified sparse metadata
    // -----------------------------------------------------------------------
    // WMMA_SP:  from VX_tcu_meta (per-warp register-file metadata store)
    // WGMMA_SP: from VX_tcu_tbuf (pre-extracted from SMEM metadata)
    // Both produce TCU_TC_M per-row slices of TCU_MAX_META_ROW_WIDTH bits,
    // indexed by (step_m, step_k).

`ifdef TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] wmma_sp_meta;
    VX_tcu_meta #(
        .INSTANCE_ID (INSTANCE_ID)
    ) tcu_meta (
        .clk    (clk),
        .reset  (reset),
        .wr_en  (meta_wr_en),
        .wr_wid (wid),
        .wr_idx (fmt_d),
        .wr_data(rs1_data),
        .rd_wid (wid),
        .step_m (step_m),
        .step_k (step_k),
        .vld_block(wmma_sp_meta)
    );

    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block;
    `ifdef TCU_WGMMA_ENABLE
        assign vld_meta_block = is_wgmma ? tbuf_sp_meta : wmma_sp_meta;
    `else
        assign vld_meta_block = wmma_sp_meta;
    `endif
`endif

    // -----------------------------------------------------------------------
    // FEDP grid: TCU_TC_M × TCU_TC_N compute elements
    // -----------------------------------------------------------------------

    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] d_val;

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
        for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j
        `ifdef TCU_SPARSE_ENABLE
            wire [TCU_TC_K-1:0][31:0] a_row, b_col, b_col_dense, b_col_sparse, b_col_1, b_col_2;
        `else
            wire [TCU_TC_K-1:0][31:0] a_row, b_col;
        `endif
            for (genvar k_idx = 0; k_idx < TCU_TC_K; ++k_idx) begin : g_slice_assign
                assign a_row[k_idx] = 32'(rs1_data[a_off + i * TCU_TC_K + k_idx]);
            `ifdef TCU_SPARSE_ENABLE
                assign b_col_dense[k_idx] = 32'(rs2_data[b_off + j * TCU_TC_K + k_idx]);
                // WGMMA_SP: tbuf_rs2_data is wide (TCU_WG_RS2_WIDTH lanes);
                //   use j directly — gather already placed each column's pair at j*tcK*2.
                // WMMA_SP: rs2_data comes from the register file (TCU_BLOCK_CAP lanes);
                //   SYM_SPARSE folds j to the packed column-pair layout.
                localparam J_SP = SYM_SPARSE ? (j % (TCU_TC_N / 2)) : j;
            `ifdef TCU_WGMMA_ENABLE
                assign b_col_1[k_idx] = 32'(is_wgmma
                    ? tbuf_rs2_data[j * TCU_TC_K * 2 + k_idx * 2]
                    : rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2]);
                assign b_col_2[k_idx] = 32'(is_wgmma
                    ? tbuf_rs2_data[j * TCU_TC_K * 2 + k_idx * 2 + 1]
                    : rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2 + 1]);
            `else
                assign b_col_1[k_idx] = 32'(rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2]);
                assign b_col_2[k_idx] = 32'(rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2 + 1]);
            `endif
            `else
                assign b_col[k_idx] = 32'(rs2_data[b_off + j * TCU_TC_K + k_idx]);
            `endif
            end

        `ifdef TCU_TMEM_ENABLE
            wire [TCU_TMEM_LANE_BITS-1:0] tmem_lane = TCU_TMEM_LANE_BITS'(tmem_warp_rank) * TCU_TMEM_LANE_BITS'(TCU_WG_TILE_M)
                                                    + TCU_TMEM_LANE_BITS'(step_m) * TCU_TMEM_LANE_BITS'(TCU_TC_M)
                                                    + TCU_TMEM_LANE_BITS'(i);
            wire [TCU_TMEM_COL_BITS-1:0]  tmem_col  = TCU_TMEM_COL_BITS'(step_n) * TCU_TMEM_COL_BITS'(TCU_TC_N)
                                                    + TCU_TMEM_COL_BITS'(j);
            wire [31:0] c_val = is_umma ? tmem_data[tmem_lane][tmem_col]
                                        : 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);
        `else
            wire [31:0] c_val = 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);
        `endif

        `ifdef TCU_SPARSE_ENABLE
            VX_tcu_sp_mux #(
                .INSTANCE_ID (INSTANCE_ID),
                .ROW_IDX     (i)
            ) tcu_sp_mux (
                .fmt_s     (fmt_s),
                .b_col_in1 (b_col_1),
                .b_col_in2 (b_col_2),
                .vld_mask  (vld_meta_block),
                .b_col_out (b_col_sparse)
            );
            assign b_col = is_sparse ? b_col_sparse : b_col_dense;
        `endif

            wire [4:0] fmt_s_r, fmt_d_r;
            wire [TCU_TC_K-1:0][31:0] a_row_r, b_col_r;
            wire [31:0] c_val_r;

            `BUFFER_EX (
                {c_val_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r},
                {c_val,   fmt_s,   fmt_d,   b_col,   a_row},
                fedp_enable,
                0, // resetw
                1  // depth
            );

        `ifdef TCU_TYPE_DPI
            VX_tcu_fedp_dpi #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_TYPE_BHF
            VX_tcu_fedp_bhf #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_TYPE_FPNEW
            VX_tcu_fedp_fpnew #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_TYPE_TFR
            VX_tcu_fedp_tfr #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .vld_mask('1),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_TYPE_DSP
            VX_tcu_fedp_dsp #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `endif

            // NaN-box the fp32 result for XLEN=64: upper 32 bits must be all-1s per RVF spec.
            if (`XLEN > 32) begin : g_result_nanbox
                assign result_if.data.data[i * TCU_TC_N + j] = {32'hffffffff, d_val[i][j]};
            end else begin : g_result_passthrough
                assign result_if.data.data[i * TCU_TC_N + j] = d_val[i][j];
            end

        `ifdef DBG_TRACE_TCU
            always @(posedge clk) begin
                if (execute_if.valid && execute_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-enq: wid=%0d, i=%0d, j=%0d, m=%0d, n=%0d, a_row=", $time, INSTANCE_ID, execute_if.data.header.wid, i, j, step_m, step_n))
                    `TRACE_ARRAY1D(2, "0x%0h", a_row, TCU_TC_K)
                    `TRACE(3, (", b_col="));
                    `TRACE_ARRAY1D(2, "0x%0h", b_col, TCU_TC_K)
                    `TRACE(3, (", c_val=0x%0h (#%0d)\n", c_val, execute_if.data.header.uuid));
                end
                if (result_if.valid && result_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-deq: wid=%0d, i=%0d, j=%0d, d_val=0x%0h (#%0d)\n", $time, INSTANCE_ID, result_if.data.header.wid, i, j, d_val[i][j], result_if.data.header.uuid));
                end
            end
        `endif // DBG_TRACE_TCU
        end
    end

`ifdef TCU_TMEM_ENABLE
    // -----------------------------------------------------------------------
    // TMEM write: delay pipe carries write address through FEDP latency
    // -----------------------------------------------------------------------
    typedef struct packed {
        logic                           valid;
        logic [TCU_TMEM_LANE_BITS-1:0]  lane_base;  // warp_rank * xtileM + step_m * tcM
        logic [TCU_TMEM_COL_BITS-1:0]   col_base;   // step_n * tcN
    } tmem_addr_t;

    tmem_addr_t tmem_addr_pipe [PIPE_LATENCY];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int p = 0; p < PIPE_LATENCY; ++p)
                tmem_addr_pipe[p] <= '0;
        end else if (fedp_enable) begin
            for (int p = 0; p < PIPE_LATENCY-1; ++p)
                tmem_addr_pipe[p] <= tmem_addr_pipe[p+1];
            tmem_addr_pipe[PIPE_LATENCY-1].valid        <= execute_fire && is_umma;
            tmem_addr_pipe[PIPE_LATENCY-1].lane_base    <= TCU_TMEM_LANE_BITS'(TCU_TMEM_LANE_BITS'(tmem_warp_rank) 
                                                         * TCU_TMEM_LANE_BITS'(TCU_WG_TILE_M)
                                                         + TCU_TMEM_LANE_BITS'(step_m) 
                                                         * TCU_TMEM_LANE_BITS'(TCU_TC_M));
            tmem_addr_pipe[PIPE_LATENCY-1].col_base     <= TCU_TMEM_COL_BITS'(TCU_TMEM_COL_BITS'(step_n) 
                                                         * TCU_TMEM_COL_BITS'(TCU_TC_N));
        end
    end

    assign tmem_wr_en        = tmem_addr_pipe[0].valid;
    assign tmem_wr_lane_base = tmem_addr_pipe[0].lane_base;
    assign tmem_wr_col_base  = tmem_addr_pipe[0].col_base;
    assign tmem_wr_data      = d_val;  // d_val is available when fedp_done fires
`endif

endmodule
