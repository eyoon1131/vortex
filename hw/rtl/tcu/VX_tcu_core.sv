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

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    input wire [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    input wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] tbuf_rs2_data,
    input wire          tbuf_ready,
`endif

    // TMEM read/write ports. Read is request/response, tile-origin in,
    // whole TC_M x TC_N tile back
`ifdef TCU_TMEM_ENABLE
    input wire [NW_WIDTH-1:0]                       cta_rank,
    output wire                                     tmem_rd_valid,
    output wire [TCU_TMEM_LANE_BITS-1:0]            tmem_rd_lane_base,
    output wire [TCU_TMEM_COL_BITS-1:0]             tmem_rd_col_base,
    input wire                                      tmem_rd_grant,
    input wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0]   tmem_rd_data,
    output wire                                     tmem_wr_valid,
    output wire [TCU_TMEM_LANE_BITS-1:0]            tmem_wr_lane_base,
    output wire [TCU_TMEM_COL_BITS-1:0]             tmem_wr_col_base,
    output wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0]  tmem_wr_data,
    input wire                                      tmem_wr_grant,
`endif

    // External metadata write port from the shared VX_tcu_agu.
`ifdef TCU_META_ENABLE
    input wire                     ext_meta_wr_en,
    input wire [NW_WIDTH-1:0]      ext_meta_wr_wid,
    input wire [4:0]               ext_meta_wr_idx,
    input wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] ext_meta_wr_data,
`endif

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_result_if.master result_if
);
    `UNUSED_SPARAM (INSTANCE_ID);

    localparam FEDP_K = TCU_WG_FEDP_K;

`ifdef VX_CFG_TCU_TYPE_DSP
    localparam FCVT_LATENCY = 1;
    localparam FMUL_LATENCY = 8;
    localparam FADD_LATENCY = 11;
    localparam FACC_LATENCY = $clog2(2 * FEDP_K) * FADD_LATENCY;
    localparam FEDP_LATENCY = FCVT_LATENCY + FMUL_LATENCY + FACC_LATENCY + FADD_LATENCY;
`elsif VX_CFG_TCU_TYPE_BHF
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 2;
    localparam FRND_LATENCY = 1;
    localparam FACC_LATENCY  = $clog2(2 * FEDP_K) * (FADD_LATENCY + FRND_LATENCY);
    localparam FEDP_LATENCY = (FMUL_LATENCY + FRND_LATENCY) + 1 + FACC_LATENCY + (FADD_LATENCY + FRND_LATENCY);
`elsif VX_CFG_TCU_TYPE_FPNEW
    localparam FMUL_LATENCY = 6;
    localparam FMUX_LATENCY = 1;
    localparam FADD_LATENCY = 7;
    localparam FACC_LATENCY = $clog2(2 * FEDP_K) * FADD_LATENCY;
    localparam FEDP_LATENCY = FMUL_LATENCY + FMUX_LATENCY + FACC_LATENCY + FADD_LATENCY;
`elsif VX_CFG_TCU_TYPE_DPI
    localparam FMUL_LATENCY = 2;
    localparam FACC_LATENCY = 2;
    localparam FEDP_LATENCY = FMUL_LATENCY + FACC_LATENCY;
`else // VX_CFG_TCU_TYPE_TFR
    localparam FMUL_LATENCY = 1;
    localparam FALN_LATENCY = 1;
    localparam FACC_LATENCY = 1;
    localparam FRND_LATENCY = 1;
    localparam FEDP_LATENCY = FMUL_LATENCY + FALN_LATENCY + FACC_LATENCY + FRND_LATENCY;
`endif

    localparam PIPE_LATENCY = FEDP_LATENCY + 1;
    localparam MDATA_QUEUE_DEPTH = 1 << $clog2(PIPE_LATENCY);

    localparam LG_A_BS    = $clog2(TCU_A_BLOCK_SIZE);
    localparam LG_B_BS    = $clog2(TCU_B_BLOCK_SIZE);
    localparam OFF_W      = $clog2(TCU_BLOCK_CAP);
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    localparam LG_WG_B_BS = $clog2(TCU_WG_B_BLOCK_SIZE);
    localparam WG_B_OFF_W = $clog2(TCU_WG_RS2_WIDTH);
`endif

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    localparam LG_B_BS_SP = $clog2(TCU_B_BLOCK_SIZE_SP);
    wire is_sparse = (execute_if.data.op_type == INST_TCU_WMMA_SP)
              `ifdef VX_CFG_TCU_WGMMA_ENABLE
                 || (execute_if.data.op_type == INST_TCU_WGMMA_SP)
              `endif
                 ;
`endif

`ifdef VX_CFG_TCU_MX_ENABLE
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire mx_is_sparse = is_sparse;
`else
    wire mx_is_sparse = 1'b0;
`endif
    localparam FEDP_SF = TCU_MX_MAX_SF;
`else
    localparam FEDP_SF = 1;
    `UNUSED_PARAM (FEDP_SF)
`endif

    // -----------------------------------------------------------------------
    // WGMMA / WMMA abstraction layer
    // -----------------------------------------------------------------------
    // All WGMMA-vs-WMMA runtime differences are resolved here behind a
    // common interface.  Downstream code uses only these wires and never
    // references tbuf_* or is_wgmma directly.

    wire [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] rs1_data;
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] rs2_data;
`else
    wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] rs2_data;
`endif
    wire exe_ready_extra; // additional ready gating (tbuf_ready)

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire is_wgmma = (execute_if.data.op_type == INST_TCU_WGMMA)
              `ifdef VX_CFG_TCU_SPARSE_ENABLE
                 || (execute_if.data.op_type == INST_TCU_WGMMA_SP)
              `endif
                 ;
    wire wg_a_smem = execute_if.data.op_args.tcu.a_from_smem;
`ifdef TCU_TMEM_ENABLE
    wire is_umma = (execute_if.data.op_type == INST_TCU_UMMA);
    wire wg_or_umma = is_wgmma || is_umma;
    wire wg_or_umma_a_smem = is_umma || wg_a_smem;
`endif

    // A/B operand mux: tile buffer (smem) or register file. The
    // RF-side rs2_data is NUM_THREADS lanes wide; the WGMMA bbuf can be
    // wider (TCU_WG_RS2_WIDTH lanes). Pad/truncate to the wgmma width on
    // the false branch so both arms match TCU_WG_RS2_WIDTH * XLEN bits.
    localparam WG_RS1_BITS = TCU_WG_A_DATA_SIZE * `VX_CFG_XLEN;
    localparam WG_RS2_BITS = TCU_WG_RS2_WIDTH * `VX_CFG_XLEN;
    wire [WG_RS1_BITS-1:0] rs1_data_rf = WG_RS1_BITS'(execute_if.data.rs1_data);
    wire [WG_RS2_BITS-1:0] rs2_data_rf = WG_RS2_BITS'(execute_if.data.rs2_data);
`ifdef TCU_TMEM_ENABLE
    assign rs1_data = ((is_wgmma && wg_a_smem) || is_umma) ? tbuf_rs1_data : rs1_data_rf;
    assign rs2_data = (is_wgmma || is_umma) ? tbuf_rs2_data : rs2_data_rf;
`else
    assign rs1_data = (is_wgmma && wg_a_smem) ? tbuf_rs1_data : rs1_data_rf;
    assign rs2_data = is_wgmma ? tbuf_rs2_data : rs2_data_rf;
`endif

  `ifdef VX_CFG_TCU_SPARSE_ENABLE
    // Sparse metadata lives in VX_tcu_sp_meta SRAM, preloaded via TCU_LD.
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block = wmma_sp_meta;
  `endif

`ifdef TCU_TMEM_ENABLE
    assign exe_ready_extra = ~(is_wgmma || is_umma) || tbuf_ready;
`else
    assign exe_ready_extra = ~is_wgmma || tbuf_ready;
`endif
`else
    assign rs1_data = execute_if.data.rs1_data;
    assign rs2_data = execute_if.data.rs2_data;
  `ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block = wmma_sp_meta;
  `endif
    assign exe_ready_extra = 1'b1;
`endif

    // Widths match tcu_args_t: step_m/step_k only ever hold 0 or 1
    // (m_steps fixed at 2, k_steps at most 2) but are kept wider for 
    // headroom; step_n needs up to 6 bits for UMMA's NRC=128 case
    // (n_steps=64)
    wire [2:0] step_m = execute_if.data.op_args.tcu.step_m;
    wire [5:0] step_n = execute_if.data.op_args.tcu.step_n;
    wire [2:0] step_k = execute_if.data.op_args.tcu.step_k;

    wire [4:0] fmt_s = execute_if.data.op_args.tcu.fmt_s;
    wire [4:0] fmt_d = execute_if.data.op_args.tcu.fmt_d;

`ifdef TCU_TMEM_ENABLE
    // TMEM addressing for UMMA. cta_rank/handle are warp-uniform (handle is
    // the base column returned by a prior TMEM_ALLOC), lane/col additionally
    // depend on the grid cell (i,j), computed per-cell alongside c_val.
    //
    // rs3 (the handle register, x12/a2) is only populated by the register
    // file on the macro-op's first micro-op. Must be latched on the first uop and
    // reused for the rest of the macro-op.
    reg [31:0] umma_handle_r;
    always @(posedge clk) begin
        if (execute_fire && is_umma && execute_if.data.op_args.tcu.is_first_uop) begin
            umma_handle_r <= execute_if.data.rs3_data[0];
        end
    end
    wire [31:0] umma_handle = execute_if.data.op_args.tcu.is_first_uop
        ? execute_if.data.rs3_data[0] : umma_handle_r;
`endif

    wire execute_fire = execute_if.valid && execute_if.ready;
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire is_wgmma_setup = is_wgmma && !execute_if.data.header.wb;
`else
    wire is_wgmma_setup = 1'b0;
`endif
    wire setup_enqueue = execute_fire && is_wgmma_setup;
    wire fedp_enqueue  = execute_fire && !is_wgmma_setup;

    // -----------------------------------------------------------------------
    // Sparse metadata: VX_tcu_sp_meta (for WMMA_SP) + optional tile-buffer mux
    // -----------------------------------------------------------------------

    tcu_header_t mdata_queue_in;
    always_comb begin
        mdata_queue_in = execute_if.data.header;
    end

    `UNUSED_VAR ({step_m, step_n, step_k, fmt_s, fmt_d, execute_if.data});

`ifdef VX_TCU_LD_TRACE
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // META_RD trace: logs vld_meta_block at FEDP consume time.
    // Format: META_RD,wid,step_m,step_k,wg_bank,word_lo32
    wire trc_is_sp = (execute_if.data.op_type == INST_OP_BITS'(INST_TCU_WMMA_SP))
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
                  || (execute_if.data.op_type == INST_OP_BITS'(INST_TCU_WGMMA_SP))
        `endif
                  ;
    wire [3:0] trc_wg_bank = ((TCU_K_STEPS > 2) ? (step_m << 1) : step_m) | step_k;
    always @(posedge clk) begin
        if (execute_fire && trc_is_sp) begin
            $write("META_RD,%0d,%0d,%0d,%0d,0x%08h\n",
                execute_if.data.header.wid, step_m, step_k, trc_wg_bank,
                vld_meta_block[31:0]);
        end
    end
`endif
`endif

    // -----------------------------------------------------------------------
    // Pipeline control
    // -----------------------------------------------------------------------

    wire mdata_queue_full;

    wire fedp_enable, fedp_done;

    reg setup_valid_r;
    tcu_header_t setup_header_r;
    tcu_header_t mdata_queue_out;

    wire setup_result_fire = setup_valid_r && result_if.ready;
    wire fedp_result_fire  = fedp_done && result_if.ready && !setup_valid_r;

    always @(posedge clk) begin
        if (reset) begin
            setup_valid_r <= 1'b0;
        end else begin
            if (setup_result_fire) begin
                setup_valid_r <= 1'b0;
            end
            if (setup_enqueue) begin
                setup_valid_r <= 1'b1;
                setup_header_r <= execute_if.data.header;
            end
        end
    end

    reg [PIPE_LATENCY-1:0] fedp_delay_pipe;
    always @(posedge clk) begin
        if (reset) begin
            fedp_delay_pipe <= '0;
        end else begin
            if (fedp_enable) begin
                fedp_delay_pipe <= fedp_delay_pipe >> 1;
            end
            if (fedp_enqueue) begin
                fedp_delay_pipe[PIPE_LATENCY-1] <= 1;
            end
        end
    end

    assign fedp_done        = fedp_delay_pipe[0];
`ifdef TCU_TMEM_ENABLE
    // A retiring UMMA op must not be reported as done until its TMEM write
    // has actually won a bank's write port
    wire tmem_wr_ok = ~tmem_addr_pipe[0].valid || tmem_wr_grant;
`else
    wire tmem_wr_ok = 1'b1;
`endif
    assign result_if.valid  = setup_valid_r || (fedp_done && tmem_wr_ok);
    assign fedp_enable      = (~fedp_done || fedp_result_fire) && tmem_wr_ok;

`ifdef TCU_TMEM_ENABLE
    // Tile-origin address for the incoming uop (pre +i/+j offset)
    wire [TCU_TMEM_LANE_BITS-1:0] umma_lane_base = TCU_TMEM_LANE_BITS'(cta_rank) * TCU_TMEM_LANE_BITS'(TCU_WG_TILE_M)
                                                  + TCU_TMEM_LANE_BITS'(step_m) * TCU_TMEM_LANE_BITS'(TCU_TC_M);
    wire [TCU_TMEM_COL_BITS-1:0]  umma_col_base  = TCU_TMEM_COL_BITS'(umma_handle)
                                                  + TCU_TMEM_COL_BITS'(step_n) * TCU_TMEM_COL_BITS'(TCU_TC_N);

    // Read request to VX_tcu_tmem: this op's tile origin, valid whenever a
    // new UMMA op is admitted. tmem_rd_valid marks whether there's a
    // pending request this cycle. 
    // Gated on ~umma_hazard: with registered read the fetch and admission
    // are different cycles. Submitting the request while a hazard is live
    // would still win bank arbitration and latch stale data that cycle,
    // which could get captured once the hazard clears one cycle later
    // without refetching.
    // Also gated on ~tmem_rd_grant: with registered read same request
    // would resubmit and could win bank arbitration a second time while
    // its first grant is still being consumed, producing stale extra
    // grant
    assign tmem_rd_valid     = is_umma && execute_if.valid && ~umma_hazard && ~tmem_rd_grant;
    assign tmem_rd_lane_base = umma_lane_base;
    assign tmem_rd_col_base  = umma_col_base;

    // TMEM lanes are sized to one warpgroup's width (TCU_TMEM_LANES =
    // NUM_TCU_BLOCKS * NUM_THREADS, see VX_tcu_pkg.sv), so this design
    // assumes CTA size == warpgroup size (NUM_TCU_BLOCKS)
    `RUNTIME_ASSERT (~execute_fire || !is_umma || (32'(cta_rank) < 32'(`VX_CFG_NUM_TCU_BLOCKS)),
        ("%s: cta_rank %0d exceeds NUM_TCU_BLOCKS (%0d) — CTA is larger than one warpgroup, violating TMEM's CTA-size==warpgroup-size assumption",
         INSTANCE_ID, cta_rank, `VX_CFG_NUM_TCU_BLOCKS))

    // RAW-hazard interlock. k-outer UMMA loop order only avoids revisiting
    // the same TMEM tile before its prior write retires when
    // n_steps*m_steps >= FEDP_LATENCY+1. This tracks every in-flight
    // write's address explicitly and stalls admission of a colliding new
    // uop so correctness doesn't depend on that bound holding. Shifted
    // in lockstep with fedp_delay_pipe, so position 0 always corresponds
    // to the same uop fedp_done reports on.
    typedef struct packed {
        logic                          valid;
        logic [TCU_TMEM_LANE_BITS-1:0] lane_base;
        logic [TCU_TMEM_COL_BITS-1:0]  col_base;
    } tmem_addr_t;

    tmem_addr_t tmem_addr_pipe [PIPE_LATENCY];

    always @(posedge clk) begin
        if (reset) begin
            for (int p = 0; p < PIPE_LATENCY; ++p)
                tmem_addr_pipe[p] <= '0;
        end else if (fedp_enable) begin
            for (int p = 0; p < PIPE_LATENCY-1; ++p)
                tmem_addr_pipe[p] <= tmem_addr_pipe[p+1];
            tmem_addr_pipe[PIPE_LATENCY-1].valid     <= fedp_enqueue && is_umma;
            tmem_addr_pipe[PIPE_LATENCY-1].lane_base <= umma_lane_base;
            tmem_addr_pipe[PIPE_LATENCY-1].col_base  <= umma_col_base;
        end
    end

    logic [PIPE_LATENCY-1:0] umma_hazard_match;
    for (genvar p = 0; p < PIPE_LATENCY; ++p) begin : g_umma_hazard
        assign umma_hazard_match[p] = tmem_addr_pipe[p].valid
                                    && (tmem_addr_pipe[p].lane_base == umma_lane_base)
                                    && (tmem_addr_pipe[p].col_base  == umma_col_base);
    end
    wire umma_hazard = is_umma && (|umma_hazard_match);

    // Bank-port stall: this op's TMEM read lost arbitration to another
    // block targeting the same bank this cycle. Retries next cycle.
    wire umma_rd_stall = is_umma && ~tmem_rd_grant;

    // TMEM write fires the same cycle fedp_done does, for whichever uop is
    // now at position 0. tmem_addr_pipe[0].valid is 0 for non-UMMA uops.
    // This is a request. VX_tcu_tmem only actually writes if it grants
    // the bank's write port (tmem_wr_grant)
    assign tmem_wr_valid     = fedp_done && tmem_addr_pipe[0].valid;
    assign tmem_wr_lane_base = tmem_addr_pipe[0].lane_base;
    assign tmem_wr_col_base  = tmem_addr_pipe[0].col_base;

    assign execute_if.ready = is_wgmma_setup
                            ? ((~setup_valid_r || result_if.ready) && exe_ready_extra)
                            : (~mdata_queue_full && fedp_enable && exe_ready_extra && ~umma_hazard && ~umma_rd_stall);
`else
    assign execute_if.ready = is_wgmma_setup
                            ? ((~setup_valid_r || result_if.ready) && exe_ready_extra)
                            : (~mdata_queue_full && fedp_enable && exe_ready_extra);
`endif

    wire mdata_push = fedp_enqueue;

    VX_fifo_queue #(
        .DATAW ($bits(tcu_header_t)),
        .DEPTH (MDATA_QUEUE_DEPTH),
        .OUT_REG (1)
    ) mdata_queue (
        .clk    (clk),
        .reset  (reset),
        .push   (mdata_push),
        .pop    (fedp_result_fire),
        .data_in(mdata_queue_in),
        .data_out(mdata_queue_out),
        `UNUSED_PIN(empty),
        `UNUSED_PIN(alm_empty),
        .full   (mdata_queue_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    assign result_if.data.header = setup_valid_r ? setup_header_r : mdata_queue_out;

    // -----------------------------------------------------------------------
    // Operand offset computation
    // -----------------------------------------------------------------------

    wire [OFF_W-1:0] a_off = (OFF_W'(step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;
    wire [OFF_W-1:0] b_off_wm;
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    assign b_off_wm = is_sparse
        ? (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS_SP-1)) << LG_B_BS_SP
        : (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`else
    assign b_off_wm = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`endif
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [WG_B_OFF_W-1:0] b_off_wg =
        (WG_B_OFF_W'(step_n) & WG_B_OFF_W'(TCU_WG_B_SUB_BLOCKS-1)) << LG_WG_B_BS;
`endif

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] wmma_sp_meta;
`endif
`ifdef VX_CFG_TCU_MX_ENABLE
    wire [TCU_BLOCK_CAP-1:0][31:0] mx_meta_a;
    wire [TCU_BLOCK_CAP-1:0][31:0] mx_meta_b;
`endif

`ifdef TCU_META_ENABLE
    VX_tcu_meta #(
        .INSTANCE_ID (INSTANCE_ID)
    ) tcu_meta (
        .clk    (clk),
        .reset  (reset),
        .wr_en  (ext_meta_wr_en),
        .wr_wid (ext_meta_wr_wid),
        .wr_idx (ext_meta_wr_idx),
        .wr_data(ext_meta_wr_data),
        .rd_wid (execute_if.data.header.wid)
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        , .step_m   (step_m)
        , .step_k   (step_k)
        , .vld_block(wmma_sp_meta)
    `endif
    `ifdef VX_CFG_TCU_MX_ENABLE
        , .meta_a   (mx_meta_a)
        , .meta_b   (mx_meta_b)
    `endif
    );
`endif

`ifdef VX_CFG_TCU_MX_ENABLE

    localparam MX_MAX_MN = TCU_TILE_M > TCU_TILE_N ? TCU_TILE_M : TCU_TILE_N;
    localparam MX_IDX_W = $clog2(MX_MAX_MN);
    localparam MX_TILE_K_MAX = `MAX(TCU_TILE_K, TCU_WG_K_STEPS * TCU_WG_FEDP_K);
    localparam MX_K_IDX_W = `LOG2UP(MX_TILE_K_MAX * TCU_MAX_ELT_RATIO);
    localparam MX_SCALE_BLOCKS_MAX = mx_scale_blocks_k_words(TCU_NVFP4_ID, MX_TILE_K_MAX);
    localparam MX_SCALE_IDX_W = $clog2(MX_MAX_MN * MX_SCALE_BLOCKS_MAX);

    function automatic [7:0] mx_scale_at(
        input logic [TCU_BLOCK_CAP-1:0][31:0] meta,
        input logic [4:0] fmt,
        input logic [MX_SCALE_IDX_W-1:0] scale_blocks_k,
        input logic [MX_IDX_W-1:0] mn_idx,
        input logic [MX_K_IDX_W-1:0] k_base_idx
    );
        logic [MX_SCALE_IDX_W-1:0] scale_k;
        logic [MX_SCALE_IDX_W-1:0] scale_idx;
        logic [`LOG2UP(TCU_BLOCK_CAP)-1:0] word_idx;
        logic [1:0] byte_idx;
        begin
            scale_k = MX_SCALE_IDX_W'(k_base_idx / mx_scale_block_size(fmt));
            scale_idx = MX_SCALE_IDX_W'(mn_idx) * scale_blocks_k + scale_k;
            word_idx = `LOG2UP(TCU_BLOCK_CAP)'(scale_idx >> 2);
            byte_idx = scale_idx[1:0];
            mx_scale_at = meta[word_idx][byte_idx * 8 +: 8];
        end
    endfunction

    wire [TCU_TC_M-1:0][FEDP_SF-1:0][7:0] mx_sf_a;
    wire [TCU_TC_N-1:0][FEDP_SF-1:0][7:0] mx_sf_b;
    wire [3:0] mx_elems_per_word = 4'(32 / tcu_fmt_width(fmt_s));
    wire [MX_SCALE_IDX_W-1:0] mx_scale_blocks_k_eff =
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        is_wgmma ? MX_SCALE_IDX_W'(mx_scale_blocks_k_words(fmt_s, TCU_WG_K_STEPS * TCU_WG_FEDP_K))
                 : MX_SCALE_IDX_W'(mx_scale_blocks_k_words(fmt_s, TCU_TILE_K));
    `else
        MX_SCALE_IDX_W'(mx_scale_blocks_k_words(fmt_s, TCU_TILE_K));
    `endif
    wire [MX_K_IDX_W:0] mx_uop_k_words =
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        is_wgmma ? (MX_K_IDX_W+1)'(TCU_WG_FEDP_K) : (MX_K_IDX_W+1)'(TCU_TC_K);
    `else
        (MX_K_IDX_W+1)'(TCU_TC_K);
    `endif
    wire [MX_K_IDX_W:0] mx_uop_k_elems = (MX_K_IDX_W+1)'(
        mx_uop_k_words * (MX_K_IDX_W+1)'(mx_elems_per_word)
        * (MX_K_IDX_W+1)'(mx_is_sparse ? 2 : 1));
    wire [MX_K_IDX_W:0] mx_fedp_k_elems = (MX_K_IDX_W+1)'(
        (MX_K_IDX_W+1)'(FEDP_K) * (MX_K_IDX_W+1)'(mx_elems_per_word)
        * (MX_K_IDX_W+1)'(mx_is_sparse ? 2 : 1));
    wire [MX_K_IDX_W-1:0] mx_k_base_idx = MX_K_IDX_W'(step_k * mx_uop_k_elems);

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_mx_sf_a_i
        wire [MX_IDX_W-1:0] mx_a_idx = MX_IDX_W'(step_m) * MX_IDX_W'(TCU_TC_M) + MX_IDX_W'(i);
        for (genvar s = 0; s < FEDP_SF; ++s) begin : g_s
            wire [MX_K_IDX_W-1:0] mx_k_idx = mx_k_base_idx + MX_K_IDX_W'((s * mx_fedp_k_elems) / FEDP_SF);
            assign mx_sf_a[i][s] = mx_scale_at(mx_meta_a, fmt_s, mx_scale_blocks_k_eff, mx_a_idx, mx_k_idx);
        end
    end

    for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_mx_sf_b_j
        wire [MX_IDX_W-1:0] mx_b_idx = MX_IDX_W'(step_n) * MX_IDX_W'(TCU_TC_N) + MX_IDX_W'(j);
        for (genvar s = 0; s < FEDP_SF; ++s) begin : g_s
            wire [MX_K_IDX_W-1:0] mx_k_idx = mx_k_base_idx + MX_K_IDX_W'((s * mx_fedp_k_elems) / FEDP_SF);
            assign mx_sf_b[j][s] = mx_scale_at(mx_meta_b, fmt_s, mx_scale_blocks_k_eff, mx_b_idx, mx_k_idx);
        end
    end
`endif

    // -----------------------------------------------------------------------
    // FEDP grid: TCU_TC_M × TCU_TC_N compute elements
    // -----------------------------------------------------------------------

    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] d_val;

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
        for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            wire [FEDP_K-1:0][31:0] a_row, b_col, b_col_dense;
            wire [TCU_TC_K-1:0][31:0] b_col_sparse, b_col_1, b_col_2;
        `else
            wire [FEDP_K-1:0][31:0] a_row, b_col;
        `endif
        `ifdef VX_CFG_TCU_MX_ENABLE
            wire [FEDP_SF-1:0][7:0] sf_a = mx_sf_a[i];
            wire [FEDP_SF-1:0][7:0] sf_b = mx_sf_b[j];
        `endif
            for (genvar k_idx = 0; k_idx < FEDP_K; ++k_idx) begin : g_slice_assign
            `ifdef VX_CFG_TCU_WGMMA_ENABLE
                localparam int WG_B_IDX = j * TCU_WG_FEDP_K + k_idx;
            `endif
                if (k_idx < TCU_TC_K) begin : g_lo
                `ifdef VX_CFG_TCU_WGMMA_ENABLE
                    wire [31:0] a_wgmma_smem = 32'(rs1_data[i * TCU_WG_FEDP_K + k_idx]);
                    wire [31:0] a_wgmma_reg  = 32'(execute_if.data.rs1_data[i * TCU_TC_K + k_idx]);
                `ifdef TCU_TMEM_ENABLE
                    assign a_row[k_idx] = wg_or_umma
                        ? (wg_or_umma_a_smem ? a_wgmma_smem : a_wgmma_reg)
                        : 32'(execute_if.data.rs1_data[a_off + i * TCU_TC_K + k_idx]);
                `else
                    assign a_row[k_idx] = is_wgmma
                        ? (wg_a_smem ? a_wgmma_smem : a_wgmma_reg)
                        : 32'(execute_if.data.rs1_data[a_off + i * TCU_TC_K + k_idx]);
                `endif
                `else
                    assign a_row[k_idx] = 32'(rs1_data[a_off + i * TCU_TC_K + k_idx]);
                `endif
                `ifdef VX_CFG_TCU_SPARSE_ENABLE
                    assign b_col_dense[k_idx] =
                    `ifdef VX_CFG_TCU_WGMMA_ENABLE
                      `ifdef TCU_TMEM_ENABLE
                        wg_or_umma ? 32'(rs2_data[int'(b_off_wg) + WG_B_IDX]) :
                      `else
                        is_wgmma ? 32'(rs2_data[int'(b_off_wg) + WG_B_IDX]) :
                      `endif
                    `endif
                        32'(rs2_data[b_off_wm + j * TCU_TC_K + k_idx]);
                `else
                    assign b_col[k_idx] =
                    `ifdef VX_CFG_TCU_WGMMA_ENABLE
                      `ifdef TCU_TMEM_ENABLE
                        wg_or_umma ? 32'(rs2_data[int'(b_off_wg) + WG_B_IDX]) :
                      `else
                        is_wgmma ? 32'(rs2_data[int'(b_off_wg) + WG_B_IDX]) :
                      `endif
                    `endif
                        32'(rs2_data[b_off_wm + j * TCU_TC_K + k_idx]);
                `endif
                end else begin : g_hi
                `ifdef VX_CFG_TCU_WGMMA_ENABLE
                    wire [31:0] a_wgmma_smem = 32'(rs1_data[i * TCU_WG_FEDP_K + k_idx]);
                    wire [31:0] a_wgmma_reg =
                    `ifdef VX_CFG_TCU_FEDP2K
                        `ifdef VX_CFG_TCU_SPARSE_ENABLE
                            is_sparse ? 32'b0 :
                        `endif
                        32'(execute_if.data.rs2_data[i * TCU_TC_K + (k_idx - TCU_TC_K)]);
                    `else
                        32'b0;
                    `endif
                `ifdef TCU_TMEM_ENABLE
                    assign a_row[k_idx] = (is_umma || (is_wgmma
                        `ifdef VX_CFG_TCU_SPARSE_ENABLE
                            && !is_sparse
                        `endif
                        )) ? (wg_or_umma_a_smem ? a_wgmma_smem : a_wgmma_reg) : 32'b0;
                `else
                    assign a_row[k_idx] = (is_wgmma
                        `ifdef VX_CFG_TCU_SPARSE_ENABLE
                            && !is_sparse
                        `endif
                        ) ? (wg_a_smem ? a_wgmma_smem : a_wgmma_reg) : 32'b0;
                `endif
                `else
                    assign a_row[k_idx] = 32'b0;
                `endif
                `ifdef VX_CFG_TCU_SPARSE_ENABLE
                    assign b_col_dense[k_idx] =
                    `ifdef VX_CFG_TCU_WGMMA_ENABLE
                      `ifdef TCU_TMEM_ENABLE
                        (is_umma || (is_wgmma && !is_sparse)) ? 32'(rs2_data[int'(b_off_wg) + WG_B_IDX]) :
                      `else
                        (is_wgmma && !is_sparse) ? 32'(rs2_data[int'(b_off_wg) + WG_B_IDX]) :
                      `endif
                    `endif
                        32'b0;
                `else
                    assign b_col[k_idx] =
                    `ifdef VX_CFG_TCU_WGMMA_ENABLE
                      `ifdef TCU_TMEM_ENABLE
                        wg_or_umma ? 32'(rs2_data[int'(b_off_wg) + WG_B_IDX]) :
                      `else
                        is_wgmma ? 32'(rs2_data[int'(b_off_wg) + WG_B_IDX]) :
                      `endif
                    `endif
                        32'b0;
                `endif
                end
            end

        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            for (genvar k_idx = 0; k_idx < TCU_TC_K; ++k_idx) begin : g_sparse_slice_assign
                localparam J_SP = SYM_SPARSE ? (j % (TCU_TC_N / 2)) : j;
                // rs2_data sparse-pair layout differs by op:
                //   WGMMA_SP: source is tbuf (shared mem), K-major →
                //     idx = k_idx*(TC_N*2) + J_SP*2 + lane
                //   WMMA_SP : source is the register file, J-major →
                //     idx = J_SP*(TC_K*2) + k_idx*2 + lane
                // The two layouts are incompatible; separate formulas are required.
            `ifdef VX_CFG_TCU_WGMMA_ENABLE
                wire [31:0] b_col_1_wg = 32'(rs2_data[k_idx * TCU_TC_N * 2 + J_SP * 2]);
                wire [31:0] b_col_2_wg = 32'(rs2_data[k_idx * TCU_TC_N * 2 + J_SP * 2 + 1]);
                wire [31:0] b_col_1_wm = 32'(rs2_data[b_off_wm + J_SP * TCU_TC_K * 2 + k_idx * 2]);
                wire [31:0] b_col_2_wm = 32'(rs2_data[b_off_wm + J_SP * TCU_TC_K * 2 + k_idx * 2 + 1]);
                assign b_col_1[k_idx] = is_wgmma ? b_col_1_wg : b_col_1_wm;
                assign b_col_2[k_idx] = is_wgmma ? b_col_2_wg : b_col_2_wm;
            `else
                assign b_col_1[k_idx] = 32'(rs2_data[b_off_wm + J_SP * TCU_TC_K * 2 + k_idx * 2]);
                assign b_col_2[k_idx] = 32'(rs2_data[b_off_wm + J_SP * TCU_TC_K * 2 + k_idx * 2 + 1]);
            `endif
            end
        `endif

        `ifdef TCU_TMEM_ENABLE
            // tmem_rd_data[i][j] is VX_tcu_tmem's response tile for this
            // op's (tmem_rd_lane_base, tmem_rd_col_base) request
            wire [31:0] c_val = is_umma ? tmem_rd_data[i][j]
                                        : 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);
        `else
            wire [31:0] c_val = 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);
        `endif

        `ifdef VX_CFG_TCU_SPARSE_ENABLE
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
            for (genvar k_idx = 0; k_idx < FEDP_K; ++k_idx) begin : g_sparse_b_select
                if (k_idx < TCU_TC_K) begin : g_lo
                    assign b_col[k_idx] = is_sparse ? b_col_sparse[k_idx] : b_col_dense[k_idx];
                end else begin : g_hi
                    assign b_col[k_idx] = is_sparse ? 32'b0 : b_col_dense[k_idx];
                end
            end

        `ifdef VX_TCU_LD_TRACE
            // GATHER trace: GATHER,wid,step_m,step_n,i,k,bword0,bword1,lo,hi,gathered
            // One line per (i, j, k_idx); emitted only for sparse ops.
            always @(posedge clk) begin
                if (execute_fire && is_sparse) begin
                    for (int kk = 0; kk < TCU_TC_K; ++kk) begin
                        $write("GATHER,%0d,%0d,%0d,%0d,%0d,0x%08h,0x%08h,?,?,0x%08h\n",
                            execute_if.data.header.wid, step_m, step_n,
                            i, j*TCU_TC_K + kk,
                            b_col_1[kk], b_col_2[kk], b_col_sparse[kk]);
                    end
                end
            end
        `endif
        `endif

        // Dual-side sparse lane mask
        `ifdef VX_CFG_TCU_TYPE_TFR
            wire [TCU_MAX_INPUTS-1:0] vld_mask_r;
        `ifdef VX_CFG_TCU_DSM_ENABLE
            wire [TCU_MAX_INPUTS-1:0] vld_mask;
            VX_tcu_dsm #(
                .N (TCU_TC_K)
            ) dual_sparse_mask (
                .fmt_s    (fmt_s),
                .a_row    (a_row),
                .b_col    (b_col),
                .vld_mask (vld_mask)
            );
            VX_pipe_register #(
                .DATAW (TCU_MAX_INPUTS)
            ) pipe_vld_mask (
                .clk      (clk),
                .reset    (reset),
                .enable   (fedp_enable),
                .data_in  (vld_mask),
                .data_out (vld_mask_r)
            );
        `else
            assign vld_mask_r = '1;
        `endif
        `endif

            wire [4:0] fmt_s_r, fmt_d_r;
            wire [FEDP_K-1:0][31:0] a_row_r, b_col_r;
        `ifdef VX_CFG_TCU_MX_ENABLE
            wire [FEDP_SF-1:0][7:0] sf_a_r, sf_b_r;
        `endif
            wire [31:0] c_val_r;

        `ifdef VX_CFG_TCU_MX_ENABLE
            VX_pipe_register #(
                .DATAW (32 + 5 + 5 + FEDP_K * 32 + FEDP_K * 32 + 2 * FEDP_SF * 8)
            ) pipe_fedp (
                .clk      (clk),
                .reset    (reset),
                .enable   (fedp_enable),
                .data_in  ({c_val,   sf_b,   sf_a,   fmt_s,   fmt_d,   b_col,   a_row}),
                .data_out ({c_val_r, sf_b_r, sf_a_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r})
            );
        `else
            VX_pipe_register #(
                .DATAW (32 + 5 + 5 + FEDP_K * 32 + FEDP_K * 32)
            ) pipe_fedp (
                .clk      (clk),
                .reset    (reset),
                .enable   (fedp_enable),
                .data_in  ({c_val,   fmt_s,   fmt_d,   b_col,   a_row}),
                .data_out ({c_val_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r})
            );
        `endif

        `ifdef VX_CFG_TCU_TYPE_DPI
            VX_tcu_fedp_dpi #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (FEDP_K),
                .SF (FEDP_SF)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_col(b_col_r),
            `ifdef VX_CFG_TCU_MX_ENABLE
                .sf_a  (sf_a_r),
                .sf_b  (sf_b_r),
            `endif
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif VX_CFG_TCU_TYPE_BHF
            VX_tcu_fedp_bhf #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (FEDP_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_col(b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif VX_CFG_TCU_TYPE_FPNEW
            VX_tcu_fedp_fpnew #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (FEDP_K)
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
        `elsif VX_CFG_TCU_TYPE_TFR
            VX_tcu_fedp_tfr #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (FEDP_K),
                .SF (FEDP_SF)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .vld_mask(vld_mask_r),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
            `ifdef VX_CFG_TCU_MX_ENABLE
                .sf_a  (sf_a_r),
                .sf_b  (sf_b_r),
            `endif
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif VX_CFG_TCU_TYPE_DSP
            VX_tcu_fedp_dsp #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (FEDP_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_col(b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `endif

            // NaN-box the fp32 result for XLEN=64: upper 32 bits must be all-1s per RVF spec.
            if (`VX_CFG_XLEN > 32) begin : g_result_nanbox
                assign result_if.data.data[i * TCU_TC_N + j] = {32'hffffffff, d_val[i][j]};
            end else begin : g_result_passthrough
                assign result_if.data.data[i * TCU_TC_N + j] = d_val[i][j];
            end

        `ifdef TCU_TMEM_ENABLE
            // UMMA's real destination — result_if.data.data[i*TCU_TC_N+j]
            // above is harmless but unused for UMMA
            assign tmem_wr_data[i][j] = d_val[i][j];
        `endif

        `ifdef DBG_TRACE_TCU
            always @(posedge clk) begin
                if (execute_if.valid && execute_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-enq: wid=%0d, cta_id=%0d, i=%0d, j=%0d, m=%0d, n=%0d, a_row=", $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.cta_id, i, j, step_m, step_n))
                    `TRACE_ARRAY1D(2, "0x%0h", a_row, FEDP_K)
                    `TRACE(3, (", b_col="));
                    `TRACE_ARRAY1D(2, "0x%0h", b_col, FEDP_K)
                    `TRACE(3, (", c_val=0x%0h (#%0d)\n", c_val, execute_if.data.header.uuid));
                end
                if (result_if.valid && result_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-deq: wid=%0d, cta_id=%0d, i=%0d, j=%0d, d_val=0x%0h (#%0d)\n", $time, INSTANCE_ID, result_if.data.header.wid, result_if.data.header.cta_id, i, j, d_val[i][j], result_if.data.header.uuid));
                end
            end
        `endif // DBG_TRACE_TCU
        end
    end

endmodule
