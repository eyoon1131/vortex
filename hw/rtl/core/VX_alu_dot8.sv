`include "VX_define.vh"

module VX_alu_dot8 import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1
) (
    input wire          clk,
    input wire          reset,

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_result_if.master result_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam PID_BITS = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH = `UP(PID_BITS);
    localparam TAG_WIDTH = UUID_WIDTH + NW_WIDTH + NUM_LANES + PC_BITS + 1 + NUM_REGS_BITS + PID_WIDTH + 1 + 1;
    localparam LATENCY_DOT8 = `LATENCY_DOT8;
    localparam PE_RATIO = 1;
    localparam NUM_PES = `UP(NUM_LANES / PE_RATIO);

    `UNUSED_VAR (execute_if.data.op_type)
    `UNUSED_VAR (execute_if.data.op_args)
    `UNUSED_VAR (execute_if.data.rs3_data)

    wire pe_enable;
    wire [NUM_LANES-1:0][2*`XLEN-1:0] data_in;
    wire [NUM_PES-1:0][2*`XLEN-1:0] pe_data_in;
    wire [NUM_PES-1:0][`XLEN-1:0] pe_data_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_data_in
        assign data_in[i][0 +: `XLEN] = execute_if.data.rs1_data[i];
        assign data_in[i][`XLEN +: `XLEN] = execute_if.data.rs2_data[i];
    end

    // PEs time-multiplexing
    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (LATENCY_DOT8),
        .DATA_IN_WIDTH (2 * `XLEN),
        .DATA_OUT_WIDTH (`XLEN),
        .TAG_WIDTH  (TAG_WIDTH),
        .PE_REG     (1)
    ) pe_serializer (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (execute_if.valid),
        .data_in    (data_in),
        .tag_in     ({
            execute_if.data.uuid,
            execute_if.data.wid,
            execute_if.data.tmask,
            execute_if.data.PC,
            execute_if.data.wb,
            execute_if.data.rd,
            execute_if.data.pid,
            execute_if.data.sop,
            execute_if.data.eop
        }),
        .ready_in   (execute_if.ready),
        .pe_enable  (pe_enable),
        .pe_data_in (pe_data_out),
        .pe_data_out(pe_data_in),
        .valid_out  (result_if.valid),
        .data_out   (result_if.data.data),
        .tag_out    ({
            result_if.data.uuid,
            result_if.data.wid,
            result_if.data.tmask,
            result_if.data.PC,
            result_if.data.wb,
            result_if.data.rd,
            result_if.data.pid,
            result_if.data.sop,
            result_if.data.eop
        }),
        .ready_out  (result_if.ready)
    );

    // PEs instancing
    for (genvar i = 0; i < NUM_PES; ++i) begin : g_PEs
        wire [`XLEN-1:0] a = pe_data_in[i][0 +: `XLEN];
        wire [`XLEN-1:0] b = pe_data_in[i][`XLEN +: `XLEN];
        wire [31:0] c, result;

        wire [31:0] a_32 = a[31:0];
        wire[31:0] a_unused = a[63:32];
        wire [31:0] b_32 = b[31:0];
        wire[31:0] b_unused = b[63:32];
        reg signed[7:0] a_byte, b_byte;
        reg signed [31:0] sum;
        always @(*) begin
            sum = 0;
            for (int j = 0; j < 4; ++j) begin
                a_byte = a_32[8 * j +: 8];
                b_byte = b_32[8 * j +: 8];
                sum += a_byte * b_byte;
            end
        end
        assign c = sum;

        `BUFFER_EX(result, c, pe_enable, 1, LATENCY_DOT8);
        assign pe_data_out[i] = `XLEN'(result);

    `ifdef DBG_TRACE_PIPELINE
        always @(posedge clk) begin
            if (pe_enable) begin
                `TRACE(2, ("%t: %s dot8[%0d]: a=0%0h, b=0x%0h, c=0x%0h\n", $time, INSTANCE_ID, i, a, b, c))
            end
        end
    `endif
    end

endmodule
