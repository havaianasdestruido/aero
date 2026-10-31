// A tiny, self-contained bootable demo that ships with the web player.
//
// The player must be able to show a *working* machine the moment someone opens
// the page: no ISO to download, no media to find, no build step at runtime. To
// do that it boots a 512-byte boot sector, generated here, that uses only the
// firmware services the emulator implements (INT 10h video, INT 16h keyboard)
// plus direct VGA text-memory writes and COM1 output.
//
// The "operating system" is deliberately small: it paints a banner, announces
// itself on the serial port, and then echoes everything typed at the keyboard,
// so a visitor can verify that the emulated CPU, firmware, VGA text mode and
// PS/2 keyboard are all alive.
//
// The generator below is a hand-rolled 16-bit real-mode assembler for the
// handful of instructions the demo needs. Every opcode is spelled out, which is
// what makes a 512-byte guest auditable: the boot sector is reproducible and
// diffable instead of being an opaque blob.

/**
 * Absolute load address of the boot sector. Labels are tracked as offsets inside
 * the sector, so any *absolute* reference (string pointers) must add this.
 */
const LOAD_ADDRESS = 0x7c00;

/** Byte offsets inside the sector used as the guest program's variables. */
const VAR_CURSOR = 0x7000; // word: byte offset into the VGA text buffer

const VIDEO_SEGMENT = 0xb800;
const BYTES_PER_ROW = 80 * 2;
const SCREEN_BYTES = BYTES_PER_ROW * 25;

const ATTR_BRIGHT_WHITE_ON_BLUE = 0x1f;
const ATTR_GREY_ON_BLACK = 0x07;
const ATTR_ECHO = 0x0f;

const PROMPT_ROW = 9;
const PROMPT_COL = 2;
const PROMPT_TEXT = "A:\\>";
const CURSOR_START = PROMPT_ROW * BYTES_PER_ROW + (PROMPT_COL + PROMPT_TEXT.length) * 2;

/** Serial announcement; lets headless smoke tests detect a successful boot. */
const SERIAL_BANNER = "AERO-DEMO-BOOT-OK\r\n";

type ScreenLine = { readonly row: number; readonly col: number; readonly attr: number; readonly text: string };

const BANNER: readonly ScreenLine[] = [
  { row: 1, col: 2, attr: ATTR_BRIGHT_WHITE_ON_BLUE, text: "****  AERO EMULATOR  ****" },
  { row: 2, col: 2, attr: 0x1e, text: "A real x86 PC in your browser." },
  { row: 3, col: 2, attr: 0x1e, text: "BIOS, VGA, keyboard: emulated." },
  { row: 5, col: 2, attr: ATTR_GREY_ON_BLACK, text: "This boot sector is the OS." },
  { row: 7, col: 2, attr: 0x1a, text: "Type below. Esc clears." },
];

type Patch = { readonly kind: "rel8" | "rel16" | "addr16"; readonly at: number; readonly label: string };

/** Two-pass emitter: label references are resolved in {@link finish}. */
class BootSectorBuilder {
  readonly bytes: number[] = [];
  private readonly labels = new Map<string, number>();
  private readonly patches: Patch[] = [];

  get pos(): number {
    return this.bytes.length;
  }

  db(...values: number[]): this {
    for (const value of values) this.bytes.push(value & 0xff);
    return this;
  }

  label(name: string): this {
    if (this.labels.has(name)) throw new Error(`duplicate label '${name}'`);
    this.labels.set(name, this.pos);
    return this;
  }

  /** `mov reg16, imm16` for AX/CX/DX/BX/SI/DI (B8+reg). */
  movReg16(reg: "ax" | "cx" | "dx" | "bx" | "sp" | "si" | "di", value: number): this {
    const opcode = { ax: 0xb8, cx: 0xb9, dx: 0xba, bx: 0xbb, sp: 0xbc, si: 0xbe, di: 0xbf }[reg];
    return this.db(opcode, value & 0xff, (value >>> 8) & 0xff);
  }

  /** `mov si, <label>` where the label is an address inside the loaded sector. */
  movSiLabel(label: string): this {
    this.bytes.push(0xbe, 0x00, 0x00);
    this.patches.push({ kind: "addr16", at: this.pos - 2, label });
    return this;
  }

  movAlImm8(value: number): this {
    return this.db(0xb0, value);
  }

  movAhImm8(value: number): this {
    return this.db(0xb4, value);
  }

  movBhImm8(value: number): this {
    return this.db(0xb7, value);
  }

  movBlImm8(value: number): this {
    return this.db(0xb3, value);
  }

  /** `mov [imm16], bx` (89 1E iw). */
  movMemBx(addr: number): this {
    return this.db(0x89, 0x1e, addr & 0xff, (addr >>> 8) & 0xff);
  }

  /** `mov bx, [imm16]` (8B 1E iw). */
  movBxMem(addr: number): this {
    return this.db(0x8b, 0x1e, addr & 0xff, (addr >>> 8) & 0xff);
  }

  movDiBx(): this {
    return this.db(0x89, 0xdf); // 89 /r, mod=11, reg=BX, rm=DI
  }

  movAxBx(): this {
    return this.db(0x89, 0xd8); // 89 /r, mod=11, reg=BX, rm=AX
  }

  movBxAx(): this {
    return this.db(0x89, 0xc3); // 89 /r, mod=11, reg=AX, rm=BX
  }

  movAlBl(): this {
    return this.db(0x88, 0xd8); // 88 /r, mod=11, reg=BL, rm=AL
  }

  /**
   * `mov es:[di], al` (26 88 05).
   *
   * 16-bit ModRM uses r/m=101 for [DI] (r/m=111 is [BX]); getting this wrong
   * silently scribbles over the wrong address, so it gets its own tested helper.
   */
  movEsDiAl(): this {
    return this.db(0x26, 0x88, 0x05);
  }

  incDi(): this {
    return this.db(0x47);
  }

  incAx(): this {
    return this.db(0x40);
  }

  /** `add bx, imm16`. */
  addBxImm(value: number): this {
    if (value > 0 && value <= 0x7f) return this.db(0x83, 0xc3, value);
    return this.db(0x81, 0xc3, value & 0xff, (value >>> 8) & 0xff);
  }

  /** `sub bx, imm16`. */
  subBxImm(value: number): this {
    if (value > 0 && value <= 0x7f) return this.db(0x83, 0xeb, value);
    return this.db(0x81, 0xeb, value & 0xff, (value >>> 8) & 0xff);
  }

  /** `cmp bx, imm16` (81 FB iw). */
  cmpBxImm(value: number): this {
    return this.db(0x81, 0xfb, value & 0xff, (value >>> 8) & 0xff);
  }

  cmpAlImm8(value: number): this {
    return this.db(0x3c, value);
  }

  /** `int imm8`. */
  int(number: number): this {
    return this.db(0xcd, number);
  }

  /** `jmp short <label>` (EB rel8). */
  jmpShort(label: string): this {
    this.bytes.push(0xeb, 0x00);
    this.patches.push({ kind: "rel8", at: this.pos - 1, label });
    return this;
  }

  /** `jmp near <label>` (E9 rel16). */
  jmpNear(label: string): this {
    this.bytes.push(0xe9, 0x00, 0x00);
    this.patches.push({ kind: "rel16", at: this.pos - 2, label });
    return this;
  }

  /**
   * Conditional near jump (0F 8x rel16) for targets beyond +-127 bytes.
   *
   * The near form reuses the *condition* code, not the short-jump opcode:
   * short 0x72 (jb) is `0F 82`, 0x74 (je) is `0F 84`, and so on.
   */
  jccNear(shortOpcode: 0x74 | 0x75 | 0x72 | 0x77, label: string): this {
    const nearOpcode = (shortOpcode & 0x0f) | 0x80;
    this.bytes.push(0x0f, nearOpcode, 0x00, 0x00);
    this.patches.push({ kind: "rel16", at: this.pos - 2, label });
    return this;
  }

  /** Conditional short jump: 0x74 jz/je, 0x75 jnz/jne, 0x72 jb, 0x77 ja. */
  jccShort(opcode: 0x74 | 0x75 | 0x72 | 0x77, label: string): this {
    this.bytes.push(opcode, 0x00);
    this.patches.push({ kind: "rel8", at: this.pos - 1, label });
    return this;
  }

  /** `call rel16 <label>` (E8 rel16). */
  call(label: string): this {
    this.bytes.push(0xe8, 0x00, 0x00);
    this.patches.push({ kind: "rel16", at: this.pos - 2, label });
    return this;
  }

  ret(): this {
    return this.db(0xc3);
  }

  hlt(): this {
    return this.db(0xf4);
  }

  cli(): this {
    return this.db(0xfa);
  }

  sti(): this {
    return this.db(0xfb);
  }

  cld(): this {
    return this.db(0xfc);
  }

  lodsb(): this {
    return this.db(0xac);
  }

  /** `xor ax, ax` */
  xorAxAx(): this {
    return this.db(0x31, 0xc0);
  }

  /** `xor dx, dx` */
  xorDxDx(): this {
    return this.db(0x31, 0xd2);
  }

  /** `xor ah, ah` */
  xorAhAh(): this {
    return this.db(0x30, 0xe4);
  }

  /** `test al, al` */
  testAlAl(): this {
    return this.db(0x84, 0xc0);
  }

  movDsAx(): this {
    return this.db(0x8e, 0xd8);
  }

  movEsAx(): this {
    return this.db(0x8e, 0xc0);
  }

  movSsAx(): this {
    return this.db(0x8e, 0xd0);
  }

  /** `out dx, al` */
  outDxAl(): this {
    return this.db(0xee);
  }

  /** `div cx` (DX:AX / CX). */
  divCx(): this {
    return this.db(0xf7, 0xf1);
  }

  /** `mul cx` (AX * CX -> DX:AX). */
  mulCx(): this {
    return this.db(0xf7, 0xe1);
  }

  /** Emit a NUL-terminated string under `label`. */
  string(label: string, text: string): this {
    this.label(label);
    for (let i = 0; i < text.length; i += 1) this.bytes.push(text.charCodeAt(i) & 0xff);
    this.bytes.push(0);
    return this;
  }

  /**
   * Resolve labels, apply the boot signature, and hand back exactly one sector.
   *
   * Throws rather than emitting something that silently cannot boot.
   */
  finish(): Uint8Array {
    for (const patch of this.patches) {
      const target = this.labels.get(patch.label);
      if (target === undefined) throw new Error(`demo boot sector: undefined label '${patch.label}'`);
      if (patch.kind === "rel8") {
        const rel = target - (patch.at + 1);
        if (rel < -128 || rel > 127) {
          throw new Error(`demo boot sector: short jump to '${patch.label}' out of range (${rel})`);
        }
        this.bytes[patch.at] = rel & 0xff;
      } else if (patch.kind === "rel16") {
        const rel = target - (patch.at + 2);
        this.bytes[patch.at] = rel & 0xff;
        this.bytes[patch.at + 1] = (rel >>> 8) & 0xff;
      } else {
        const addr = LOAD_ADDRESS + target;
        this.bytes[patch.at] = addr & 0xff;
        this.bytes[patch.at + 1] = (addr >>> 8) & 0xff;
      }
    }

    if (this.bytes.length > 510) {
      throw new Error(`demo boot sector: program too large (${this.bytes.length} bytes, limit 510)`);
    }

    const sector = new Uint8Array(512);
    sector.set(this.bytes, 0);
    sector[510] = 0x55;
    sector[511] = 0xaa;
    return sector;
  }
}

/**
 * Build the 512-byte boot sector the player boots as its zero-setup demo.
 */
export function buildDemoBootSector(): Uint8Array {
  const asm = new BootSectorBuilder();

  // --- entry -------------------------------------------------------------
  asm.cli().cld().xorAxAx().movDsAx().movEsAx().movSsAx().movReg16("sp", LOAD_ADDRESS).sti();

  // Text mode 3 (80x25, 16 colours). INT 10h AH=00h also clears the screen.
  asm.movReg16("ax", 0x0003).int(0x10);

  // Announce on COM1 first, so a headless test fails loudly if anything below
  // goes wrong.
  asm.movSiLabel("serial_banner").call("puts_serial");

  // Progress markers on COM1. Two bytes of output that make a headless run
  // self-diagnosing: `12` means video and the drawing loop both ran and the
  // guest is about to poll INT 16h.
  asm.movAlImm8(0x31).call("putc_serial"); // '1'

  // --- banner ------------------------------------------------------------
  asm.movReg16("ax", VIDEO_SEGMENT).movEsAx();
  for (const line of BANNER) {
    asm
      .movReg16("di", line.row * BYTES_PER_ROW + line.col * 2)
      .movBlImm8(line.attr)
      .movSiLabel(`line_${line.row}`)
      .call("puts_video");
  }

  // Prompt, then park the echo cursor just past it.
  asm
    .movReg16("di", PROMPT_ROW * BYTES_PER_ROW + PROMPT_COL * 2)
    .movBlImm8(ATTR_ECHO)
    .movSiLabel("prompt")
    .call("puts_video")
    .movReg16("bx", CURSOR_START)
    .movMemBx(VAR_CURSOR);

  asm.movAlImm8(0x32).call("putc_serial"); // '2'

  // --- keyboard loop -----------------------------------------------------
  asm.label("key_loop").xorAhAh().int(0x16); // blocking read: AL=ASCII, AH=scancode
  asm.testAlAl().jccShort(0x74, "key_loop"); // no ASCII (arrows/F-keys): ignore

  asm.cmpAlImm8(0x0d).jccNear(0x74, "do_enter");
  asm.cmpAlImm8(0x08).jccNear(0x74, "do_backspace");
  asm.cmpAlImm8(0x1b).jccNear(0x74, "do_clear");
  asm.call("echo_char").jmpShort("key_loop");

  // --- echo one character ------------------------------------------------
  // Mirroring the character to COM1 gives headless tests (and anyone watching
  // the serial panel) an unambiguous record of what the guest actually received,
  // independent of what the canvas happens to be showing.
  asm.label("echo_char");
  asm.call("putc_serial");
  asm.movBxMem(VAR_CURSOR).movDiBx().movEsDiAl().incDi();
  asm.movAlImm8(ATTR_ECHO).movEsDiAl();
  asm.addBxImm(2).movMemBx(VAR_CURSOR);
  asm.cmpBxImm(SCREEN_BYTES - 2).jccShort(0x72, "echo_done");
  asm.call("newline");
  asm.label("echo_done").ret();

  // --- newline -----------------------------------------------------------
  // Advance the cursor to the start of the next row, scrolling when the last
  // row is reached. row = cursor / 160, next = (row + 1) * 160.
  asm.label("newline");
  asm.movBxMem(VAR_CURSOR).movAxBx();
  asm.xorDxDx().movReg16("cx", BYTES_PER_ROW).divCx().incAx().mulCx().movBxAx();
  asm.cmpBxImm(SCREEN_BYTES).jccShort(0x72, "newline_store");
  // Scroll the screen up one line and keep the cursor on the last row.
  asm.movReg16("ax", 0x0601).movBhImm8(ATTR_BRIGHT_WHITE_ON_BLUE);
  asm.movReg16("cx", 0).movReg16("dx", 0x184f).int(0x10);
  asm.movReg16("bx", (25 - 1) * BYTES_PER_ROW);
  asm.label("newline_store").movMemBx(VAR_CURSOR).ret();

  // --- backspace ---------------------------------------------------------
  asm.label("do_backspace");
  asm.movBxMem(VAR_CURSOR).subBxImm(2).cmpBxImm(CURSOR_START).jccNear(0x72, "key_loop");
  asm.movMemBx(VAR_CURSOR).movDiBx();
  asm.movAlImm8(0x20).movEsDiAl().incDi();
  asm.movAlImm8(ATTR_ECHO).movEsDiAl();
  asm.jmpNear("key_loop");

  // --- enter -------------------------------------------------------------
  asm.label("do_enter").call("serial_crlf").call("newline").jmpNear("key_loop");

  // --- escape ------------------------------------------------------------
  asm.label("do_clear");
  asm.movReg16("ax", 0x0003).int(0x10); // re-set mode 3: clears the screen
  asm.movReg16("ax", VIDEO_SEGMENT).movEsAx();
  asm.movReg16("bx", 0).movMemBx(VAR_CURSOR).jmpNear("key_loop");

  // --- helpers -----------------------------------------------------------
  // puts_video: DS:SI -> NUL-terminated string, ES:DI destination, BL attribute.
  asm.label("puts_video");
  asm.lodsb().testAlAl().jccShort(0x74, "puts_video_done");
  asm.movEsDiAl().incDi().movAlBl().movEsDiAl().incDi();
  asm.jmpShort("puts_video");
  asm.label("puts_video_done").ret();

  // putc_serial: write AL to COM1. The emulated UART's FIFO never overruns at
  // the rates a human types, so no line-status poll is needed here.
  asm.label("putc_serial").movReg16("dx", 0x03f8).outDxAl().ret();

  // serial_crlf: CR LF on COM1, so echoed lines stay readable in the log.
  asm.label("serial_crlf");
  asm.movAlImm8(0x0d).call("putc_serial");
  asm.movAlImm8(0x0a).call("putc_serial").ret();

  // puts_serial: DS:SI -> NUL-terminated string, written to COM1 (0x3F8).
  asm.label("puts_serial").movReg16("dx", 0x03f8);
  asm.label("puts_serial_loop");
  asm.lodsb().testAlAl().jccShort(0x74, "puts_serial_done");
  asm.outDxAl().jmpShort("puts_serial_loop");
  asm.label("puts_serial_done").ret();

  // --- data --------------------------------------------------------------
  asm.string("serial_banner", SERIAL_BANNER);
  asm.string("prompt", PROMPT_TEXT);
  for (const line of BANNER) asm.string(`line_${line.row}`, line.text);

  return asm.finish();
}
