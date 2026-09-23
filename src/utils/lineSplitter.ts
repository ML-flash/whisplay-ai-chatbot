// Splits a TCP byte stream into newline-terminated messages.
// TCP does not preserve message boundaries: one "data" event can carry
// several messages ("OK\nOK\n") or only part of one, so text is buffered
// until a newline completes each message.
export class LineSplitter {
  private buffer = "";

  push(chunk: string): string[] {
    this.buffer += chunk;
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop() ?? "";
    return lines.map((line) => line.trim()).filter((line) => line !== "");
  }

  reset(): void {
    this.buffer = "";
  }
}
