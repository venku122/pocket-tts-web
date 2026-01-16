class RingBufferProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    this.readOffset = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type === 'push') {
        const incoming = event.data.payload;
        const remaining = this.buffer.length - this.readOffset;
        const next = new Float32Array(remaining + incoming.length);
        if (remaining > 0) {
          next.set(this.buffer.subarray(this.readOffset), 0);
        }
        next.set(incoming, remaining);
        this.buffer = next;
        this.readOffset = 0;
      } else if (event.data?.type === 'reset') {
        this.buffer = new Float32Array(0);
        this.readOffset = 0;
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output) {
      return true;
    }
    const channel = output[0];
    if (!channel) {
      return true;
    }
    const available = this.buffer.length - this.readOffset;
    if (available <= 0) {
      channel.fill(0);
      return true;
    }
    const toCopy = Math.min(channel.length, available);
    channel.set(this.buffer.subarray(this.readOffset, this.readOffset + toCopy));
    if (toCopy < channel.length) {
      channel.fill(0, toCopy);
    }
    this.readOffset += toCopy;
    return true;
  }
}

registerProcessor('ring-buffer-processor', RingBufferProcessor);
