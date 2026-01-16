export class AudioPlayer {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;

  async init(sampleRate: number) {
    if (this.context) {
      return;
    }
    this.context = new AudioContext({ sampleRate });
    await this.context.audioWorklet.addModule('/audio-worklet-processor.js');
    this.node = new AudioWorkletNode(this.context, 'ring-buffer-processor');
    this.node.connect(this.context.destination);
  }

  async resume() {
    if (this.context?.state === 'suspended') {
      await this.context.resume();
    }
  }

  pushAudio(data: Float32Array) {
    if (!this.node) {
      return;
    }
    this.node.port.postMessage({ type: 'push', payload: data }, [data.buffer]);
  }

  reset() {
    this.node?.port.postMessage({ type: 'reset' });
  }

  async close() {
    if (this.node) {
      this.node.disconnect();
      this.node = null;
    }
    if (this.context) {
      await this.context.close();
      this.context = null;
    }
  }
}
