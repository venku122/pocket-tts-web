import * as ort from 'onnxruntime-web';
import { loadModels, type LoadedModels, type ModelFiles } from '../model/loadModels';
import { TextTokenizer } from '../model/textTokenizer';
import type { ModelManifest, StreamState } from '../model/ttsPipeline';
import { initializeState, streamTts } from '../model/ttsPipeline';

export type WorkerRequest =
  | { type: 'load-models'; payload: ModelFiles }
  | { type: 'load-tokenizer'; payload: ArrayBuffer }
  | { type: 'load-manifest'; payload: ModelManifest }
  | { type: 'load-voice-state'; payload: { name: string; data: ArrayBuffer; shape: number[] } }
  | { type: 'synthesize'; payload: { text: string } }
  | { type: 'reset' };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'tokenizer-ready' }
  | { type: 'manifest-ready' }
  | { type: 'voice-ready'; payload: { name: string } }
  | { type: 'pcm-chunk'; payload: Float32Array }
  | { type: 'done' }
  | { type: 'error'; payload: string };

const ctx: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;

let models: LoadedModels | null = null;
let tokenizer: TextTokenizer | null = null;
let manifest: ModelManifest | null = null;
let voiceState: ort.Tensor | null = null;
let streamState: StreamState | null = null;

ctx.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  try {
    const message = event.data;
    switch (message.type) {
      case 'load-models': {
        models = await loadModels(message.payload);
        ctx.postMessage({ type: 'ready' } satisfies WorkerResponse);
        return;
      }
      case 'load-tokenizer': {
        tokenizer = new TextTokenizer();
        await tokenizer.load(new Uint8Array(message.payload));
        ctx.postMessage({ type: 'tokenizer-ready' } satisfies WorkerResponse);
        return;
      }
      case 'load-manifest': {
        manifest = message.payload;
        ctx.postMessage({ type: 'manifest-ready' } satisfies WorkerResponse);
        return;
      }
      case 'load-voice-state': {
        const tensor = new ort.Tensor(
          'float32',
          new Float32Array(message.payload.data),
          message.payload.shape
        );
        voiceState = tensor;
        ctx.postMessage({
          type: 'voice-ready',
          payload: { name: message.payload.name }
        } satisfies WorkerResponse);
        return;
      }
      case 'synthesize': {
        if (!models || !tokenizer || !manifest || !voiceState) {
          throw new Error('Models, tokenizer, manifest, and voice state must be loaded first');
        }
        const tokens = tokenizer.encode(message.payload.text);
        if (!streamState) {
          streamState = await initializeState(models, manifest, voiceState);
        }
        for await (const chunk of streamTts(models, manifest, tokens, voiceState, streamState)) {
          ctx.postMessage({ type: 'pcm-chunk', payload: chunk }, [chunk.buffer]);
        }
        ctx.postMessage({ type: 'done' } satisfies WorkerResponse);
        return;
      }
      case 'reset': {
        streamState = null;
        return;
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    ctx.postMessage({ type: 'error', payload: message } satisfies WorkerResponse);
  }
};
