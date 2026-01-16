import * as ort from 'onnxruntime-web';
import type { LoadedModels } from './loadModels';

export type ModelManifest = {
  sampleRate: number;
  voiceStateShape: number[];
  streamStep: {
    inputs: {
      tokens: string;
      voiceState: string;
      kvCache: string[];
    };
    outputs: {
      nextLatents: string;
      kvCache: string[];
    };
  };
  decoder: {
    inputs: {
      latents: string;
    };
    outputs: {
      pcm: string;
    };
  };
};

export type StreamState = {
  kvCache: ort.Tensor[];
};

export async function initializeState(
  models: LoadedModels,
  manifest: ModelManifest,
  voiceState: ort.Tensor
): Promise<StreamState> {
  const zeroInputs: Record<string, ort.Tensor> = {
    [manifest.streamStep.inputs.tokens]: new ort.Tensor('int64', [0n], [1])
  };

  for (const cacheName of manifest.streamStep.inputs.kvCache) {
    zeroInputs[cacheName] = new ort.Tensor('float32', new Float32Array(0), [0]);
  }

  zeroInputs[manifest.streamStep.inputs.voiceState] = voiceState;

  const results = await models.streamStep.run(zeroInputs);
  const kvCache = manifest.streamStep.outputs.kvCache.map((name) => results[name]);

  return { kvCache };
}

export async function* streamTts(
  models: LoadedModels,
  manifest: ModelManifest,
  tokens: number[],
  voiceState: ort.Tensor,
  state: StreamState
) {
  for (const token of tokens) {
    const inputs: Record<string, ort.Tensor> = {
      [manifest.streamStep.inputs.tokens]: new ort.Tensor('int64', [BigInt(token)], [1]),
      [manifest.streamStep.inputs.voiceState]: voiceState
    };

    manifest.streamStep.inputs.kvCache.forEach((name, index) => {
      inputs[name] = state.kvCache[index];
    });

    const outputs = await models.streamStep.run(inputs);
    state.kvCache = manifest.streamStep.outputs.kvCache.map((name) => outputs[name]);
    const latents = outputs[manifest.streamStep.outputs.nextLatents] as ort.Tensor;

    const decoderOut = await models.decoder.run({
      [manifest.decoder.inputs.latents]: latents
    });

    const pcmTensor = decoderOut[manifest.decoder.outputs.pcm] as ort.Tensor;
    const pcmData = pcmTensor.data as Float32Array;
    yield pcmData;
  }
}
