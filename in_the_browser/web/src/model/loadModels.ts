import * as ort from 'onnxruntime-web';

export type ModelFiles = {
  promptEncoder?: ArrayBuffer;
  streamStep: ArrayBuffer;
  decoder: ArrayBuffer;
};

export type LoadedModels = {
  streamStep: ort.InferenceSession;
  decoder: ort.InferenceSession;
  promptEncoder?: ort.InferenceSession;
};

export async function loadModels(files: ModelFiles): Promise<LoadedModels> {
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = navigator.hardwareConcurrency
    ? Math.min(2, navigator.hardwareConcurrency)
    : 1;

  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  };

  const streamStep = await ort.InferenceSession.create(files.streamStep, sessionOptions);
  const decoder = await ort.InferenceSession.create(files.decoder, sessionOptions);

  let promptEncoder: ort.InferenceSession | undefined;
  if (files.promptEncoder) {
    promptEncoder = await ort.InferenceSession.create(files.promptEncoder, sessionOptions);
  }

  return { streamStep, decoder, promptEncoder };
}
