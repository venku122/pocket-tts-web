import { useEffect, useMemo, useState } from 'react';
import { AudioPlayer } from './audio/AudioPlayer';
import type { WorkerRequest, WorkerResponse } from './worker/tts.worker';
import type { ModelManifest } from './model/ttsPipeline';

const DEFAULT_SAMPLE_RATE = 24000;

export function App() {
  const worker = useMemo(
    () => new Worker(new URL('./worker/tts.worker.ts', import.meta.url), { type: 'module' }),
    []
  );
  const audioPlayer = useMemo(() => new AudioPlayer(), []);

  const [status, setStatus] = useState('Idle');
  const [text, setText] = useState('Hello from Pocket TTS in the browser.');
  const [manifest, setManifest] = useState<ModelManifest | null>(null);
  const [voiceStateShape, setVoiceStateShape] = useState<number[] | null>(null);
  const [voiceName, setVoiceName] = useState<string | null>(null);
  const [sampleRate, setSampleRate] = useState(DEFAULT_SAMPLE_RATE);

  useEffect(() => {
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      switch (message.type) {
        case 'ready':
          setStatus('Models loaded');
          return;
        case 'tokenizer-ready':
          setStatus('Tokenizer ready');
          return;
        case 'manifest-ready':
          setStatus('Manifest loaded');
          return;
        case 'voice-ready':
          setVoiceName(message.payload.name);
          setStatus(`Voice loaded: ${message.payload.name}`);
          return;
        case 'pcm-chunk':
          audioPlayer.pushAudio(message.payload);
          return;
        case 'done':
          setStatus('Done');
          return;
        case 'error':
          setStatus(`Error: ${message.payload}`);
          return;
      }
    };

    return () => {
      worker.terminate();
      void audioPlayer.close();
    };
  }, [audioPlayer, worker]);

  const post = (payload: WorkerRequest, transfer?: Transferable[]) => {
    worker.postMessage(payload, transfer ?? []);
  };

  const handleModelFiles = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) {
      return;
    }
    const fileMap = new Map(Array.from(files).map((file) => [file.name, file]));
    const streamStep = fileMap.get('stream_step.onnx');
    const decoder = fileMap.get('decoder.onnx');
    const promptEncoder = fileMap.get('prompt_encoder.onnx');
    if (!streamStep || !decoder) {
      setStatus('Missing stream_step.onnx or decoder.onnx');
      return;
    }
    const payload = {
      streamStep: await streamStep.arrayBuffer(),
      decoder: await decoder.arrayBuffer(),
      promptEncoder: promptEncoder ? await promptEncoder.arrayBuffer() : undefined
    };
    post({ type: 'load-models', payload });
  };

  const handleTokenizerFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const buffer = await file.arrayBuffer();
    post({ type: 'load-tokenizer', payload: buffer }, [buffer]);
  };

  const handleManifestFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const text = await file.text();
    const parsed = JSON.parse(text) as ModelManifest;
    setManifest(parsed);
    setVoiceStateShape(parsed.voiceStateShape ?? null);
    setSampleRate(parsed.sampleRate ?? DEFAULT_SAMPLE_RATE);
    post({ type: 'load-manifest', payload: parsed });
  };

  const handleVoiceStateFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const buffer = await file.arrayBuffer();
    const shape =
      voiceStateShape ??
      prompt('Enter voice state shape (comma separated), e.g. 1,256,80')
        ?.split(',')
        .map((value) => Number(value.trim()));
    if (!shape || shape.length === 0) {
      setStatus('Voice state shape required');
      return;
    }
    post(
      {
        type: 'load-voice-state',
        payload: { name: file.name, data: buffer, shape }
      },
      [buffer]
    );
  };

  const handleSpeak = async () => {
    if (!manifest) {
      setStatus('Load manifest before speaking');
      return;
    }
    await audioPlayer.init(sampleRate);
    await audioPlayer.resume();
    audioPlayer.reset();
    setStatus('Synthesizing...');
    post({ type: 'synthesize', payload: { text } });
  };

  const handleReset = () => {
    post({ type: 'reset' });
    audioPlayer.reset();
    setStatus('Reset');
  };

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: 900 }}>
      <h1>Pocket TTS (Browser)</h1>
      <p>
        This is a client-only demo using onnxruntime-web and SentencePiece WASM. Load your
        exported models and tokenizer, then synthesize audio locally.
      </p>

      <section style={{ marginBottom: '1.5rem' }}>
        <h2>Load model assets</h2>
        <div style={{ display: 'grid', gap: '0.5rem' }}>
          <label>
            ONNX models (stream_step.onnx, decoder.onnx, optional prompt_encoder.onnx)
            <input type="file" multiple onChange={handleModelFiles} />
          </label>
          <label>
            Tokenizer model (tokenizer.model)
            <input type="file" onChange={handleTokenizerFile} />
          </label>
          <label>
            Manifest JSON (model_manifest.json)
            <input type="file" onChange={handleManifestFile} />
          </label>
          <label>
            Voice state (Float32 binary)
            <input type="file" onChange={handleVoiceStateFile} />
          </label>
        </div>
      </section>

      <section style={{ marginBottom: '1.5rem' }}>
        <h2>Text</h2>
        <textarea
          rows={4}
          style={{ width: '100%', padding: '0.75rem' }}
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.75rem' }}>
          <button type="button" onClick={handleSpeak}>
            Speak
          </button>
          <button type="button" onClick={handleReset}>
            Reset
          </button>
        </div>
      </section>

      <section style={{ marginBottom: '1.5rem' }}>
        <h2>Status</h2>
        <p>{status}</p>
        <p>Voice: {voiceName ?? 'none loaded'}</p>
        <p>Sample rate: {sampleRate} Hz</p>
        <p>Voice state shape: {voiceStateShape ? voiceStateShape.join(' × ') : 'unknown'}</p>
      </section>
    </div>
  );
}
