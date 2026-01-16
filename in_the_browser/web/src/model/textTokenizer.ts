import { SentencePieceProcessor } from 'sentencepiece';
import sentencePieceWasmUrl from 'sentencepiece/dist/sentencepiece.wasm?url';

export class TextTokenizer {
  private processor: SentencePieceProcessor | null = null;

  async load(modelBytes: Uint8Array) {
    const init = (SentencePieceProcessor as unknown as { init?: (path: string) => Promise<void> })
      .init;
    if (init) {
      await init(sentencePieceWasmUrl);
    }
    this.processor = new SentencePieceProcessor();
    await this.processor.load(modelBytes);
  }

  encode(text: string): number[] {
    if (!this.processor) {
      throw new Error('Tokenizer not loaded');
    }
    return this.processor.encode(text);
  }
}
