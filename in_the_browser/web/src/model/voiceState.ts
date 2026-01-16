export type VoiceState = {
  name: string;
  data: Float32Array;
};

export async function loadVoiceState(file: File): Promise<VoiceState> {
  const arrayBuffer = await file.arrayBuffer();
  return {
    name: file.name,
    data: new Float32Array(arrayBuffer)
  };
}
