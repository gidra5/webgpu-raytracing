declare module 'parse-hdr' {
  export default function parseHdr(data: ArrayBuffer | Uint8Array): {
    shape: [width: number, height: number];
    exposure: number;
    gamma: number;
    data: Float32Array;
  };
}

declare interface EXRData {
  colorSpace: 'srgb-linear';
}
