import { Iterator } from 'iterator-js';
import { EXRData } from 'parse-exr';

function sRGBtoLin(colorChannel: number) {
  // Send this function a decimal sRGB gamma encoded color value
  // between 0.0 and 1.0, and it returns a linearized value.

  if (colorChannel <= 0.04045) {
    return colorChannel / 12.92;
  } else {
    return Math.pow((colorChannel + 0.055) / 1.055, 2.4);
  }
}

// generate importance sampled skybox
// when sampled with a uniform 2d vector,
// the result is importance sampled 2d vector with pdf
// that can be used to generate a ray and sample the skybox
export const preprocess = async (skybox: EXRData) => {
  let luminance = Array.from(
    { length: skybox.width },
    () => new Array<number>(skybox.height)
  );

  let luminanceTotal = 0;
  for (const i of Iterator.natural(skybox.width)) {
    for (const j of Iterator.natural(skybox.height)) {
      let idx = i + j * skybox.width;
      const color = skybox.data.subarray(idx * 4, (idx + 1) * 4);
      const gammaCorrected = color.map(sRGBtoLin);
      const luminanceValue =
        0.2126 * gammaCorrected[0] +
        0.7152 * gammaCorrected[1] +
        0.0722 * gammaCorrected[2];
      luminance[i][j] = luminanceValue;
      luminanceTotal += luminanceValue;
    }
  }

  for (const i of Iterator.natural(skybox.width)) {
    for (const j of Iterator.natural(skybox.height)) {
      luminance[i][j] /= luminanceTotal;
    }
  }

  let marginalCDFTotal = 0;
  const marginalCDF = new Array<number>(skybox.height);
  const conditionalCDF = Array.from(
    { length: skybox.width },
    () => new Array<number>(skybox.height)
  );

  for (const i of Iterator.natural(skybox.width)) {
    let colTotal = 0;
    const colCDF = new Array(skybox.height);
    for (const j of Iterator.natural(skybox.height)) {
      colTotal += luminance[i][j];
      colCDF[j] = colTotal;
    }

    for (const j of Iterator.natural(skybox.height)) {
      conditionalCDF[i][j] = colCDF[j] / colTotal;
    }

    marginalCDFTotal += colTotal;
    marginalCDF[i] = marginalCDFTotal;
  }

  // normalize marginal CDF
  for (const i of Iterator.natural(skybox.width)) {
    marginalCDF[i] /= marginalCDFTotal;
  }

  let buffer = Array.from(
    { length: skybox.width },
    () => new Array<[number, number]>(skybox.height)
  );

  for (const i of Iterator.natural(skybox.width)) {
    let low = 0;
    let high = skybox.width - 1;
    let vIdx = 0;

    while (low <= high) {
      let mid = Math.floor(low + (high - low) / 2);
      if (marginalCDF[mid] < i / skybox.width) {
        low = mid + 1;
      } else {
        vIdx = mid;
        high = mid - 1;
      }
    }
    const v = vIdx / skybox.width;

    for (const j of Iterator.natural(skybox.height)) {
      let low = 0;
      let high = skybox.height - 1;
      let u = 0;

      while (low <= high) {
        let mid = Math.floor(low + (high - low) / 2);
        if (conditionalCDF[vIdx][mid] < j / skybox.height) {
          low = mid + 1;
        } else {
          u = mid;
          high = mid - 1;
        }
      }
      u /= skybox.height;

      buffer[i][j] = [v, u];
    }
  }

  const x = function* () {
    for (const j of Iterator.natural(skybox.height)) {
      for (const i of Iterator.natural(skybox.width)) {
        yield buffer[i][j][0];
        yield buffer[i][j][1];
        yield luminance[i][j];
        yield 0;
      }
    }
  };

  return new Float32Array(x());
};
