export const PROJECT_GEOMETRY_WORKGROUP_SIZE = 64;

export const projectGeometryWGSL = (bindings: string): string => /* wgsl */ `
  ${bindings}

  @compute @workgroup_size(${PROJECT_GEOMETRY_WORKGROUP_SIZE})
  fn main(@builtin(global_invocation_id) globalInvocationId: vec3u) {
    let vertexIndex = globalInvocationId.x;
    if (vertexIndex >= arrayLength(&vertices)) {
      return;
    }

    let vertex = vertices[vertexIndex];
    outVertices[vertexIndex] = viewProjMatrix * vec4f(vertex, 1.0);
  }
`;
