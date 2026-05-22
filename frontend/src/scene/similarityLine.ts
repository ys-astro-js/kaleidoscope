import * as THREE from "three";
import {
  NEXT_LINE_RADIUS,
  PRIMARY_LINE_RADIUS,
  SECONDARY_LINE_RADIUS,
} from "./constants";
import type { ShimmerLineMaterial, SimilarityLineStyle } from "./types";

export function createSimilarityLine(
  start: THREE.Vector3,
  end: THREE.Vector3,
  style: SimilarityLineStyle
): THREE.Group {
  const direction = end.clone().sub(start);
  const length = direction.length();
  const group = new THREE.Group();
  if (length < 0.001) {
    return group;
  }

  const radius =
    style === "next" ? NEXT_LINE_RADIUS : style === "secondary" ? SECONDARY_LINE_RADIUS : PRIMARY_LINE_RADIUS;
  const geometry = createLineGeometry(length, radius);
  const material = createShimmerLineMaterial(style);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(start).addScaledVector(direction, 0.5);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize()
  );
  mesh.renderOrder = style === "next" ? 2 : style === "secondary" ? 0 : 1;
  group.add(mesh);

  return group;
}

export function collectShimmerMaterials(group: THREE.Group): ShimmerLineMaterial[] {
  const materials: ShimmerLineMaterial[] = [];
  group.traverse((object) => {
    const mesh = object as THREE.Mesh;
    const material = mesh.material;
    if (material instanceof THREE.ShaderMaterial && "uTime" in material.uniforms) {
      materials.push(material as ShimmerLineMaterial);
    }
  });
  return materials;
}

function createLineGeometry(length: number, radius: number): THREE.CylinderGeometry {
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 14, 56, true);
  const positions = geometry.attributes.position;
  const progress = new Float32Array(positions.count);
  for (let index = 0; index < positions.count; index += 1) {
    progress[index] = (positions.getY(index) + length / 2) / length;
  }
  geometry.setAttribute("lineProgress", new THREE.BufferAttribute(progress, 1));
  return geometry;
}

function createShimmerLineMaterial(style: SimilarityLineStyle): ShimmerLineMaterial {
  const dashed = style === "secondary";
  const isNext = style === "next";
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uBaseColor: { value: new THREE.Color(isNext ? "#ffd84a" : dashed ? "#a8a8a8" : "#d7d7d7") },
      uShimmerColor: { value: new THREE.Color(isNext ? "#fff4a8" : "#ffffff") },
      uOpacity: { value: isNext ? 0.96 : dashed ? 0.42 : 0.78 },
      uDashed: { value: dashed ? 1 : 0 }
    },
    vertexShader: `
      attribute float lineProgress;
      varying float vLineProgress;

      void main() {
        vLineProgress = lineProgress;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float uTime;
      uniform vec3 uBaseColor;
      uniform vec3 uShimmerColor;
      uniform float uOpacity;
      uniform float uDashed;
      varying float vLineProgress;

      void main() {
        float dash = 1.0;
        if (uDashed > 0.5) {
          dash = step(0.42, fract(vLineProgress * 12.0));
        }

        float head = fract(uTime * 0.58);
        float distanceFromHead = abs(vLineProgress - head);
        distanceFromHead = min(distanceFromHead, 1.0 - distanceFromHead);
        float shimmer = smoothstep(0.18, 0.0, distanceFromHead);
        float leadingEdge = smoothstep(head - 0.025, head, vLineProgress)
          * (1.0 - smoothstep(head, head + 0.09, vLineProgress));
        shimmer = max(shimmer * 0.72, leadingEdge);

        vec3 color = mix(uBaseColor, uShimmerColor, shimmer);
        float alpha = (uOpacity + shimmer * 0.22) * dash;
        gl_FragColor = vec4(color, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  }) as ShimmerLineMaterial;
}
