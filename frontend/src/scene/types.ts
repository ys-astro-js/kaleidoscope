import * as THREE from "three";
import type { Track } from "../types";

export type NodeRecord = {
  track: Track;
  mesh: THREE.Mesh;
};

export type ScoreLabelRecord = {
  mesh: THREE.Mesh;
  sprite: THREE.Sprite;
};

export type ShimmerLineMaterial = THREE.ShaderMaterial & {
  uniforms: {
    uTime: { value: number };
    uBaseColor: { value: THREE.Color };
    uShimmerColor: { value: THREE.Color };
    uOpacity: { value: number };
    uDashed: { value: number };
  };
};

export type SimilarityLineStyle = "primary" | "secondary" | "next";
