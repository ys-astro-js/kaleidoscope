import * as THREE from "three";
import { artUrl } from "../api";
import type { Track } from "../types";

export function getTexture(
  cache: Map<string, THREE.Texture>,
  loader: THREE.TextureLoader,
  trackId: string
): THREE.Texture {
  const cached = cache.get(trackId);
  if (cached) {
    return cached;
  }

  const texture = loader.load(artUrl(trackId));
  texture.colorSpace = THREE.SRGBColorSpace;
  cache.set(trackId, texture);
  return texture;
}

export function disposeUnusedTextures(cache: Map<string, THREE.Texture>, tracks: Track[]): void {
  const activeIds = new Set(tracks.map((track) => track.id));
  for (const [trackId, texture] of cache) {
    if (!activeIds.has(trackId)) {
      texture.dispose();
      cache.delete(trackId);
    }
  }
}
