import * as THREE from "three";
import type { Track, ViewMode } from "../types";
import {
  FINAL_SEPARATION_ITERATIONS,
  MIN_2D_NODE_DISTANCE,
  SEPARATION_ITERATIONS,
} from "./constants";

export function createDisplayPositions(tracks: Track[], viewMode: ViewMode): Map<string, THREE.Vector3> {
  if (viewMode === "3d") {
    return new Map(tracks.map((track) => [track.id, rawTrackPosition(track, viewMode)]));
  }

  return createSeparated2DPositions(tracks);
}

export function trackPosition(track: Track, positions: Map<string, THREE.Vector3>): THREE.Vector3 {
  return positions.get(track.id) ?? rawTrackPosition(track, "3d");
}

function createSeparated2DPositions(tracks: Track[]): Map<string, THREE.Vector3> {
  if (tracks.length === 0) {
    return new Map();
  }

  const centroid = tracks.reduce(
    (sum, track) => {
      sum.x += track.x ?? 0;
      sum.y += track.y ?? 0;
      return sum;
    },
    { x: 0, y: 0 }
  );
  centroid.x /= tracks.length;
  centroid.y /= tracks.length;

  const scale = Math.max(1.25, Math.min(3.0, Math.sqrt(tracks.length) / 2.4));
  const points = tracks.map((track) => {
    const angle = hashToAngle(track.id);
    const baseX = ((track.x ?? 0) - centroid.x) * scale;
    const baseY = ((track.y ?? 0) - centroid.y) * scale;
    return {
      id: track.id,
      baseX,
      baseY,
      x: baseX + Math.cos(angle) * 0.04,
      y: baseY + Math.sin(angle) * 0.04
    };
  });

  for (let iteration = 0; iteration < SEPARATION_ITERATIONS; iteration += 1) {
    if (!separatePoints(points, MIN_2D_NODE_DISTANCE, true)) {
      break;
    }
  }
  for (let iteration = 0; iteration < FINAL_SEPARATION_ITERATIONS; iteration += 1) {
    if (!separatePoints(points, MIN_2D_NODE_DISTANCE, false)) {
      break;
    }
  }

  return new Map(points.map((point) => [point.id, new THREE.Vector3(point.x, point.y, 0)]));
}

function separatePoints(
  points: Array<{ id: string; baseX: number; baseY: number; x: number; y: number }>,
  minDistance: number,
  pullToBase: boolean
): boolean {
  let moved = false;

  for (let leftIndex = 0; leftIndex < points.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < points.length; rightIndex += 1) {
      const left = points[leftIndex];
      const right = points[rightIndex];
      let deltaX = right.x - left.x;
      let deltaY = right.y - left.y;
      let distance = Math.hypot(deltaX, deltaY);

      if (distance < 0.0001) {
        const angle = hashToAngle(`${left.id}:${right.id}`);
        deltaX = Math.cos(angle);
        deltaY = Math.sin(angle);
        distance = 1;
      }

      if (distance >= minDistance) {
        continue;
      }

      const push = ((minDistance - distance) / distance) * 0.52;
      const moveX = deltaX * push;
      const moveY = deltaY * push;
      left.x -= moveX;
      left.y -= moveY;
      right.x += moveX;
      right.y += moveY;
      moved = true;
    }
  }

  if (!pullToBase) {
    return moved;
  }

  for (const point of points) {
    point.x += (point.baseX - point.x) * 0.015;
    point.y += (point.baseY - point.y) * 0.015;
  }
  return moved;
}

function rawTrackPosition(track: Track, viewMode: ViewMode): THREE.Vector3 {
  return new THREE.Vector3(track.x ?? 0, track.y ?? 0, viewMode === "2d" ? 0 : track.z ?? 0);
}

function hashToAngle(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) / 4294967295) * Math.PI * 2;
}
