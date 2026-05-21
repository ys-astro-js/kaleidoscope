import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { artUrl } from "./api";
import type { Track, ViewMode } from "./types";

type Props = {
  tracks: Track[];
  selectedId: string | null;
  viewMode: ViewMode;
  onSelect: (track: Track) => void;
};

type NodeRecord = {
  track: Track;
  mesh: THREE.Mesh;
};

type ScoreLabelRecord = {
  mesh: THREE.Mesh;
  sprite: THREE.Sprite;
};

type ShimmerLineMaterial = THREE.ShaderMaterial & {
  uniforms: {
    uTime: { value: number };
    uBaseColor: { value: THREE.Color };
    uShimmerColor: { value: THREE.Color };
    uOpacity: { value: number };
    uDashed: { value: number };
  };
};

const CAMERA_2D_DISTANCE = 26;
const CAMERA_3D_POSITION = new THREE.Vector3(0, 0, 12);
const NODE_SIZE = 1.1;
const MIN_2D_NODE_DISTANCE = 1.55;
const SEPARATION_ITERATIONS = 56;
const FINAL_SEPARATION_ITERATIONS = 16;
const PRIMARY_SIMILAR_LIMIT = 5;
const SECONDARY_SIMILAR_LIMIT = 3;
const PRIMARY_LINE_RADIUS = 0.016;
const SECONDARY_LINE_RADIUS = PRIMARY_LINE_RADIUS;

export default function MusicScene({ tracks, selectedId, viewMode, onSelect }: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const onSelectRef = useRef(onSelect);
  const tracksRef = useRef(tracks);
  const selectedRef = useRef(selectedId);
  const viewModeRef = useRef<ViewMode>(viewMode);
  const rebuildSceneRef = useRef<(() => void) | null>(null);
  const applyViewModeRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    tracksRef.current = tracks;
    rebuildSceneRef.current?.();
  }, [tracks]);

  useEffect(() => {
    selectedRef.current = selectedId;
    rebuildSceneRef.current?.();
  }, [selectedId]);

  useEffect(() => {
    if (viewModeRef.current === viewMode) {
      return;
    }

    viewModeRef.current = viewMode;
    applyViewModeRef.current?.();
    rebuildSceneRef.current?.();
  }, [viewMode]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#050505");

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    camera.position.set(0, 0, 12);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 4;
    controls.maxDistance = 26;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const textureLoader = new THREE.TextureLoader();
    const nodeGroup = new THREE.Group();
    const lineGroup = new THREE.Group();
    const labelGroup = new THREE.Group();
    const nodeRecords: NodeRecord[] = [];
    const scoreLabelRecords: ScoreLabelRecord[] = [];
    const shimmerMaterials: ShimmerLineMaterial[] = [];
    const textureCache = new Map<string, THREE.Texture>();
    let displayPositions = new Map<string, THREE.Vector3>();
    let nodeSignature = "";
    let linkSignature = "";
    let animationFrame = 0;
    let pointerStart: { x: number; y: number } | null = null;
    let suppressNextClick = false;
    scene.add(nodeGroup, lineGroup, labelGroup);

    const resize = () => {
      const { width, height } = mount.getBoundingClientRect();
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };

    const applyViewMode = () => {
      const is2d = viewModeRef.current === "2d";
      controls.enableRotate = !is2d;
      controls.enablePan = is2d;
      controls.screenSpacePanning = is2d;
      controls.mouseButtons.LEFT = is2d ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
      controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
      controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
      controls.touches.ONE = is2d ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE;
      controls.touches.TWO = THREE.TOUCH.DOLLY_PAN;
      controls.minDistance = is2d ? 10 : 4;
      controls.maxDistance = is2d ? 60 : 26;
      controls.target.set(0, 0, 0);
      if (is2d) {
        camera.position.set(0, 0, CAMERA_2D_DISTANCE);
      } else {
        camera.position.copy(CAMERA_3D_POSITION);
      }
      controls.update();
    };

    const rebuild = () => {
      const readyTracks = tracksRef.current.filter(
        (track) => track.status === "ready" && track.x !== null && track.y !== null && track.z !== null
      );
      const nextNodeSignature = createNodeSignature(readyTracks, viewModeRef.current);
      const nodesChanged = nextNodeSignature !== nodeSignature;

      if (nodesChanged) {
        displayPositions = createDisplayPositions(readyTracks, viewModeRef.current);
        disposeGroup(nodeGroup, { disposeTextures: false });
        nodeRecords.length = 0;
        disposeUnusedTextures(textureCache, readyTracks);

        for (const track of readyTracks) {
          const texture = getTexture(textureCache, textureLoader, track.id);
          const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
          const mesh = new THREE.Mesh(new THREE.PlaneGeometry(NODE_SIZE, NODE_SIZE), material);
          mesh.position.copy(trackPosition(track, displayPositions));
          mesh.renderOrder = 1;
          mesh.userData.trackId = track.id;
          nodeGroup.add(mesh);
          nodeRecords.push({ track, mesh });
        }
        nodeSignature = nextNodeSignature;
      } else {
        const nextTracksById = new Map(readyTracks.map((track) => [track.id, track]));
        for (const record of nodeRecords) {
          record.track = nextTracksById.get(record.track.id) ?? record.track;
        }
      }

      const nextLinkSignature = createLinkSignature(readyTracks, selectedRef.current);
      if (!nodesChanged && nextLinkSignature === linkSignature) {
        return;
      }

      disposeGroup(lineGroup);
      disposeGroup(labelGroup);
      scoreLabelRecords.length = 0;
      shimmerMaterials.length = 0;

      const selected = readyTracks.find((track) => track.id === selectedRef.current);
      if (selected) {
        const readyTracksById = new Map(readyTracks.map((track) => [track.id, track]));
        const related = selected.similar
          .slice(0, PRIMARY_SIMILAR_LIMIT)
          .map((similar) => ({
            score: similar.score,
            track: readyTracksById.get(similar.id)
          }))
          .filter((similar): similar is { score: number; track: Track } => Boolean(similar.track));

        for (const target of related) {
          const line = createSimilarityLine(
            trackPosition(selected, displayPositions),
            trackPosition(target.track, displayPositions),
            false
          );
          lineGroup.add(line);
          shimmerMaterials.push(...collectShimmerMaterials(line));

          const targetRecord = nodeRecords.find((node) => node.track.id === target.track.id);
          if (targetRecord) {
            const sprite = createScoreSprite(target.score);
            labelGroup.add(sprite);
            scoreLabelRecords.push({ mesh: targetRecord.mesh, sprite });
          }
        }

        for (const link of createSecondarySimilarityLinks(
          selected,
          related.map((item) => item.track),
          readyTracksById
        )) {
          const line = createSimilarityLine(
            trackPosition(link.source, displayPositions),
            trackPosition(link.target, displayPositions),
            true
          );
          lineGroup.add(line);
          shimmerMaterials.push(...collectShimmerMaterials(line));
        }
      }
      linkSignature = nextLinkSignature;
    };

    const handlePointerDown = (event: PointerEvent) => {
      pointerStart = { x: event.clientX, y: event.clientY };
      suppressNextClick = false;
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!pointerStart) {
        return;
      }
      const deltaX = event.clientX - pointerStart.x;
      const deltaY = event.clientY - pointerStart.y;
      if (deltaX * deltaX + deltaY * deltaY > 16) {
        suppressNextClick = true;
      }
    };

    const handleClick = (event: MouseEvent) => {
      pointerStart = null;
      if (suppressNextClick) {
        suppressNextClick = false;
        return;
      }

      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const intersects = raycaster.intersectObjects(nodeGroup.children);
      const trackId = intersects[0]?.object.userData.trackId;
      const record = nodeRecords.find((node) => node.track.id === trackId);
      if (record) {
        onSelectRef.current(record.track);
      }
    };

    const animate = () => {
      for (const record of nodeRecords) {
        record.mesh.position.copy(trackPosition(record.track, displayPositions));
        if (viewModeRef.current === "2d") {
          record.mesh.rotation.set(0, 0, 0);
        } else {
          record.mesh.lookAt(camera.position);
        }
        const isSelected = record.track.id === selectedRef.current;
        record.mesh.scale.setScalar(isSelected ? 1.18 : 1);
      }
      for (const record of scoreLabelRecords) {
        record.sprite.position.copy(record.mesh.position);
        record.sprite.position.y -= 0.78;
      }
      const elapsed = performance.now() * 0.001;
      for (const material of shimmerMaterials) {
        material.uniforms.uTime.value = elapsed;
      }
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };

    resize();
    rebuildSceneRef.current = rebuild;
    applyViewModeRef.current = applyViewMode;
    applyViewMode();
    rebuild();
    animate();

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("click", handleClick);

    return () => {
      rebuildSceneRef.current = null;
      applyViewModeRef.current = null;
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("click", handleClick);
      disposeGroup(nodeGroup, { disposeTextures: false });
      disposeGroup(lineGroup);
      disposeGroup(labelGroup);
      for (const texture of textureCache.values()) {
        texture.dispose();
      }
      renderer.dispose();
      if (renderer.domElement.parentElement === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, []);

  return <div className={`scene scene-${viewMode}`} ref={mountRef} />;
}

function getTexture(
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

function disposeUnusedTextures(cache: Map<string, THREE.Texture>, tracks: Track[]): void {
  const activeIds = new Set(tracks.map((track) => track.id));
  for (const [trackId, texture] of cache) {
    if (!activeIds.has(trackId)) {
      texture.dispose();
      cache.delete(trackId);
    }
  }
}

function createDisplayPositions(tracks: Track[], viewMode: ViewMode): Map<string, THREE.Vector3> {
  if (viewMode === "3d") {
    return new Map(tracks.map((track) => [track.id, rawTrackPosition(track, viewMode)]));
  }

  return createSeparated2DPositions(tracks);
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

function trackPosition(track: Track, positions: Map<string, THREE.Vector3>): THREE.Vector3 {
  return positions.get(track.id) ?? rawTrackPosition(track, "3d");
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

function createNodeSignature(tracks: Track[], viewMode: ViewMode): string {
  return tracks
    .map(
      (track) =>
        `${viewMode}:${track.id}:${formatCoord(track.x)}:${formatCoord(track.y)}:${formatCoord(track.z)}`
    )
    .join("|");
}

function createLinkSignature(tracks: Track[], selectedId: string | null): string {
  const selected = tracks.find((track) => track.id === selectedId);
  if (!selected) {
    return "";
  }
  const tracksById = new Map(tracks.map((track) => [track.id, track]));
  const similar = selected.similar
    .slice(0, PRIMARY_SIMILAR_LIMIT)
    .map((track) => `${track.id}:${track.score.toFixed(4)}`)
    .join(",");
  const primaryTracks = selected.similar
    .slice(0, PRIMARY_SIMILAR_LIMIT)
    .map((track) => tracksById.get(track.id))
    .filter((track): track is Track => Boolean(track));
  const secondary = createSecondarySimilarityLinks(selected, primaryTracks, tracksById)
    .map((link) => `${link.source.id}>${link.target.id}`)
    .join(",");
  return `${selected.id}:${similar}:${secondary}`;
}

function formatCoord(value: number | null): string {
  return value === null ? "" : value.toFixed(4);
}

function createScoreSprite(score: number): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 96;

  const context = canvas.getContext("2d");
  if (context) {
    const percent = `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "rgba(5, 5, 5, 0.82)";
    context.roundRect(28, 14, 200, 68, 18);
    context.fill();
    context.strokeStyle = "rgba(255, 255, 255, 0.22)";
    context.lineWidth = 3;
    context.stroke();
    context.fillStyle = "#ffffff";
    context.font = "700 42px Arial, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(percent, canvas.width / 2, canvas.height / 2 + 2);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.95, 0.36, 1);
  return sprite;
}

function createSimilarityLine(
  start: THREE.Vector3,
  end: THREE.Vector3,
  dashed: boolean
): THREE.Group {
  const direction = end.clone().sub(start);
  const length = direction.length();
  const group = new THREE.Group();
  if (length < 0.001) {
    return group;
  }

  const radius = dashed ? SECONDARY_LINE_RADIUS : PRIMARY_LINE_RADIUS;
  const geometry = createLineGeometry(length, radius);
  const material = createShimmerLineMaterial(dashed);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(start).addScaledVector(direction, 0.5);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize()
  );
  mesh.renderOrder = dashed ? 0 : 1;
  group.add(mesh);

  return group;
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

function createShimmerLineMaterial(dashed: boolean): ShimmerLineMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uBaseColor: { value: new THREE.Color(dashed ? "#a8a8a8" : "#d7d7d7") },
      uShimmerColor: { value: new THREE.Color("#ffffff") },
      uOpacity: { value: dashed ? 0.42 : 0.78 },
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

function collectShimmerMaterials(group: THREE.Group): ShimmerLineMaterial[] {
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

function createSecondarySimilarityLinks(
  selected: Track,
  primaryTracks: Track[],
  tracksById: Map<string, Track>
): Array<{ source: Track; target: Track }> {
  const links: Array<{ source: Track; target: Track }> = [];
  const seenPairs = new Set<string>();

  for (const source of primaryTracks) {
    let addedForSource = 0;
    for (const similar of source.similar) {
      if (similar.id === selected.id || similar.id === source.id) {
        continue;
      }

      const target = tracksById.get(similar.id);
      if (!target) {
        continue;
      }

      const key = [source.id, target.id].sort().join(":");
      if (seenPairs.has(key)) {
        continue;
      }

      seenPairs.add(key);
      links.push({ source, target });
      addedForSource += 1;
      if (addedForSource === SECONDARY_SIMILAR_LIMIT) {
        break;
      }
    }
  }

  return links;
}

function disposeGroup(group: THREE.Group, options: { disposeTextures?: boolean } = {}): void {
  const disposeTextures = options.disposeTextures ?? true;
  group.traverse((object) => {
    const disposable = object as THREE.Mesh | THREE.Line | THREE.Sprite;
    disposable.geometry?.dispose();

    const material = disposable.material;
    if (Array.isArray(material)) {
      for (const item of material) {
        disposeMaterial(item, disposeTextures);
      }
    } else if (material) {
      disposeMaterial(material, disposeTextures);
    }
  });
  group.clear();
}

function disposeMaterial(material: THREE.Material, disposeTexture: boolean): void {
  const textureMaterial = material as THREE.Material & { map?: THREE.Texture };
  if (disposeTexture) {
    textureMaterial.map?.dispose();
  }
  material.dispose();
}
