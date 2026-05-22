import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  CAMERA_2D_DISTANCE,
  CAMERA_3D_POSITION,
  NODE_SIZE,
  PRIMARY_SIMILAR_LIMIT,
} from "./scene/constants";
import { disposeGroup } from "./scene/dispose";
import { createDisplayPositions, trackPosition } from "./scene/layout";
import { createSecondarySimilarityLinks } from "./scene/links";
import { createScoreSprite } from "./scene/scoreSprite";
import { createLinkSignature, createNodeSignature } from "./scene/signatures";
import { collectShimmerMaterials, createSimilarityLine } from "./scene/similarityLine";
import { disposeUnusedTextures, getTexture } from "./scene/textures";
import type { NodeRecord, ScoreLabelRecord, ShimmerLineMaterial } from "./scene/types";
import type { SimilarTrack, Track, ViewMode } from "./types";

type Props = {
  tracks: Track[];
  selectedId: string | null;
  viewMode: ViewMode;
  linkSourceId: string | null;
  similarLinks: SimilarTrack[] | null;
  highlightedLinkId: string | null;
  focusedTrackIds: string[] | null;
  onSelect: (track: Track) => void;
};

export default function MusicScene({
  tracks,
  selectedId,
  viewMode,
  linkSourceId,
  similarLinks,
  highlightedLinkId,
  focusedTrackIds,
  onSelect,
}: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const onSelectRef = useRef(onSelect);
  const tracksRef = useRef(tracks);
  const selectedRef = useRef(selectedId);
  const linkSourceRef = useRef(linkSourceId);
  const similarLinksRef = useRef(similarLinks);
  const highlightedLinkRef = useRef(highlightedLinkId);
  const focusedTrackIdsRef = useRef(focusedTrackIds);
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
    linkSourceRef.current = linkSourceId;
    rebuildSceneRef.current?.();
  }, [linkSourceId]);

  useEffect(() => {
    similarLinksRef.current = similarLinks;
    rebuildSceneRef.current?.();
  }, [similarLinks]);

  useEffect(() => {
    highlightedLinkRef.current = highlightedLinkId;
    rebuildSceneRef.current?.();
  }, [highlightedLinkId]);

  useEffect(() => {
    focusedTrackIdsRef.current = focusedTrackIds;
  }, [focusedTrackIds]);

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
          const material = new THREE.MeshBasicMaterial({
            map: texture,
            side: THREE.DoubleSide,
            transparent: true,
          });
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

      const nextLinkSignature = createLinkSignature(
        readyTracks,
        selectedRef.current,
        linkSourceRef.current,
        similarLinksRef.current,
        highlightedLinkRef.current
      );
      if (!nodesChanged && nextLinkSignature === linkSignature) {
        return;
      }

      disposeGroup(lineGroup);
      disposeGroup(labelGroup);
      scoreLabelRecords.length = 0;
      shimmerMaterials.length = 0;

      const source = readyTracks.find(
        (track) => track.id === (linkSourceRef.current ?? selectedRef.current)
      );
      if (source) {
        const readyTracksById = new Map(readyTracks.map((track) => [track.id, track]));
        const primaryLinks = similarLinksRef.current ?? source.similar;
        const related = primaryLinks
          .slice(0, PRIMARY_SIMILAR_LIMIT)
          .map((similar) => ({
            score: similar.score,
            track: readyTracksById.get(similar.id)
          }))
          .filter((similar): similar is { score: number; track: Track } => Boolean(similar.track));

        for (const target of related) {
          const line = createSimilarityLine(
            trackPosition(source, displayPositions),
            trackPosition(target.track, displayPositions),
            target.track.id === highlightedLinkRef.current ? "next" : "primary"
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

        if (!similarLinksRef.current) {
          for (const link of createSecondarySimilarityLinks(
            source,
            related.map((item) => item.track),
            readyTracksById
          )) {
            const line = createSimilarityLine(
              trackPosition(link.source, displayPositions),
              trackPosition(link.target, displayPositions),
              "secondary"
            );
            lineGroup.add(line);
            shimmerMaterials.push(...collectShimmerMaterials(line));
          }
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
        const isFocused =
          !focusedTrackIdsRef.current || focusedTrackIdsRef.current.includes(record.track.id);
        const material = record.mesh.material as THREE.MeshBasicMaterial;
        material.opacity = isFocused ? 1 : 0.18;
        record.mesh.renderOrder = isFocused ? 1 : 0;
        record.mesh.scale.setScalar(isSelected ? 1.18 : isFocused ? 1 : 0.82);
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
