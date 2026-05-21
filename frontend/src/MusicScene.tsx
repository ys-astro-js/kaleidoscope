import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { artUrl } from "./api";
import type { Track } from "./types";

type Props = {
  tracks: Track[];
  selectedId: string | null;
  onSelect: (track: Track) => void;
};

type NodeRecord = {
  track: Track;
  mesh: THREE.Mesh;
};

export default function MusicScene({ tracks, selectedId, onSelect }: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const onSelectRef = useRef(onSelect);
  const tracksRef = useRef(tracks);
  const selectedRef = useRef(selectedId);

  useEffect(() => {
    onSelectRef.current = onSelect;
    tracksRef.current = tracks;
    selectedRef.current = selectedId;
  }, [onSelect, selectedId, tracks]);

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
    controls.enablePan = false;
    controls.minDistance = 4;
    controls.maxDistance = 26;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const textureLoader = new THREE.TextureLoader();
    const nodeGroup = new THREE.Group();
    const lineGroup = new THREE.Group();
    const nodeRecords: NodeRecord[] = [];
    let animationFrame = 0;
    scene.add(nodeGroup, lineGroup);

    const resize = () => {
      const { width, height } = mount.getBoundingClientRect();
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };

    const rebuild = () => {
      nodeGroup.clear();
      lineGroup.clear();
      nodeRecords.length = 0;

      const readyTracks = tracksRef.current.filter(
        (track) => track.status === "ready" && track.x !== null && track.y !== null && track.z !== null
      );

      for (const track of readyTracks) {
        const texture = textureLoader.load(artUrl(track.id));
        texture.colorSpace = THREE.SRGBColorSpace;
        const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.1), material);
        mesh.position.set(track.x ?? 0, track.y ?? 0, track.z ?? 0);
        mesh.userData.trackId = track.id;
        nodeGroup.add(mesh);
        nodeRecords.push({ track, mesh });
      }

      const selected = readyTracks.find((track) => track.id === selectedRef.current);
      if (selected) {
        const related = readyTracks.filter((track) => selected.similar.includes(track.id));
        for (const target of related) {
          const geometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(selected.x ?? 0, selected.y ?? 0, selected.z ?? 0),
            new THREE.Vector3(target.x ?? 0, target.y ?? 0, target.z ?? 0)
          ]);
          const material = new THREE.LineBasicMaterial({ color: "#8a8a8a", transparent: true, opacity: 0.65 });
          lineGroup.add(new THREE.Line(geometry, material));
        }
      }
    };

    const handleClick = (event: MouseEvent) => {
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
        record.mesh.lookAt(camera.position);
        const isSelected = record.track.id === selectedRef.current;
        record.mesh.scale.setScalar(isSelected ? 1.18 : 1);
      }
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };

    resize();
    rebuild();
    animate();

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    renderer.domElement.addEventListener("click", handleClick);

    const interval = window.setInterval(rebuild, 1000);
    return () => {
      cancelAnimationFrame(animationFrame);
      window.clearInterval(interval);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("click", handleClick);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  return <div className="scene" ref={mountRef} />;
}
