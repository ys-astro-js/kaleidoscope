import * as THREE from "three";

export function disposeGroup(
  group: THREE.Group,
  options: { disposeTextures?: boolean } = {}
): void {
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
