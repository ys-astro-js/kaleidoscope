import * as THREE from "three";

export function createScoreSprite(score: number): THREE.Sprite {
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
