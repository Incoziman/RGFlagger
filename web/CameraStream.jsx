// Drop this file into any React app (Vite, Next.js, CRA, ...).
//
//   import { CameraStream, CameraGrid } from "./CameraStream";
//   <CameraStream name="camera1" />
//   <CameraGrid />
//
// The video comes from the camera relay (`python -m cameras relay`),
// so no password is ever needed in the browser.

// Change this to the PC that runs the relay.
export const RELAY_URL = "http://localhost:8889";

export const CAMERAS = ["camera1", "camera2", "camera3", "camera4"];

export function CameraStream({ name, relay = RELAY_URL, style, ...props }) {
  return (
    <iframe
      title={name}
      src={`${relay}/${name}?controls=false&muted=true&autoplay=true&playsinline=true`}
      allow="autoplay; fullscreen"
      style={{ border: 0, width: "100%", aspectRatio: "16 / 9", background: "#000", ...style }}
      {...props}
    />
  );
}

export function CameraGrid({ cameras = CAMERAS, relay = RELAY_URL }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
      {cameras.map((name) => (
        <CameraStream key={name} name={name} relay={relay} />
      ))}
    </div>
  );
}
