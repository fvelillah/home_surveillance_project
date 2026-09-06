"""ONVIF discovery and stream URI extraction for Dahua NVR."""

import argparse
import sys
from onvif import ONVIFCamera
import cv2


def test_onvif_connection(ip: str, port: int, user: str, password: str):
    print(f"Connecting to ONVIF service on {ip}:{port} with user '{user}'...")
    try:
        # Initialize ONVIF Camera/NVR client
        mycam = ONVIFCamera(ip, port, user, password)
        
        # Get Device Information
        dev_info = mycam.devicemgmt.GetDeviceInformation()
        print("\n✅ Conectado exitosamente al NVR Dahua:")
        print(f"   - Fabricante: {dev_info.Manufacturer}")
        print(f"   - Modelo:     {dev_info.Model}")
        print(f"   - Firmware:   {dev_info.FirmwareVersion}")
        print(f"   - Serial:     {dev_info.SerialNumber}")

        # Initialize Media Service
        media = mycam.create_media_service()
        profiles = media.GetProfiles()
        print(f"\n✅ Perfiles de Video Encontrados: {len(profiles)}")

        streams = []
        for i, profile in enumerate(profiles, start=1):
            token = profile.token
            name = getattr(profile, 'Name', f'Profile_{i}')
            
            # Request stream URI for this profile
            obj = media.create_type('GetStreamUri')
            obj.StreamSetup = {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            }
            obj.ProfileToken = token
            uri_info = media.GetStreamUri(obj)
            uri = uri_info.Uri
            
            # Inject credentials if needed
            if "@" not in uri and user and password:
                parts = uri.split("://")
                uri = f"{parts[0]}://{user}:{password}@{parts[1]}"

            print(f"\n🎥 [Perfil {i}] {name} (Token: {token})")
            print(f"   URL RTSP Generada: {uri}")
            streams.append((name, uri))

        if streams:
            print("\n" + "="*60)
            print("🧪 Probando captura del primer fotograma en vivo...")
            first_name, first_uri = streams[0]
            cap = cv2.VideoCapture(first_uri, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"🎉 ¡ÉXITO TOTAL! Fotograma capturado de [{first_name}]: {w}x{h} píxeles.")
                else:
                    print("⚠️ Se abrió el flujo pero no se pudo leer el fotograma.")
                cap.release()
            else:
                print("❌ No se pudo abrir el stream con OpenCV.")
            print("="*60)

    except Exception as e:
        print(f"\n❌ Error al conectar por ONVIF: {e}")
        print("\nConsejo: Verifica que ONVIF esté activado en el NVR:")
        print("  1. Entra a http://<IP_NVR> en tu navegador")
        print("  2. Ve a Ajustes > Red > Acceso a plataforma / ONVIF")
        print("  3. Activa la casilla ONVIF y añade un usuario ONVIF")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dahua NVR ONVIF Probe")
    parser.add_argument("--ip", default="192.168.1.126", help="NVR IP Address")
    parser.add_argument("--port", type=int, default=80, help="ONVIF HTTP Port (default: 80)")
    parser.add_argument("--user", default="admin", help="NVR Username")
    parser.add_argument("--password", default="abc12345", help="NVR Password")
    args = parser.parse_args()

    test_onvif_connection(args.ip, args.port, args.user, args.password)
