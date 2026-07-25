"""Sensor rig (Appendix B, Table B-1): three front RGB cameras stitched to a
704 x 160 strip, one 32-beam LiDAR, IMU/speedometer and GNSS.
Requires the `carla` client package matching the simulator build (0.9.14).
"""
import math

import numpy as np

CAMERAS = [
    # name, yaw (deg); each camera: 60 deg FOV, 256x160, cropped/stitched
    ("cam_left", -55.0), ("cam_front", 0.0), ("cam_right", 55.0),
]
CAM_W, CAM_H, CAM_FOV = 320, 160, 60
STITCH_W = 704


def sensor_blueprints(world, with_depth=False):
    """`with_depth` additionally spawns one depth camera per RGB camera, at the
    same pose and with the same intrinsics, to provide the privileged depth
    target of the auxiliary head (only needed during data collection)."""
    bp = world.get_blueprint_library()
    out = []
    for name, yaw in CAMERAS:
        cam = bp.find("sensor.camera.rgb")
        cam.set_attribute("image_size_x", str(CAM_W))
        cam.set_attribute("image_size_y", str(CAM_H))
        cam.set_attribute("fov", str(CAM_FOV))
        out.append((name, cam, dict(x=1.3, z=2.3, yaw=yaw)))
    if with_depth:
        for name, yaw in CAMERAS:
            dep = bp.find("sensor.camera.depth")
            dep.set_attribute("image_size_x", str(CAM_W))
            dep.set_attribute("image_size_y", str(CAM_H))
            dep.set_attribute("fov", str(CAM_FOV))
            out.append((name + "_depth", dep, dict(x=1.3, z=2.3, yaw=yaw)))
    lid = bp.find("sensor.lidar.ray_cast")
    lid.set_attribute("channels", "32")
    lid.set_attribute("range", "45")
    lid.set_attribute("rotation_frequency", "10")
    lid.set_attribute("points_per_second", "600000")
    out.append(("lidar", lid, dict(x=0.0, z=2.5)))
    imu = bp.find("sensor.other.imu")
    out.append(("imu", imu, dict(x=0.0, z=0.0)))
    gnss = bp.find("sensor.other.gnss")
    gnss.set_attribute("noise_lat_stddev", "0.0000045")   # ~0.5 m
    gnss.set_attribute("noise_lon_stddev", "0.0000045")
    out.append(("gnss", gnss, dict(x=0.0, z=0.0)))
    return out


def spawn_rig(world, vehicle, callback, with_depth=False):
    """Spawn all sensors attached to `vehicle`; `callback(name, data)`."""
    import carla
    actors = []
    for name, bp, pos in sensor_blueprints(world, with_depth):
        tf = carla.Transform(
            carla.Location(x=pos.get("x", 0.0), z=pos.get("z", 0.0)),
            carla.Rotation(yaw=pos.get("yaw", 0.0)))
        actor = world.spawn_actor(bp, tf, attach_to=vehicle)
        actor.listen(lambda data, n=name: callback(n, data))
        actors.append(actor)
    return actors


def stitch_cameras(images):
    """images: dict name -> H x W x 3 uint8. Crop overlap and stitch to
    704 x 160 (132 deg total field of view)."""
    crop = (3 * CAM_W - STITCH_W) // 4        # symmetric overlap crop
    left = images["cam_left"][:, : CAM_W - crop]
    front = images["cam_front"][:, crop // 2: CAM_W - crop + crop // 2]
    right = images["cam_right"][:, crop:]
    strip = np.concatenate([left, front, right], axis=1)
    return strip[:, :STITCH_W]


def carla_image_to_array(image):
    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    a = a.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    return a.copy()


def carla_depth_to_array(image, max_range=100.0):
    """CARLA depth image -> H x W metric depth in metres (clipped)."""
    a = np.frombuffer(image.raw_data, dtype=np.uint8)
    a = a.reshape((image.height, image.width, 4))[:, :, :3].astype(np.float32)
    # official CARLA decoding: normalized = (R + G*256 + B*256^2) / (256^3 - 1)
    norm = (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / 16777215.0
    return np.clip(norm * 1000.0, 0.0, max_range)


def depth_to_normalized_inverse(depth_m, near=1.0, far=100.0):
    """Metric depth -> inverse depth normalized to [0, 1] (the target of the
    auxiliary head, Eq. 3.22).  Inverse depth is used because it distributes
    resolution towards the near field, which is what matters for driving."""
    d = np.clip(depth_m, near, far)
    return (1.0 / d - 1.0 / far) / (1.0 / near - 1.0 / far)


def carla_lidar_to_array(meas):
    """CARLA lidar -> N x 4 (x fwd, y left, z up, intensity) in ego frame."""
    pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4).copy()
    pts[:, 1] *= -1.0                          # UE4 left-handed -> right-handed
    return pts


def transform_to_ego(points_world, ego_transform):
    """World -> ego frame for privileged BEV ground truth."""
    yaw = math.radians(ego_transform.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    dx = points_world[:, 0] - ego_transform.location.x
    dy = points_world[:, 1] - ego_transform.location.y
    x = c * dx + s * dy
    y = -(-s * dx + c * dy)                    # to (x fwd, y left)
    return np.stack([x, y], axis=1)
