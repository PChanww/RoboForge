export interface Joint {
  name: string
  child: string
  type: 'fixed' | 'revolute' | 'prismatic'
  axis: number[]
  range: number[] | null
  damping: number
  armature: number
  initial: number
}

export interface Collision {
  name?: string | null
  type: string
  size: number[]
  fromto?: number[] | null
  pos: number[]
  mass_kg: number
  friction: number[]
  contype: number
  conaffinity: number
  rgba: number[]
  group: number
}

export interface Visual {
  mesh: string
  pos: number[]
  quat: number[]
  scale: number[]
  rgba: number[]
}

export interface Link {
  name: string
  parent: string | null
  origin: number[]
  gravcomp: boolean
  visuals: Visual[]
  collisions: Collision[]
}

export interface Actuator {
  joint: string
  kind: string
  kp: number
  kv: number
  force_limit: number
}

export interface Robot {
  id: string
  root_link: string
  links: Link[]
  joints: Joint[]
  actuators: Actuator[]
  end_effector: string
  tool_offset_m: number
  gripper: { left_joint: string; right_joint: string; open_m: number; closed_m: number } | null
  mesh_dir: string
}

export interface Sensor {
  id: string
  type: string
  parent: string
  position: number[]
  xyaxes: number[]
  intrinsics: { fx: number; fy: number; cx: number; cy: number }
  resolution: number[]
  fps: number
  latency_s: number
  noise_std: number
}

export interface Controller {
  id: string
  mode: string
  action_names: string[]
  action_dim: number
  units: string[]
  frequency_hz: number
  action_range: number[][]
}

export interface SceneCollider {
  name: string
  shape: 'box' | 'plane' | 'cylinder'
  full_size_m: number[]
  position: number[]
  quaternion_xyzw: number[]
  enabled: boolean
  friction: number[]
  contype: number
  conaffinity: number
  tag: string
}

export interface ProjectConfig {
  schema_version: string
  project_id: string
  version: number
  validation_status: string
  scene: { physics_dt: number; seed: number; gravity: number[]; use_default_table?: boolean;
    sensor_renderer?: 'mujoco' | 'hybrid'; work_surface_z?: number; colliders?: SceneCollider[];
    environment?: { asset_id: string; revision: number; position: number[]; quaternion_xyzw: number[]; uniform_scale: number } | null }
  robots: Robot[]
  sensors: Sensor[]
  controllers: Controller[]
  task: { id: string; asset: string; instruction: string; goal_xy: number[]; time_limit_s: number }
}

export interface Frame {
  frame_id?: number
  capture_time: number
  available_time: number
  width: number
  height: number
  jpeg_b64: string
}

export interface CameraStatus {
  state: string
  frame_id: number | null
  capture_time: number | null
  available_time: number | null
  age_s: number | null
  configured_fps: number
  actual_capture_fps: number
  pending_frames: number
  evicted_frames: number
  queue_len: number
}

export interface Observation {
  schema_version: string
  robot: string
  controller: string
  control_mode: string
  config_version: number
  tick: number
  sim_time: number
  seed: number
  state: {
    time: number
    joint_pos: Record<string, number>
    joint_vel: Record<string, number>
    eef_position: number[]
    gripper: number | null
    body_poses: Record<string, { position: number[]; matrix: number[][] }>
    object_pose: { position: number[]; matrix: number[][] }
  }
  images: Record<string, Frame | null>
  camera_status?: Record<string, CameraStatus>
  epoch?: number
  timing: {
    image_age_s: Record<string, number>
    capture_count: Record<string, number>
    actual_capture_intervals_s: Record<string, number[]>
    control_period_s: number
    camera_period_s: Record<string, number>
  }
  instruction: string
}

export type Selection =
  | { kind: 'robot'; id: string }
  | { kind: 'link'; id: string }
  | { kind: 'joint'; id: string }
  | { kind: 'sensor'; id: string }
  | { kind: 'task' }
  | null
