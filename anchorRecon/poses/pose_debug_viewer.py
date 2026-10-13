import base64
import hashlib
import importlib
import math
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from dataloaders.read_write_model import qvec2rotmat, read_model, write_model


_VISER_SERVER_CREATION_LOCK = threading.Lock()
_VISER_ADAPTIVE_DPR_PATTERN = re.compile(
    rb"window\.devicePixelRatio\s*\*\s*\(\s*(?:0?\.)2\s*\+\s*"
    rb"(?:0?\.)8\s*\*\s*[$A-Za-z_][$A-Za-z0-9_]*\s*\)"
)
_VISER_CANVAS_DPR_PATTERN = re.compile(
    rb"(jsxRuntimeExports\.jsxs\(Canvas,\{)camera:"
)
_VISER_ADAPTIVE_DPR_NODE_PATTERN = re.compile(
    rb",jsxRuntimeExports\.jsx\(AdaptiveDpr,\{\}\)"
)
_POSE_DEBUG_CONTENT_ROOT = "/pose_debug/content"
_POSE_DEBUG_TRANSFORM_GIZMO = "/pose_debug/transform_gizmo"


def _prepare_fixed_dpr_viser_client(viser):
    """Create a temporary viser client with adaptive resolution disabled.

    viser 0.2.7 hard-codes its live-canvas resolution controller in the
    compiled browser bundle and exposes no Python setting for it. Patch only a
    temporary copy of that bundle so the installed package remains untouched.
    """

    client_autobuild = importlib.import_module("viser._client_autobuild")
    client_autobuild.ensure_client_is_built()

    source_root = Path(viser.__file__).resolve().parent / "client" / "build"
    if not (source_root / "index.html").is_file():
        raise RuntimeError(
            f"Could not find the viser browser client at {source_root}."
        )

    temporary_dir = tempfile.TemporaryDirectory(
        prefix="skyanchor-viser-fixed-dpr-"
    )
    custom_root = Path(temporary_dir.name) / "build"
    shutil.copytree(source_root, custom_root)

    patched_bundle = None
    patched_payload = None
    for bundle_path in sorted((custom_root / "assets").glob("index-*.js")):
        payload = bundle_path.read_bytes()
        fixed_payload, expression_count = _VISER_ADAPTIVE_DPR_PATTERN.subn(
            b"1", payload
        )
        if expression_count == 0:
            continue
        fixed_payload, canvas_count = _VISER_CANVAS_DPR_PATTERN.subn(
            rb"\g<1>dpr:1,camera:", fixed_payload
        )
        fixed_payload, monitor_count = _VISER_ADAPTIVE_DPR_NODE_PATTERN.subn(
            b"", fixed_payload
        )
        if (expression_count, canvas_count, monitor_count) != (1, 1, 1):
            temporary_dir.cleanup()
            raise RuntimeError(
                "Could not disable viser's adaptive DPR safely: expected one "
                "controller expression, one Canvas, and one monitor node; found "
                f"{expression_count}, {canvas_count}, and {monitor_count}."
            )
        if patched_bundle is not None:
            temporary_dir.cleanup()
            raise RuntimeError(
                "Could not disable viser's adaptive DPR safely: multiple browser "
                "entry bundles matched."
            )
        patched_bundle = bundle_path
        patched_payload = fixed_payload

    if patched_bundle is None or patched_payload is None:
        temporary_dir.cleanup()
        raise RuntimeError(
            "Could not disable viser's adaptive DPR safely: the resolution "
            "controller was not found."
        )

    # Give the patched asset a new URL so a browser cannot reuse the original
    # bundle from its cache after the pose-debug process is restarted.
    digest = hashlib.sha256(patched_payload).hexdigest()[:12]
    fixed_bundle = patched_bundle.with_name(
        f"{patched_bundle.stem}-fixed-dpr-{digest}{patched_bundle.suffix}"
    )
    fixed_bundle.write_bytes(patched_payload)

    index_path = custom_root / "index.html"
    index_html = index_path.read_text(encoding="utf-8")
    if index_html.count(patched_bundle.name) != 1:
        temporary_dir.cleanup()
        raise RuntimeError(
            "Could not update the viser client entry point for fixed resolution."
        )
    index_path.write_text(
        index_html.replace(patched_bundle.name, fixed_bundle.name),
        encoding="utf-8",
    )
    return temporary_dir, custom_root


def _start_fixed_dpr_viser_server(viser, **server_kwargs):
    """Start Viser against the fixed-resolution temporary browser client."""

    temporary_dir, client_root = _prepare_fixed_dpr_viser_client(viser)
    viser_impl = importlib.import_module(viser.ViserServer.__module__)

    # Viser 0.2.7 hard-codes its HTTP root inside ViserServer. Replace the
    # constructor only for the duration of this server creation; the websocket
    # server retains the resolved path after we restore the package global.
    with _VISER_SERVER_CREATION_LOCK:
        original_websock_server = viser_impl.infra.WebsockServer

        def fixed_client_websock_server(*args, **kwargs):
            kwargs["http_server_root"] = client_root
            return original_websock_server(*args, **kwargs)

        viser_impl.infra.WebsockServer = fixed_client_websock_server
        try:
            server = viser.ViserServer(**server_kwargs)
        finally:
            viser_impl.infra.WebsockServer = original_websock_server

    print(
        "[viser] adaptive resolution disabled "
        "(fixed DPR = 1.0; camera controls preserved)"
    )
    return server, temporary_dir


class _MarkdownImageHandle:
    """Compatibility image handle for viser releases before GuiApi.add_image."""

    def __init__(self, gui, image, label: str, jpeg_quality: int, visible: bool):
        self._label = label
        self._jpeg_quality = jpeg_quality
        # Supplying an image root avoids a spurious warning in viser 0.2.x when
        # the markdown image already uses a self-contained data URL.
        self._markdown = gui.add_markdown("", image_root=os.getcwd(), visible=visible)
        self.image = image

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        self._image = np.ascontiguousarray(image)
        success, encoded = cv2.imencode(
            ".jpg",
            # OpenCV encodes BGR, while viewer images are RGB.
            cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not success:
            raise RuntimeError("Failed to encode the viser match image as JPEG.")
        payload = base64.b64encode(encoded).decode("ascii")
        self._markdown.content = (
            f"**{self._label}**\n\n"
            f"![{self._label}](data:image/jpeg;base64,{payload})"
        )


class ViserPoseDebugger:
    def __init__(self, width: int, height: int, focal: float, args):
        try:
            import viser
            from viser import transforms as tf
        except ImportError as exc:
            raise ImportError(
                "Viser pose debug viewer requires the `viser` package. "
                "Install it in the active environment first."
            ) from exc

        self.width = width
        self.height = height
        self.focal = float(focal)
        self.max_points = args.pose_debug_max_points
        self.match_step = max(1, args.pose_debug_match_step)
        self.frustum_scale = args.pose_debug_frustum_scale
        self.ba_frustum_scale = (
            float(args.pose_debug_ba_frustum_scale)
            if args.pose_debug_ba_frustum_scale > 0
            else self.frustum_scale
        )
        self.gt_frustum_scale = (
            float(args.pose_debug_gt_frustum_scale)
            if args.pose_debug_gt_frustum_scale > 0
            else self.frustum_scale
        )
        self.ba_point_size = float(args.pose_debug_ba_point_size)
        self.gt_point_size = float(args.pose_debug_gt_point_size)
        self.pose_scale = float(getattr(args, "scene_scale", 1.0))
        self._tf = tf
        self._scene_names: dict[str, set[str]] = {}
        self._scene_handles = {}
        self._last_ba_points = None
        self._accumulated_ba_groups = None
        self._gt_model = None
        # viser 0.2.x does not expose dynamic setters for frustum scale or point
        # size. Keep the exact CPU payload sent to the browser so GUI sliders can
        # recreate only the affected nodes without retaining GPU keyframes.
        self._camera_specs = {}
        self._point_specs = {}
        self._scene_lock = threading.RLock()
        self._redraw_timers = {}
        self._scene_transform_pivot = np.zeros(3, dtype=np.float32)
        self._scene_transform_pivot_initialized = False
        self._scene_transform_has_pose_pivot = False
        self._scene_transform_last_online_count = None

        self.server, self._viser_client_temporary_dir = _start_fixed_dpr_viser_server(
            viser,
            host=args.pose_debug_host,
            port=args.pose_debug_port,
            label="Pose Debug",
        )
        # This viser release exposes background images rather than a solid-color
        # setter. A tiny lossless black image gives every client a pure-black
        # scene background and is retained for clients that connect later.
        self.server.scene.set_background_image(
            np.zeros((2, 2, 3), dtype=np.uint8),
            format="png",
        )
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid(
            "/world/grid",
            width=10,
            height=10,
            cell_size=0.1,
            cell_thickness=0.25,
            section_size=1.0,
            section_thickness=0.6,
        )
        # Keep the transform gizmo separate from the content root. Hiding the
        # gizmo therefore never hides the scene, while its callback applies the
        # same rigid transform to every pose, trajectory, and point-cloud node.
        self._scene_content = self.server.scene.add_frame(
            _POSE_DEBUG_CONTENT_ROOT,
            show_axes=False,
        )
        self.scene_transform_gizmo = self.server.scene.add_transform_controls(
            _POSE_DEBUG_TRANSFORM_GIZMO,
            scale=90.0,
            line_width=3.0,
            fixed=True,
            auto_transform=True,
            depth_test=False,
            opacity=0.9,
        )

        self.server.gui.add_markdown(
            "Pose debug viewer\n\n"
            "- White: online poses\n"
            "- Green: BA poses\n"
            "- Red: anchor poses\n"
            "- Dark red: GT poses\n"
            "- Yellow/green points: PnP candidates/inliers\n"
            "- Dense BA points: RGB keyframe points\n"
            "- Candidate tracks: all final-BA candidates\n"
            "- Selected anchor-connected tracks\n"
            "- Selected non-anchor tracks\n"
            "- Pose trajectory lines: hidden by default\n"
            "- View controls: left-drag rotate, right/middle-drag pan, wheel zoom"
        )
        self.show_online_poses = self.server.gui.add_checkbox("Online poses", True)
        self.show_ba_poses = self.server.gui.add_checkbox("BA poses", True)
        self.show_anchor_poses = self.server.gui.add_checkbox("Anchor poses", True)
        self.show_pose_trajectories = self.server.gui.add_checkbox(
            "Pose trajectories", False
        )
        self.show_pnp_points = self.server.gui.add_checkbox("PnP 3D points", True)
        self.show_ba_points = self.server.gui.add_checkbox("Dense BA points", True)
        self.show_ba_candidate_tracks = self.server.gui.add_checkbox("Candidate tracks", False)
        self.show_ba_selected_anchor_tracks = self.server.gui.add_checkbox("Selected anchor tracks", True)
        self.show_ba_selected_non_anchor_tracks = self.server.gui.add_checkbox("Selected non-anchor tracks", True)
        self.show_gt_poses = self.server.gui.add_checkbox("GT poses", True)
        self.show_gt_points = self.server.gui.add_checkbox("GT sparse points", False)
        self.show_match_image = self.server.gui.add_checkbox("Match image", True)
        with self.server.gui.add_folder(
            "Scene transform", expand_by_default=True
        ):
            self.server.gui.add_markdown(
                "Drag the RGB arrows to translate and the RGB rings to rotate. "
                "This changes visualization only; pose/BA/GT data stay unchanged. "
                "The grid stays fixed as a world-frame reference."
            )
            self.show_scene_transform_gizmo = self.server.gui.add_checkbox(
                "Show transform gizmo", True
            )
            self.reset_scene_transform_button = self.server.gui.add_button(
                "Reset scene transform"
            )
        with self.server.gui.add_folder("Display sizes", expand_by_default=True):
            self.ba_point_size_slider = self.server.gui.add_slider(
                "BA point size",
                min=0.001,
                max=max(0.2, self.ba_point_size * 4.0),
                step=0.001,
                initial_value=self.ba_point_size,
                hint="Controls dense BA points and final-BA track groups.",
            )
            self.gt_point_size_slider = self.server.gui.add_slider(
                "GT point size",
                min=0.001,
                max=max(0.2, self.gt_point_size * 4.0),
                step=0.001,
                initial_value=self.gt_point_size,
            )
            self.frustum_scale_slider = self.server.gui.add_slider(
                "Online/anchor frustum scale",
                min=0.01,
                max=max(2.0, self.frustum_scale * 4.0),
                step=0.01,
                initial_value=self.frustum_scale,
            )
            self.ba_frustum_scale_slider = self.server.gui.add_slider(
                "BA frustum scale",
                min=0.01,
                max=max(2.0, self.ba_frustum_scale * 4.0),
                step=0.01,
                initial_value=self.ba_frustum_scale,
            )
            self.gt_frustum_scale_slider = self.server.gui.add_slider(
                "GT frustum scale",
                min=0.01,
                max=max(2.0, self.gt_frustum_scale * 4.0),
                step=0.01,
                initial_value=self.gt_frustum_scale,
            )
        self._register_gui_callbacks()
        self.status = self.server.gui.add_markdown("Waiting for pose data.")
        initial_match_image = np.zeros((64, 128, 3), dtype=np.uint8)
        if hasattr(self.server.gui, "add_image"):
            self.match_image = self.server.gui.add_image(
                initial_match_image,
                label="Latest matched keypoints",
                jpeg_quality=80,
                visible=True,
            )
        else:
            self.match_image = _MarkdownImageHandle(
                self.server.gui,
                initial_match_image,
                label="Latest matched keypoints",
                jpeg_quality=80,
                visible=True,
            )

        print(
            "Pose debug viewer listening at "
            f"http://{args.pose_debug_host}:{args.pose_debug_port}"
        )
        if args.pose_debug_gt_colmap_dir:
            self.load_gt_colmap(args.pose_debug_gt_colmap_dir)

    def _register_gui_callbacks(self):
        @self.scene_transform_gizmo.on_update
        def _(handle):
            self._apply_scene_transform(handle.wxyz, handle.position)

        @self.show_scene_transform_gizmo.on_update
        def _(_event):
            self.scene_transform_gizmo.visible = bool(
                self.show_scene_transform_gizmo.value
            )
            self.server.flush()

        @self.reset_scene_transform_button.on_click
        def _(_event):
            self._reset_scene_transform()

        @self.show_online_poses.on_update
        def _(_event):
            self._set_groups_visible(("online",), self.show_online_poses.value)

        @self.show_ba_poses.on_update
        def _(_event):
            self._set_groups_visible(("ba",), self.show_ba_poses.value)

        @self.show_anchor_poses.on_update
        def _(_event):
            self._set_groups_visible(("anchors",), self.show_anchor_poses.value)

        @self.show_gt_poses.on_update
        def _(_event):
            self._set_groups_visible(("gt_poses",), self.show_gt_poses.value)

        @self.show_pose_trajectories.on_update
        def _(_event):
            self._refresh_trajectory_visibility()

        @self.show_pnp_points.on_update
        def _(_event):
            self._set_groups_visible(
                ("pnp_candidates", "pnp_inliers"),
                self.show_pnp_points.value,
            )

        @self.show_ba_points.on_update
        def _(_event):
            self._set_groups_visible(
                ("ba_points_dense", "ba_points"),
                self.show_ba_points.value,
            )

        @self.show_ba_candidate_tracks.on_update
        def _(_event):
            self._set_groups_visible(
                ("ba_candidate_tracks",),
                self.show_ba_candidate_tracks.value,
            )

        @self.show_ba_selected_anchor_tracks.on_update
        def _(_event):
            self._set_groups_visible(
                ("ba_selected_anchor_tracks",),
                self.show_ba_selected_anchor_tracks.value,
            )

        @self.show_ba_selected_non_anchor_tracks.on_update
        def _(_event):
            self._set_groups_visible(
                ("ba_selected_non_anchor_tracks",),
                self.show_ba_selected_non_anchor_tracks.value,
            )

        @self.show_gt_points.on_update
        def _(_event):
            self._set_groups_visible(("gt_points",), self.show_gt_points.value)

        @self.ba_point_size_slider.on_update
        def _(event):
            self.ba_point_size = float(event.target.value)
            self._schedule_redraw(
                "ba_point_size",
                lambda: self._redraw_point_groups(
                    {
                        "online_points",
                        "ba_points_dense",
                        "ba_points",
                        "ba_candidate_tracks",
                        "ba_selected_anchor_tracks",
                        "ba_selected_non_anchor_tracks",
                    }
                ),
            )

        @self.gt_point_size_slider.on_update
        def _(event):
            self.gt_point_size = float(event.target.value)
            self._schedule_redraw(
                "gt_point_size",
                lambda: self._redraw_point_groups({"gt_points"}),
            )

        @self.frustum_scale_slider.on_update
        def _(event):
            self.frustum_scale = float(event.target.value)
            self._schedule_redraw(
                "frustum_scale",
                lambda: self._redraw_camera_groups({"online", "anchors", "pnp"}),
            )

        @self.ba_frustum_scale_slider.on_update
        def _(event):
            self.ba_frustum_scale = float(event.target.value)
            self._schedule_redraw(
                "ba_frustum_scale",
                lambda: self._redraw_camera_groups({"ba"}),
            )

        @self.gt_frustum_scale_slider.on_update
        def _(event):
            self.gt_frustum_scale = float(event.target.value)
            self._schedule_redraw(
                "gt_frustum_scale",
                lambda: self._redraw_camera_groups({"gt_poses"}),
            )

    def _update_scene_transform_pivot(self, group: str, centres):
        """Keep the gizmo centred on the active submap without moving content."""

        centre_array = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
        centre_array = centre_array[np.all(np.isfinite(centre_array), axis=1)]
        if centre_array.shape[0] == 0:
            return

        with self._scene_lock:
            should_update = False
            if group == "gt_poses":
                # GT provides a useful initial location while waiting for the
                # online bootstrap, but must not override a pose-based pivot.
                should_update = not self._scene_transform_pivot_initialized
            elif group in {"online", "ba", "anchors"}:
                if not self._scene_transform_has_pose_pivot:
                    should_update = True
                    self._scene_transform_has_pose_pivot = True
                if group == "online":
                    previous_count = self._scene_transform_last_online_count
                    current_count = int(centre_array.shape[0])
                    # Keyframe count grows within a submap and drops back to the
                    # bootstrap set at the next one. Re-centre only at that
                    # boundary so online updates do not fight an active drag.
                    if (
                        previous_count is not None
                        and current_count < previous_count
                    ):
                        should_update = True
                    self._scene_transform_last_online_count = current_count

            if not should_update:
                return

            pivot = np.median(centre_array, axis=0).astype(np.float32)
            content_wxyz = np.asarray(
                self._scene_content.wxyz, dtype=np.float64
            ).reshape(4)
            content_wxyz /= max(float(np.linalg.norm(content_wxyz)), 1e-12)
            content_rotation = self._tf.SO3(content_wxyz).as_matrix()
            content_position = np.asarray(
                self._scene_content.position, dtype=np.float64
            ).reshape(3)
            gizmo_position = content_rotation @ pivot + content_position

            self._scene_transform_pivot = pivot
            self._scene_transform_pivot_initialized = True
            with self.server.atomic():
                self.scene_transform_gizmo.wxyz = content_wxyz.astype(np.float32)
                self.scene_transform_gizmo.position = gizmo_position.astype(
                    np.float32
                )
        self.server.flush()

    def _apply_scene_transform(self, wxyz, gizmo_position):
        """Apply the gizmo's rigid transform to content about its scene pivot."""

        quaternion = np.asarray(wxyz, dtype=np.float64).reshape(4)
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-12:
            return
        quaternion /= quaternion_norm
        position = np.asarray(gizmo_position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(position)):
            return

        rotation = self._tf.SO3(quaternion).as_matrix()
        pivot = self._scene_transform_pivot.astype(np.float64)
        content_position = position - rotation @ pivot
        with self._scene_lock, self.server.atomic():
            self._scene_content.wxyz = quaternion.astype(np.float32)
            self._scene_content.position = content_position.astype(np.float32)
        self.server.flush()

    def _reset_scene_transform(self):
        """Restore the visualization transform while retaining the chosen pivot."""

        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with self._scene_lock, self.server.atomic():
            self.scene_transform_gizmo.wxyz = identity
            self.scene_transform_gizmo.position = self._scene_transform_pivot
            self._scene_content.wxyz = identity
            self._scene_content.position = np.zeros(3, dtype=np.float32)
        self.server.flush()

    def _schedule_redraw(self, key: str, callback, delay: float = 0.12):
        """Coalesce slider events so dragging does not flood the websocket."""
        with self._scene_lock:
            previous = self._redraw_timers.pop(key, None)
            if previous is not None:
                previous.cancel()

            timer = None

            def run():
                with self._scene_lock:
                    if self._redraw_timers.get(key) is not timer:
                        return
                    self._redraw_timers.pop(key, None)
                    try:
                        callback()
                        self.server.flush()
                    except Exception as exc:
                        print(f"[pose_debug] GUI redraw failed ({key}): {exc}")

            timer = threading.Timer(delay, run)
            timer.daemon = True
            self._redraw_timers[key] = timer
            timer.start()

    def _set_groups_visible(self, groups, visible: bool):
        with self._scene_lock, self.server.atomic():
            for group in groups:
                for name in tuple(self._scene_names.get(group, set())):
                    handle = self._scene_handles.get(name)
                    if handle is None:
                        continue
                    is_trajectory = name.endswith("/trajectory")
                    node_visible = bool(visible) and (
                        not is_trajectory or self.show_pose_trajectories.value
                    )
                    handle.visible = node_visible
                    if name in self._camera_specs:
                        self._camera_specs[name]["visible"] = node_visible
                    if name in self._point_specs:
                        self._point_specs[name]["visible"] = node_visible
        self.server.flush()

    def _refresh_trajectory_visibility(self):
        group_visibility = {
            "online": self.show_online_poses.value,
            "ba": self.show_ba_poses.value,
            "anchors": self.show_anchor_poses.value,
            "gt_poses": self.show_gt_poses.value,
        }
        with self._scene_lock, self.server.atomic():
            for group, group_visible in group_visibility.items():
                trajectory_name = f"{_POSE_DEBUG_CONTENT_ROOT}/{group}/trajectory"
                handle = self._scene_handles.get(trajectory_name)
                if handle is not None:
                    handle.visible = bool(
                        group_visible and self.show_pose_trajectories.value
                    )
        self.server.flush()

    def _camera_scale_for_group(self, group: str) -> float:
        if group == "ba":
            return self.ba_frustum_scale
        if group == "gt_poses":
            return self.gt_frustum_scale
        if group == "pnp":
            return self.frustum_scale * 1.4
        return self.frustum_scale

    def _pose_visibility_for_group(self, group: str, fallback: bool) -> bool:
        controls = {
            "online": self.show_online_poses,
            "ba": self.show_ba_poses,
            "anchors": self.show_anchor_poses,
            "gt_poses": self.show_gt_poses,
        }
        control = controls.get(group)
        return bool(control.value) if control is not None else bool(fallback)

    def _point_visibility_for_group(self, group: str, fallback: bool) -> bool:
        controls = {
            "pnp_candidates": self.show_pnp_points,
            "pnp_inliers": self.show_pnp_points,
            "ba_points_dense": self.show_ba_points,
            "ba_points": self.show_ba_points,
            "ba_candidate_tracks": self.show_ba_candidate_tracks,
            "ba_selected_anchor_tracks": self.show_ba_selected_anchor_tracks,
            "ba_selected_non_anchor_tracks": self.show_ba_selected_non_anchor_tracks,
            "gt_points": self.show_gt_points,
        }
        control = controls.get(group)
        return bool(control.value) if control is not None else bool(fallback)

    def _point_size_for_group(self, group: str) -> float:
        multipliers = {
            "online_points": 2.0 / 3.0,
            "ba_points_dense": 2.0 / 3.0,
            "ba_points": 1.0,
            "ba_candidate_tracks": 0.75,
            "ba_selected_anchor_tracks": 4.0 / 3.0,
            "ba_selected_non_anchor_tracks": 7.0 / 6.0,
        }
        if group == "gt_points":
            return max(self.gt_point_size, 1e-4)
        return self._ba_point_size(multipliers.get(group, 1.0))

    def _redraw_camera_groups(self, groups):
        with self._scene_lock, self.server.atomic():
            for name, spec in tuple(self._camera_specs.items()):
                group = spec.get("group")
                if group not in groups:
                    continue
                old_handle = self._scene_handles.get(name)
                if old_handle is not None:
                    old_handle.remove()
                scale = self._camera_scale_for_group(group)
                handle = self.server.scene.add_camera_frustum(
                    name,
                    fov=spec["fov"],
                    aspect=spec["aspect"],
                    scale=scale,
                    color=spec["color"],
                    wxyz=spec["wxyz"],
                    position=spec["position"],
                    visible=spec["visible"],
                )
                spec["scale"] = scale
                self._scene_handles[name] = handle

    def _redraw_point_groups(self, groups):
        with self._scene_lock, self.server.atomic():
            for name, spec in tuple(self._point_specs.items()):
                group = spec.get("group")
                if group not in groups:
                    continue
                old_handle = self._scene_handles.get(name)
                if old_handle is not None:
                    old_handle.remove()
                point_size = self._point_size_for_group(group)
                handle = self.server.scene.add_point_cloud(
                    name,
                    spec["points"],
                    spec["colors"],
                    point_size=point_size,
                    point_shape="circle",
                    visible=spec["visible"],
                )
                spec["point_size"] = point_size
                self._scene_handles[name] = handle

    def update_keyframes(self, keyframes, focal, stage: str = "online", label: str = ""):
        if len(keyframes) == 0:
            return

        self.focal = self._as_float(focal, self.focal)
        visible = (
            self.show_ba_poses.value if stage == "ba" else self.show_online_poses.value
        )
        color = (80, 255, 120) if stage == "ba" else (245, 245, 245)
        self._draw_keyframe_set(stage, keyframes, self.focal, color, visible)

        anchor_items = [
            (keyframe, keyframe.info["anchor_Rt"])
            for keyframe in keyframes
            if keyframe.info.get("anchor_Rt", None) is not None
        ]
        self._draw_pose_items(
            "anchors",
            anchor_items,
            self.focal,
            (255, 80, 80),
            visible=self.show_anchor_poses.value,
            frustum_scale=self.frustum_scale,
        )

        if stage == "online":
            self._draw_points_from_keyframes(
                "online_points",
                keyframes,
                (255, 220, 40),
                visible=False,
            )

        suffix = f" ({label})" if label else ""
        self.status.content = (
            f"{stage} poses: {len(keyframes)} keyframes, focal={self.focal:.2f}{suffix}"
        )
        self.server.flush()

    def update_ba(self, keyframes, focal, points=None, label: str = ""):
        self.update_keyframes(keyframes, focal, stage="ba", label=label)
        self._draw_points_from_keyframes(
            "ba_points_dense",
            keyframes,
            (80, 220, 255),
            visible=self.show_ba_points.value,
            use_keypoint_colors=True,
        )
        reuse_cached_points = points == "keep"
        if reuse_cached_points:
            points = self._last_ba_points
        if points is None:
            self._clear_group("ba_points")
            self._clear_group("ba_candidate_tracks")
            self._clear_group("ba_selected_anchor_tracks")
            self._clear_group("ba_selected_non_anchor_tracks")
        else:
            point_colors = None
            point_xyz = points
            groups = None
            if isinstance(points, dict):
                point_xyz = points.get("xyz", None)
                point_colors = points.get("rgb", None)
                groups = points.get("groups", None)
            if groups is not None:
                # "keep" is a redraw request for the already accumulated CPU
                # payload. Merging it again would duplicate every track group
                # after each submap and grow memory exponentially.
                if not reuse_cached_points:
                    groups = self._accumulate_ba_groups(groups)
                self._last_ba_points = {"groups": groups}
                self._clear_group("ba_points")
                self._draw_point_payload(
                    "ba_candidate_tracks",
                    groups.get("candidate_tracks"),
                    (255, 210, 40),
                    self.show_ba_candidate_tracks.value,
                    self._ba_point_size(0.75),
                )
                self._draw_point_payload(
                    "ba_selected_anchor_tracks",
                    groups.get("selected_anchor_connected_tracks"),
                    (80, 255, 120),
                    self.show_ba_selected_anchor_tracks.value,
                    self._ba_point_size(4.0 / 3.0),
                )
                self._draw_point_payload(
                    "ba_selected_non_anchor_tracks",
                    groups.get("selected_non_anchor_tracks"),
                    (255, 120, 80),
                    self.show_ba_selected_non_anchor_tracks.value,
                    self._ba_point_size(7.0 / 6.0),
                )
            else:
                self._last_ba_points = points
                self._draw_points(
                    "ba_points",
                    point_xyz,
                    (80, 220, 255),
                    visible=self.show_ba_points.value,
                    point_size=self._ba_point_size(),
                    colors=point_colors,
                )
                self._clear_group("ba_candidate_tracks")
                self._clear_group("ba_selected_anchor_tracks")
                self._clear_group("ba_selected_non_anchor_tracks")
        self.server.flush()

    def load_gt_colmap(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"GT COLMAP directory not found: {model_dir}")

        cameras, images, points3D = read_model(model_dir)
        scale = float(self.pose_scale)
        gt_images = []
        for _, image in sorted(images.items()):
            camera = cameras[image.camera_id]
            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = qvec2rotmat(image.qvec).astype(np.float32)
            Rt[:3, 3] = image.tvec.astype(np.float32) * np.float32(scale)
            gt_images.append(
                {
                    "name": image.name,
                    "Rt": Rt,
                    "focal": self._camera_focal(camera),
                    "width": int(camera.width),
                    "height": int(camera.height),
                }
            )

        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        if points3D:
            xyz = np.stack([point.xyz for _, point in sorted(points3D.items())]).astype(np.float32) * np.float32(scale)
            rgb = np.stack([point.rgb for _, point in sorted(points3D.items())]).astype(np.uint8)

        self._gt_model = {
            "dir": model_dir,
            "scale": scale,
            "cameras": cameras,
            "raw_images": images,
            "raw_points3D": points3D,
            "images": gt_images,
            "points_xyz": xyz,
            "points_rgb": rgb,
        }
        if scale != 1.0:
            print(
                f"[pose_debug] Loaded GT COLMAP from {model_dir} with scene_scale={scale:g} "
                f"({len(gt_images)} cams, {len(xyz)} points)"
            )
        self._draw_gt_model()

    def export_scaled_gt_colmap(self, out_dir: str, ext: str = ".txt"):
        """
        Write a rescaled copy of the loaded GT COLMAP model to ``out_dir``.

        - tvec scaled by ``self.pose_scale``
        - 3D point xyz scaled by ``self.pose_scale``
        - cameras / 2D observations / track ids unchanged

        No-op if ``self._gt_model`` is None or ``self.pose_scale == 1.0``.
        """
        if self._gt_model is None:
            return None
        scale = float(self._gt_model.get("scale", self.pose_scale))
        if scale == 1.0:
            return None
        cameras = self._gt_model.get("cameras")
        raw_images = self._gt_model.get("raw_images")
        raw_points3D = self._gt_model.get("raw_points3D")
        if cameras is None or raw_images is None or raw_points3D is None:
            return None

        os.makedirs(out_dir, exist_ok=True)

        scaled_images = {}
        for image_id, image in raw_images.items():
            scaled_images[image_id] = image._replace(
                tvec=np.asarray(image.tvec, dtype=np.float64) * scale
            )

        scaled_points3D = {}
        for point_id, point in raw_points3D.items():
            scaled_points3D[point_id] = point._replace(
                xyz=np.asarray(point.xyz, dtype=np.float64) * scale
            )

        write_model(cameras, scaled_images, scaled_points3D, out_dir, ext=ext)
        print(
            f"[pose_debug] Exported scaled GT COLMAP (scale={scale:g}) to {out_dir} "
            f"({len(scaled_images)} images, {len(scaled_points3D)} points)"
        )
        return out_dir

    def _accumulate_ba_groups(self, groups):
        if groups is None:
            return self._accumulated_ba_groups
        if self._accumulated_ba_groups is None:
            self._accumulated_ba_groups = {}
        merged = {}
        for key in [
            "candidate_tracks",
            "selected_anchor_connected_tracks",
            "selected_non_anchor_tracks",
        ]:
            merged[key] = self._merge_point_payload(
                self._accumulated_ba_groups.get(key),
                groups.get(key) if isinstance(groups, dict) else None,
            )
        self._accumulated_ba_groups = merged
        return self._accumulated_ba_groups

    def _merge_point_payload(self, current, new_payload):
        if new_payload is None:
            return current

        new_xyz = self._payload_tensor(new_payload.get("xyz", None), torch.float32)
        if new_xyz is None or new_xyz.numel() == 0:
            return current
        new_rgb = self._payload_tensor(new_payload.get("rgb", None), torch.uint8)

        if current is None:
            return {"xyz": new_xyz, "rgb": new_rgb}

        merged_xyz = torch.cat([current["xyz"], new_xyz], dim=0)
        current_rgb = current.get("rgb", None)
        if current_rgb is None or new_rgb is None:
            merged_rgb = None
        else:
            merged_rgb = torch.cat([current_rgb, new_rgb], dim=0)
        return {"xyz": merged_xyz, "rgb": merged_rgb}

    def _payload_tensor(self, value, dtype):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    def update_incremental(
        self,
        label,
        curr_image,
        ref_keyframe,
        matches,
        candidate_points=None,
        inlier_points=None,
        pnp_Rt=None,
        refined_Rt=None,
        focal=None,
    ):
        self.focal = self._as_float(focal, self.focal)
        self._clear_group("pnp_candidates")
        self._clear_group("pnp_inliers")

        if self.show_match_image.value and ref_keyframe is not None and matches is not None:
            self.match_image.image = self._make_match_image(
                curr_image,
                ref_keyframe.image_pyr[0],
                matches,
            )

        if candidate_points is not None:
            self._draw_points(
                "pnp_candidates",
                candidate_points,
                (255, 210, 40),
                visible=self.show_pnp_points.value,
                point_size=0.018,
            )
        if inlier_points is not None:
            self._draw_points(
                "pnp_inliers",
                inlier_points,
                (80, 255, 120),
                visible=self.show_pnp_points.value,
                point_size=0.024,
            )
        pose_items = []
        if pnp_Rt is not None:
            pose_items.append((f"{label}/pnp", pnp_Rt, (255, 160, 40)))
        if refined_Rt is not None:
            pose_items.append((f"{label}/miniba", refined_Rt, (80, 255, 120)))
        with self._scene_lock:
            self._clear_group("pnp")
            for name, Rt, color in pose_items:
                scene_name = (
                    f"{_POSE_DEBUG_CONTENT_ROOT}/pnp/{self._safe_name(name)}"
                )
                handle = self._add_camera(
                    scene_name,
                    Rt,
                    self.focal,
                    color=color,
                    visible=True,
                    scale=self.frustum_scale * 1.4,
                )
                self._remember_scene_handle("pnp", scene_name, handle)

        n_matches = 0 if matches is None else len(matches.kpts)
        n_candidates = self._point_count(candidate_points)
        n_inliers = self._point_count(inlier_points)
        self.status.content = (
            f"latest PnP: {label}, matches={n_matches}, "
            f"candidate_3d={n_candidates}, inlier_3d={n_inliers}, "
            f"focal={self.focal:.2f}"
        )
        self.server.flush()

    def _draw_keyframe_set(self, group: str, keyframes, focal, color, visible: bool):
        items = [(keyframe, keyframe.get_Rt()) for keyframe in keyframes]
        frustum_scale = self.ba_frustum_scale if group == "ba" else self.frustum_scale
        self._draw_pose_items(group, items, focal, color, visible, frustum_scale)

    def _draw_pose_items(self, group: str, items, focal, color, visible: bool, frustum_scale: float):
        with self._scene_lock:
            self._draw_pose_items_locked(
                group,
                items,
                focal,
                color,
                visible,
                frustum_scale,
            )

    def _draw_pose_items_locked(
        self,
        group: str,
        items,
        focal,
        color,
        visible: bool,
        frustum_scale: float,
    ):
        # GUI callbacks can run while pose matrices are prepared. Resolve the
        # final visibility and scale only after taking the scene lock.
        visible = self._pose_visibility_for_group(group, visible)
        if group in {"online", "ba", "anchors", "gt_poses", "pnp"}:
            frustum_scale = self._camera_scale_for_group(group)
        self._clear_group(group)
        centres = []
        for i, (keyframe_or_name, Rt) in enumerate(items):
            if hasattr(keyframe_or_name, "info"):
                label = keyframe_or_name.info.get("name", str(keyframe_or_name.index))
            else:
                label = str(keyframe_or_name)
            name = (
                f"{_POSE_DEBUG_CONTENT_ROOT}/{group}/"
                f"cam_{i:04d}_{self._safe_name(label)}"
            )
            handle = self._add_camera(name, Rt, focal, color, visible, frustum_scale)
            self._remember_scene_handle(group, name, handle)
            centres.append(self._camera_centre(Rt))

        self._update_scene_transform_pivot(group, centres)
        if len(centres) >= 2:
            points = np.stack(centres).astype(np.float32)
            path_name = f"{_POSE_DEBUG_CONTENT_ROOT}/{group}/trajectory"
            handle = self._add_trajectory(
                path_name,
                points,
                color,
                visible and self.show_pose_trajectories.value,
            )
            self._remember_scene_handle(group, path_name, handle)

    def _draw_gt_model(self):
        with self._scene_lock:
            self._draw_gt_model_locked()

    def _draw_gt_model_locked(self):
        if self._gt_model is None:
            self._clear_group("gt_poses")
            self._clear_group("gt_points")
            return

        self._clear_group("gt_poses")
        centres = []
        for image_idx, image_info in enumerate(self._gt_model["images"]):
            name = (
                f"{_POSE_DEBUG_CONTENT_ROOT}/gt_poses/"
                f"cam_{image_idx:04d}_{self._safe_name(image_info['name'])}"
            )
            handle = self._add_camera(
                name,
                image_info["Rt"],
                image_info["focal"],
                color=(210, 40, 40),
                visible=self.show_gt_poses.value,
                scale=self.gt_frustum_scale,
                width=image_info["width"],
                height=image_info["height"],
            )
            self._remember_scene_handle("gt_poses", name, handle)
            centres.append(self._camera_centre(image_info["Rt"]))

        self._update_scene_transform_pivot("gt_poses", centres)
        if len(centres) >= 2:
            points = np.stack(centres).astype(np.float32)
            path_name = f"{_POSE_DEBUG_CONTENT_ROOT}/gt_poses/trajectory"
            handle = self._add_trajectory(
                path_name,
                points,
                (210, 40, 40),
                self.show_gt_poses.value and self.show_pose_trajectories.value,
            )
            self._remember_scene_handle("gt_poses", path_name, handle)

        self._draw_points(
            "gt_points",
            self._gt_model["points_xyz"],
            (210, 40, 40),
            visible=self.show_gt_points.value,
            point_size=self.gt_point_size,
            colors=self._gt_model["points_rgb"],
        )

    def _draw_points_from_keyframes(
        self,
        group: str,
        keyframes,
        color,
        visible: bool,
        use_keypoint_colors: bool = False,
    ):
        pts = []
        colors = []
        for keyframe in keyframes:
            desc_kpts = keyframe.desc_kpts
            if desc_kpts is None or not hasattr(desc_kpts, "pts3d"):
                continue
            mask = desc_kpts.has_pt3d
            if hasattr(desc_kpts, "pts_conf"):
                mask = mask & (desc_kpts.pts_conf > 0)
            if mask.any():
                pts.append(desc_kpts.pts3d[mask].detach().cpu())
                if use_keypoint_colors:
                    colors.append(self._sample_keypoint_colors(keyframe, mask).detach().cpu())
        if len(pts) == 0:
            self._clear_group(group)
            return
        self._draw_points(
            group,
            torch.cat(pts, dim=0),
            color,
            visible,
            self._ba_point_size(2.0 / 3.0),
            colors=torch.cat(colors, dim=0) if colors else None,
        )

    def _sample_keypoint_colors(self, keyframe, mask):
        image = self._tensor_to_numpy(keyframe.image_pyr[0])
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0.0, 1.0)
        if image.shape[-1] == 4:
            image = image[..., :3]
        kpts = self._tensor_to_numpy(keyframe.desc_kpts.kpts[mask])
        xy = np.round(kpts).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, image.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, image.shape[0] - 1)
        sampled = image[xy[:, 1], xy[:, 0]]
        return torch.from_numpy((sampled * 255.0).round().astype(np.uint8))

    def _draw_points(
        self,
        group: str,
        points,
        color,
        visible: bool,
        point_size: float,
        colors=None,
    ):
        with self._scene_lock:
            self._draw_points_locked(
                group,
                points,
                color,
                visible,
                point_size,
                colors,
            )

    def _draw_points_locked(
        self,
        group: str,
        points,
        color,
        visible: bool,
        point_size: float,
        colors=None,
    ):
        # Use the latest GUI values rather than the possibly stale values that
        # were captured before CPU point preparation completed.
        visible = self._point_visibility_for_group(group, visible)
        if group in {
            "online_points",
            "ba_points_dense",
            "ba_points",
            "ba_candidate_tracks",
            "ba_selected_anchor_tracks",
            "ba_selected_non_anchor_tracks",
            "gt_points",
        }:
            point_size = self._point_size_for_group(group)
        self._clear_group(group)
        pts = self._tensor_to_numpy(points)
        if pts.size == 0:
            return
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if len(pts) == 0:
            return
        point_colors = None
        if colors is not None:
            point_colors = self._tensor_to_numpy(colors)
            if point_colors.ndim == 1:
                point_colors = np.repeat(point_colors[None], len(pts), axis=0)
            point_colors = point_colors[valid]
        if len(pts) > self.max_points:
            idx = np.linspace(0, len(pts) - 1, self.max_points).astype(np.int64)
            pts = pts[idx]
            if point_colors is not None:
                point_colors = point_colors[idx]
        if point_colors is None:
            point_colors = np.array(color, dtype=np.uint8)
        else:
            point_colors = np.clip(point_colors, 0, 255).astype(np.uint8)
        name = f"{_POSE_DEBUG_CONTENT_ROOT}/{group}/points"
        handle = self.server.scene.add_point_cloud(
            name,
            pts.astype(np.float32),
            point_colors,
            point_size=point_size,
            point_shape="circle",
            visible=visible,
        )
        self._remember_scene_handle(group, name, handle)
        self._point_specs[name] = {
            "group": group,
            "points": np.ascontiguousarray(pts.astype(np.float32)),
            "colors": np.ascontiguousarray(point_colors),
            "point_size": float(point_size),
            "visible": bool(visible),
        }

    def _draw_point_payload(self, group: str, payload, color, visible: bool, point_size: float):
        if payload is None:
            self._clear_group(group)
            return
        point_xyz = payload.get("xyz", None) if isinstance(payload, dict) else payload
        point_colors = payload.get("rgb", None) if isinstance(payload, dict) else None
        self._draw_points(
            group,
            point_xyz,
            color,
            visible,
            point_size,
            colors=point_colors,
        )

    def _add_camera(self, name, Rt, focal, color, visible: bool, scale: float, width=None, height=None):
        width = self.width if width is None else width
        height = self.height if height is None else height
        position, wxyz = self._camera_position_wxyz(Rt)
        fov = 2.0 * math.atan(height / (2.0 * max(float(focal), 1e-6)))
        aspect = width / height
        handle = self.server.scene.add_camera_frustum(
            name,
            fov=fov,
            aspect=aspect,
            scale=scale,
            color=color,
            wxyz=wxyz,
            position=position,
            visible=visible,
        )
        self._camera_specs[name] = {
            "group": None,
            "fov": float(fov),
            "aspect": float(aspect),
            "scale": float(scale),
            "color": tuple(int(channel) for channel in color),
            "wxyz": np.asarray(wxyz, dtype=np.float32).copy(),
            "position": np.asarray(position, dtype=np.float32).copy(),
            "visible": bool(visible),
        }
        return handle

    def _add_trajectory(self, name, points, color, visible):
        if hasattr(self.server.scene, "add_line_segments"):
            segments = np.stack([points[:-1], points[1:]], axis=1)
            return self.server.scene.add_line_segments(
                name,
                segments,
                np.array(color, dtype=np.uint8),
                line_width=2.0,
                visible=visible,
            )
        return self.server.scene.add_spline_catmull_rom(
            name,
            points,
            line_width=2.0,
            color=color,
            visible=visible,
        )

    def _remember_scene_handle(self, group, name, handle):
        self._scene_names.setdefault(group, set()).add(name)
        self._scene_handles[name] = handle
        if name in self._camera_specs:
            self._camera_specs[name]["group"] = group
        if name in self._point_specs:
            self._point_specs[name]["group"] = group

    def _camera_position_wxyz(self, Rt):
        Rt_np = self._matrix_to_numpy(Rt)
        c2w = np.linalg.inv(Rt_np)
        rot = self._tf.SO3.from_matrix(c2w[:3, :3])
        return c2w[:3, 3].astype(np.float32), rot.wxyz.astype(np.float32)

    def _camera_centre(self, Rt):
        Rt_np = self._matrix_to_numpy(Rt)
        return np.linalg.inv(Rt_np)[:3, 3].astype(np.float32)

    def _make_match_image(self, curr_image, ref_image, matches):
        curr = self._image_to_uint8(curr_image)
        ref = self._image_to_uint8(ref_image)
        if curr.shape[0] != ref.shape[0]:
            new_w = int(ref.shape[1] * curr.shape[0] / ref.shape[0])
            ref = cv2.resize(ref, (new_w, curr.shape[0]), interpolation=cv2.INTER_AREA)
        canvas = np.ascontiguousarray(np.concatenate([curr, ref], axis=1))

        mkpts_curr = self._tensor_to_numpy(matches.kpts)
        mkpts_ref = self._tensor_to_numpy(matches.kpts_other)
        if len(mkpts_curr) > 0:
            step = max(self.match_step, int(math.ceil(len(mkpts_curr) / 800)))
            ref_offset = curr.shape[1]
            for p_curr, p_ref in zip(mkpts_curr[::step], mkpts_ref[::step]):
                p0 = tuple(np.round(p_curr).astype(int))
                p1 = tuple(np.round(p_ref + np.array([ref_offset, 0])).astype(int))
                color = (80, 255, 120)
                cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, p0, 3, (255, 210, 40), -1, cv2.LINE_AA)
                cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)

        max_width = 1600
        if canvas.shape[1] > max_width:
            scale = max_width / canvas.shape[1]
            canvas = cv2.resize(
                canvas,
                (max_width, int(canvas.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return canvas

    def _image_to_uint8(self, image):
        arr = self._tensor_to_numpy(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    def _clear_group(self, group: str):
        with self._scene_lock:
            for name in self._scene_names.get(group, set()):
                handle = self._scene_handles.pop(name, None)
                if handle is not None:
                    handle.remove()
                elif hasattr(self.server.scene, "remove_by_name"):
                    self.server.scene.remove_by_name(name)
                self._camera_specs.pop(name, None)
                self._point_specs.pop(name, None)
            self._scene_names[group] = set()

    def _matrix_to_numpy(self, value):
        arr = self._tensor_to_numpy(value).astype(np.float64)
        if arr.shape == (3, 4):
            Rt = np.eye(4, dtype=np.float64)
            Rt[:3, :] = arr
            return Rt
        return arr

    def _tensor_to_numpy(self, value):
        if value is None:
            return np.empty((0, 3), dtype=np.float32)
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy()
        return np.asarray(value)

    def _as_float(self, value, default):
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            return float(value.detach().flatten()[0].cpu().item())
        return float(value)

    def _camera_focal(self, camera):
        params = np.asarray(camera.params, dtype=np.float64)
        if camera.model == "PINHOLE":
            return float((params[0] + params[1]) * 0.5)
        return float(params[0])

    def _ba_point_size(self, multiplier: float = 1.0):
        return max(self.ba_point_size * float(multiplier), 1e-4)

    def _point_count(self, value):
        if value is None:
            return 0
        return int(self._tensor_to_numpy(value).shape[0])

    def _safe_name(self, value) -> str:
        return (
            str(value)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
